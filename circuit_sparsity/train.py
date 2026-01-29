"""
Ultra-minimalistic implementation of a training script for a sparse transformer model from Gao et al. 2025.

Modify this script to fit your needs (your own dataset).

This script can train the models from the paper, albeit slowly.
"""

import json
import os
import random
from dataclasses import asdict
from functools import partial, wraps
from typing import Literal
import blobfile as bf

import numpy as np
import torch
import torch.nn.functional as F
from circuit_sparsity.inference.gpt import ( 
    GPT,
    GPTConfig,
)
from tqdm import tqdm

world_size = 1
rank = 0


def topk(x, k, abs=False, dim=None):
    """
    by default gets global topk of x.
    if dim is set, gets topk along some dimension. k must be divisible by x.shape[1-dim]
    k is ALWAYS global, even when dim is set. so if you want one per dim, you need to set k=k*x.shape[1-dim]

    returns
    - vals: the k largest values (either globally or along dim)
    - inds: the indices into the flattened x
    - shape for both: [k]
    """
    if k >= x.numel():
        k = x.numel()

    topk_fn = partial(torch.topk, sorted=False)

    if dim is None:  # glboal topk
        x = x.flatten()
        topk_fn = partial(topk_fn, dim=0)
    else:
        assert dim in [0, 1]
        assert k % x.shape[1 - dim] == 0, f"{k} % {x.shape[1 - dim]} != 0"
        k //= x.shape[1 - dim]
        assert len(x.shape) == 2, "todo: generalize to higher dims"

        def _topk_fn(x, k, topk_fn):
            vals, inds = topk_fn(x, k, dim=dim)
            # flatten inds
            if dim == 0:
                inds = inds * x.shape[1] + torch.arange(
                    x.shape[1], device=x.device, dtype=inds.dtype
                ).unsqueeze(0)
            else:
                inds = (
                    inds
                    + torch.arange(x.shape[0], device=x.device, dtype=inds.dtype).unsqueeze(1)
                    * x.shape[1]
                )

            inds = inds.flatten()
            vals = vals.flatten()
            return vals, inds

        topk_fn = partial(_topk_fn, topk_fn=topk_fn)

    if abs:
        _, inds = topk_fn(x.abs(), k)
        vals = x.flatten()[inds]
    else:
        vals, inds = topk_fn(x, k)

    return vals, inds


def apply_topk_(
    model,
    *,
    pfrac,
    expansion_factor,
    expansion_factor_mlp,
    abs=True,
    achyuta_mixed_neuronwise_v2=False,
    minimum_alive_per_neuron=0,
    final_pfrac,
):
    def _lerp(a, b, frac):
        return a + (b - a) * frac

    _maybe_adjust_pfrac_embbias = lambda x: _lerp(
        final_pfrac,
        expansion_factor,
        (x - final_pfrac) / (expansion_factor * expansion_factor_mlp - final_pfrac),
    )
    for pn, p in model.named_parameters():
        if (
            len(p.shape) > 1 and not p.is_sparse or "bias" in pn
        ):  # if p is sparse, then its topk is managed in CustomAdam
            if "bigram_table" in pn:
                # skip bigram table
                continue

            if model.config.cat_pos_emb and p is model.transformer.wpe.weight:
                continue

            if p is model.transformer.wte.weight:
                L0frac = pfrac_to_L0frac(
                    _maybe_adjust_pfrac_embbias(pfrac),
                    expansion_factor,
                    expansion_factor_mlp,
                    is_embed=True,
                )

            elif p is model.lm_head.weight:
                L0frac = pfrac_to_L0frac(
                    _maybe_adjust_pfrac_embbias(pfrac),
                    expansion_factor,
                    expansion_factor_mlp,
                    is_embed=True,
                )
            elif "bias" in pn:
                L0frac = pfrac_to_L0frac(
                    _maybe_adjust_pfrac_embbias(pfrac),
                    expansion_factor,
                    expansion_factor_mlp,
                    is_bias=True,
                )
            else:
                L0frac = pfrac_to_L0frac(
                    pfrac, expansion_factor, expansion_factor_mlp, is_embed=False
                )
            L0frac = min(L0frac, 1)
            k = int(L0frac * p.numel())

            if "bias" not in pn:
                dim = topk_neuronwise_dim(pn, p)

            if (
                achyuta_mixed_neuronwise_v2
                and minimum_alive_per_neuron > 0
                and ".bias"
                not in pn  # needed for sparse biases - we want to sparsify them but not do neuronwise
                and (
                    "mlp.c_fc" in pn
                    or "mlp.c_proj" in pn
                    or "attn.c_attn" in pn
                    or "attn.c_proj" in pn
                )
            ):
                assert abs
                dim = topk_neuronwise_dim(pn, p)
                # print(pn, p.data.shape)
                score = p.data.abs()
                n_neurons = p.data.shape[1 - dim]
                assert minimum_alive_per_neuron * n_neurons <= k, (
                    "ur L0 is too small for ur min alive"
                )
                # get the top minimum_alive_per_neuron values along the neuron dim for every neuron
                # our custom topk function, even when dim is set, expects k to be global (not per-dim)
                indices = topk(
                    score,
                    k=minimum_alive_per_neuron * n_neurons,
                    abs=False,
                    dim=dim,
                )[1]
                score.flatten()[indices] = 1e10

                indices = topk(score, k=k, abs=False)[1]
            else:
                indices = topk(p.data.abs(), k, abs=False)[1]

            if len(p.data.shape) == 2:
                mask = torch.ones_like(p.data.flatten(), dtype=torch.bool)
                mask.index_fill_(0, indices, 0)
                mask = mask.view_as(p.data)

                # normal codepath: only zero out the current param inds
                p.data[mask] = 0


def topk_neuronwise_dim(name, p):
    # for a parameter of a given name, figure out what dimension topk should be along to apply to neurons

    if "mlp.c_fc" in name:  # d_model -> 4 * d_model  --  shape order is (output, input)
        # assert p.shape[0] == p.shape[1] * 4, f"unexpected shape {p.shape} for {name}"
        topk_dim = 1
    elif "mlp.c_proj" in name:  # 4 * d_model -> d_model  --  shape order is (output, input)
        # assert p.shape[0] * 4 == p.shape[1], f"unexpected shape {p.shape} for {n  ame}"
        topk_dim = 0
    elif "attn.c_attn" in name:  # d_model -> 3 * d_model  --  shape order is (output, input)
        # assert p.shape[0] == p.shape[1] * 3, f"unexpected shape {p.shape} for {name}"
        topk_dim = 1
    elif "attn.c_proj" in name:  # d_model -> d_model  --  shape order is (output, input)
        # assert p.shape[0] == p.shape[1], f"unexpected shape {p.shape} for {name}"
        topk_dim = 0
    elif "wte" in name:  # vocab_size -> d_model  --  shape order is (input, output) !!!
        topk_dim = 1
        assert p.shape[0] == 50304 or p.shape[0] == 2048, f"unexpected shape {p.shape} for {name}"
    elif "wpe" in name:  # n_ctx -> d_model  --  shape order is (input, output) !!!
        topk_dim = 1
        assert p.shape[0] == 1024, f"unexpected shape {p.shape} for {name}"
    elif "lm_head.weight" in name:  # vocab_size -> d_model  --  shape order is (output, input) !!!
        topk_dim = 1
        assert p.shape[0] == 50304 or p.shape[0] == 2048, f"unexpected shape {p.shape} for {name}"
    else:
        assert len(p.shape) < 2, f"unhandled shape {p.shape} for {name}"
    return topk_dim


def batch(xs, bs):
    ret = []
    for x in xs:
        ret.append(x)
        if len(ret) == bs:
            yield ret
            ret = []


def pfrac_to_L0frac(pfrac, expansion_factor, expansion_factor_mlp, is_embed=False, is_bias=False):
    assert not (is_embed and is_bias)
    if is_embed or is_bias:
        return pfrac / expansion_factor
    else:
        return pfrac / (expansion_factor * expansion_factor_mlp)


def load_dataset(
    dataset_name,
    global_bs,
    n_ctx,
    seed: int | None = None,
):
    """
    Dummy mock dataset generator for testing. Please replace with your own dataset loader, which follows the same interface.

    Args:
        dataset_name: Name of the dataset to load.
        global_bs: Global batch size.
        n_ctx: Context length.
        seed: Random seed.

    Returns:
        _iter: Iterator over the dataset.
        test_tokens: Test tokens.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.seed()

    vocab_size = 2048

    def _iter():
        while True:
            yield torch.randint(
                0,
                vocab_size,
                (global_bs, n_ctx),
                generator=rng,
            )

    test_tokens = torch.randint(
        0,
        vocab_size,
        (global_bs, n_ctx),
        generator=rng,
    ).cuda()

    return _iter(), test_tokens


def repeat_iter_fn(f):
    @wraps(f)
    def _f(*a, **k):
        while True:
            yield from f(*a, **k)

    return _f


def _sched_fn(
    step,
    *,
    x_anneal_initial,
    x_anneal_final,
    x_anneal_exponent,
    x_anneal_start_step,
    x_anneal_stop_step,
):
    total_anneal_steps = x_anneal_stop_step - x_anneal_start_step
    sched_progress = min(1, max(0, (step - x_anneal_start_step) / total_anneal_steps))

    frac = 1 - (1 - sched_progress) ** x_anneal_exponent

    def _lerp(a, b, frac):
        return a + (b - a) * frac

    return _lerp(
        x_anneal_initial,
        x_anneal_final,
        frac,
    )


def _loglog_linear_sched_fn(
    step: int,
    *,
    x_anneal_initial_value: float,
    x_anneal_final_value: float,
    x_anneal_start_step: int,
    x_anneal_stop_step: int,
) -> float:
    s0 = float(x_anneal_start_step)
    s1 = float(x_anneal_stop_step)
    y0 = float(x_anneal_initial_value)
    y1 = float(x_anneal_final_value)

    if s1 <= s0:
        raise ValueError("x_anneal_stop_step must be > x_anneal_start_step.")
    if y0 <= 0 or y1 <= 0:
        raise ValueError("Values must be positive to appear on a log-log plot.")

    logx0 = np.log10(s0)
    logx1 = np.log10(s1)
    logy0 = np.log10(y0)
    logy1 = np.log10(y1)

    slope = (logy1 - logy0) / (logx1 - logx0)

    if step < s0:
        return y0
    elif step > s1:
        return x_anneal_final_value
    else:
        logy = slope * (np.log10(step) - logx0) + logy0
        return 10**logy


def main(
    *,
    expname: str,
    n_ctx: int,
    global_bs: int,
    vocab_size: int,
    lr: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    pfrac: float | None,
    pfrac_anneal: bool,
    pfrac_anneal_mode: Literal["linear", "power_law"],
    pfrac_anneal_exponent: int,
    pfrac_anneal_start_frac: float,
    pfrac_anneal_stop_frac: float,
    pfrac_anneal_initial: float | Literal["max"],
    pfrac_anneal_schedule_lr_with_L0: bool,
    achyuta_mixed_neuronwise_v2: bool,
    minimum_alive_per_neuron: int,
    expansion_factor: float,
    expansion_factor_mlp: float | None = None,
    lr_decay: bool,
    gradclip: bool,
    activation_type: str,
    residual_activation_type: str,
    enable_mixed: bool = True,
    dataset_name: str,
    total_tokens: int,
    d_head: int | None,
    enable_bigram_table: bool,
    lr_warmup_frac: float | None,
    n_layer: int,
    eps: float,
    afrac: float | None = None,
    afrac_loctypes: str,
    val_every: int = 20,
    attn_sink: bool = False,
    seed: int = 0,
):
    if afrac == 1:
        afrac = None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_steps = total_tokens // global_bs // n_ctx
    smol_model_config = GPTConfig(
        block_size=1024,
        vocab_size=2048,
        n_layer=4,
        n_head=8,
        d_model=256,
        dropout=0.0,
        rms_norm=True,
        residual_activation_type="identity",
        activation_type="relu",
        tied_unembed=False,
    )

    if expansion_factor_mlp is None:
        expansion_factor_mlp = expansion_factor

    if d_head is not None:
        assert (
            int(smol_model_config.n_head * expansion_factor_mlp * smol_model_config.d_head) % d_head
            == 0
        ), (
            f"{smol_model_config.n_head} * {expansion_factor_mlp} * {smol_model_config.d_head} % {d_head} != 0"
        )

    n_head = (
        int(smol_model_config.n_head * expansion_factor_mlp * smol_model_config.d_head) // d_head
        if d_head is not None
        else smol_model_config.n_head
    )
    d_head = d_head if d_head is not None else int(smol_model_config.d_head * expansion_factor)

    print(f"{n_head=}, {d_head=}")

    assert smol_model_config.vocab_size == vocab_size, "VOCAB SIZES DONT MATCH"

    d_mlp = int(smol_model_config.d_model * 4 * expansion_factor_mlp)
    beeg_config = GPTConfig(
        block_size=smol_model_config.block_size,
        vocab_size=vocab_size,
        n_layer=smol_model_config.n_layer if n_layer is None else n_layer,
        n_head=n_head,
        d_model=int(smol_model_config.d_model * expansion_factor),
        dropout=0.0,
        rms_norm=True,  # use rmsnorm
        residual_activation_type=residual_activation_type,
        activation_type=activation_type,
        debug_nans=False,
        tied_unembed=False,
        d_mlp=d_mlp,
        d_head=d_head,
        afrac=afrac,  # could get overridden by afrac_anneal
        afrac_loctypes=afrac_loctypes,
        enable_bigram_table=enable_bigram_table,
        learnable_bigram_table=enable_bigram_table,
        sink=attn_sink,
    )

    n_layer = beeg_config.n_layer

    if achyuta_mixed_neuronwise_v2:
        # TODO: write this assert
        assert pfrac * d_mlp >= minimum_alive_per_neuron, (
            f"{pfrac=}, {d_mlp=}, {minimum_alive_per_neuron=}"
        )

    beeg_model = GPT(
        beeg_config,
    ).cuda()
    beeg_model.transformer.wpe.weight.data.zero_()
    beeg_model.transformer.wpe.requires_grad = False

    if enable_bigram_table:
        beeg_model.bigram_table.data = torch.rand_like(beeg_model.bigram_table.data).cuda() * 0.02

    if pfrac_anneal_initial == "max":
        pfrac_anneal_initial = expansion_factor**2

    all_params = list(beeg_model.parameters())

    betas = (adam_beta1, adam_beta2)

    optimizer = torch.optim.Adam(
        all_params,
        lr=lr,
        eps=eps,
        betas=betas,
        fused=True,
    )

    dset_iter, my_test_xs = load_dataset(
        dataset_name,
        global_bs,
        n_ctx,
        seed=seed,
    )

    n_params = sum(p.numel() for p in beeg_model.parameters())
    n_params_wd = sum(p.numel() for p in beeg_model.parameters() if len(p.shape) > 1)

    print("n_params", n_params, "n_params_wd", n_params_wd)

    scaler = torch.cuda.amp.GradScaler(enabled=enable_mixed)
    autocast_ctx_manager = torch.cuda.amp.autocast()

    # apply mask before training as derisking for kernel stuff
    if pfrac is not None and not pfrac_anneal:
        # print("PFRAC", pfrac)
        # assert dense_embeddings
        apply_topk_(
            beeg_model,
            pfrac=pfrac,
            expansion_factor=expansion_factor,
            expansion_factor_mlp=expansion_factor_mlp,
            achyuta_mixed_neuronwise_v2=achyuta_mixed_neuronwise_v2,
            minimum_alive_per_neuron=minimum_alive_per_neuron,
            final_pfrac=pfrac,
        )

    lr_warmup_steps = int(lr_warmup_frac * total_steps) if lr_warmup_frac is not None else 0

    def iter_is_final(it):
        sentinel = object()
        last = sentinel
        for x in it:
            if last is not sentinel:
                yield last, False
            last = x
        yield last, True

    pbar = tqdm(total=total_steps, desc="Training")
    for (i, x), _is_final in iter_is_final(enumerate(dset_iter)):
        # lr sched - linear decay
        if lr_decay:
            my_lr = lr * min(1, 1 - (i - lr_warmup_steps) / (total_steps - lr_warmup_steps))
        else:
            my_lr = lr

        if lr_warmup_frac is not None and i < lr_warmup_steps:
            my_lr = lr * i / lr_warmup_steps

        if pfrac_anneal:
            match pfrac_anneal_mode:
                case "linear":
                    this_pfrac = _sched_fn(
                        i,
                        x_anneal_initial=pfrac_anneal_initial,
                        x_anneal_final=pfrac,
                        x_anneal_exponent=pfrac_anneal_exponent,
                        x_anneal_start_step=pfrac_anneal_start_frac * total_steps,
                        x_anneal_stop_step=pfrac_anneal_stop_frac * total_steps,
                    )
                case "power_law":
                    this_pfrac = _loglog_linear_sched_fn(
                        i,
                        x_anneal_initial_value=pfrac_anneal_initial,
                        x_anneal_final_value=pfrac,
                        x_anneal_start_step=pfrac_anneal_start_frac * total_steps,
                        x_anneal_stop_step=pfrac_anneal_stop_frac * total_steps,
                    )
                case _:
                    raise ValueError(f"Invalid pfrac_anneal_mode: {pfrac_anneal_mode}")
        else:
            this_pfrac = pfrac

        if afrac is not None:
            this_afrac = afrac
            beeg_model.config.afrac = this_afrac

        if pfrac_anneal_schedule_lr_with_L0:
            my_lr = my_lr * ((pfrac / this_pfrac) ** 0.5)

        for g in optimizer.param_groups:
            g["lr"] = my_lr * g.get("lr_scale", 1.0)
        optimizer.zero_grad()
        toks = x.cuda()
        with autocast_ctx_manager:
            beeg_logits, _, _ = beeg_model(toks, include_resid_mid=True)

        n_vocab = beeg_logits.shape[-1]

        kl = F.cross_entropy(
            beeg_logits[:, :-1].reshape(-1, n_vocab),
            toks[:, 1:].reshape(-1),
        )

        loss = kl
        loss = scaler.scale(loss)
        loss.backward()

        scaler.unscale_(optimizer)

        if gradclip:
            with torch.no_grad():
                model_norm = torch.nn.utils.clip_grad_norm_(all_params, 9999999999)
                num_params = sum(p.numel() for p in all_params)
                model_rms = model_norm / num_params**0.5
                for p in all_params:
                    p.grad.data /= model_rms + 1e-5

        if i % val_every == 0:
            # test
            testdatabs = 32
            with torch.no_grad():
                for testdataidx in range(0, my_test_xs.shape[0], testdatabs):
                    my_test_xs_slice = my_test_xs[testdataidx : testdataidx + testdatabs]
                    beeg_logits, beeg_test_loss, _ = beeg_model(
                        my_test_xs_slice[:, :-1].clone(), my_test_xs_slice[:, 1:].clone()
                    )

        scaler.step(optimizer)
        scaler.update()

        if pfrac is not None:
            apply_topk_(
                beeg_model,
                pfrac=this_pfrac,
                expansion_factor=expansion_factor,
                expansion_factor_mlp=expansion_factor_mlp,
                achyuta_mixed_neuronwise_v2=achyuta_mixed_neuronwise_v2,
                minimum_alive_per_neuron=minimum_alive_per_neuron,
                final_pfrac=pfrac,
            )
        if weight_decay:
            it = list(beeg_model.parameters())

            for p in it:
                if len(p.shape) > 1 and p is not beeg_model.bigram_table:
                    p.data -= weight_decay * my_lr * p.data

        if i % val_every == 0:
            print(f"TEST: XE {beeg_test_loss.item():.4f}  ")

        if i >= total_steps:
            break

        pbar.update(1)

    pbar.close()

    if rank == 0:
        # replace this with wherever you want to save the results
        save_dir = os.path.abspath(expname)
        os.makedirs(save_dir, exist_ok=True)

        config_path = bf.join(save_dir, "beeg_config.json")
        with bf.BlobFile(config_path, "w") as f:
            json.dump(asdict(beeg_config), f)

        model_path = bf.join(save_dir, "final_model.pt")
        with bf.BlobFile(model_path, "wb") as fh:
            torch.save(beeg_model.state_dict(), fh)
        print(f"Saved checkpoint to {save_dir}")

if __name__ == "__main__":
    main(
        expname="example",
        n_ctx=256,
        global_bs=512,
        vocab_size=2048,
        lr=1.28e-2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        pfrac=1 / 4,
        pfrac_anneal=True,
        pfrac_anneal_mode="linear",
        pfrac_anneal_exponent=1,
        pfrac_anneal_start_frac=0.01,
        pfrac_anneal_stop_frac=0.5,
        pfrac_anneal_initial="max",
        pfrac_anneal_schedule_lr_with_L0=True,
        achyuta_mixed_neuronwise_v2=True,
        minimum_alive_per_neuron=4,
        expansion_factor=4,
        lr_decay=True,
        gradclip=True,
        activation_type="gelu",
        residual_activation_type="identity",
        enable_mixed=True,
        dataset_name="placeholder",
        total_tokens=32_000_000_000,
        d_head=16,
        enable_bigram_table=True,
        lr_warmup_frac=0.01,
        n_layer=8,
        eps=0.1,
        afrac=0.25,
        afrac_loctypes="attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q",
        val_every=20,
        attn_sink=False,
        seed=0,
    )
