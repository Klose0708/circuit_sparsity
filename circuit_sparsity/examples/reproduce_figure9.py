"""
Reproduce Figure 9 (qualitatively) from:
  "Weight-sparse transformers have interpretable circuits" (Gao et al. 2025)
  https://arxiv.org/pdf/2511.13653

This script implements the core Figure 9 experiment:
  - pick a sparse model feature/channel that discriminates two conditions
  - construct a "counterfactual" perturbation in sparse activation space
  - map that perturbation into dense activation space via a linear bridge
  - inject (steer) the dense model at a specific hook location
  - sweep steering strength and plot the resulting behavioral probability

Important notes (Jan 2026):
  - The blob currently serves `models/csp_bridges{1,2}/final_bridges.pt` as a 0-byte file.
  - Additionally, `models/csp_bridges1/final_model.pt` appears to be corrupted on blob
    (zip file missing the central directory). Because of this, by default we DO NOT
    use `csp_bridges*` models.
  - Instead, we fit the needed bridge maps on-the-fly via ridge regression using paired
    activations from the released dense model (`dense1_4x`) and a small released sparse
    sweep model (default: `csp_sweep1_1x_0.9Mnonzero_afrac0.125`).

Run from repo root (PowerShell / cmd OK):
  python circuit_sparsity/examples/reproduce_figure9.py

Outputs:
  - artifacts/figure9_left.csv, artifacts/figure9_left.png
  - artifacts/figure9_right.csv, artifacts/figure9_right.png
"""

from __future__ import annotations

import csv
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import blobfile as bf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tiktoken import Encoding
from tiktoken.load import read_file_cached

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.inference.hook_utils import hook_recorder
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


ARTIFACTS_DIR = Path("artifacts")


def _truncate_zeros_1d(sample: torch.Tensor) -> torch.Tensor:
    """Remove trailing 0 padding tokens (common in viz task_samples)."""
    assert sample.ndim == 1
    nz = (sample != 0).nonzero().flatten()
    if nz.numel() == 0:
        return sample
    return sample[: nz[-1].item() + 1]


def _download_with_retries(src: str, dst: Path, *, expected_size: int | None, retries: int = 5) -> None:
    """
    Download a single blob path to local file with retries and size checks.
    Uses blobfile's streaming copy, then verifies byte size against blob stat if provided.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for i in range(retries):
        try:
            # Start fresh each attempt to avoid keeping a truncated file.
            if dst.exists():
                try:
                    dst.unlink()
                except Exception:
                    pass
            bf.copy(src, dst.as_posix(), overwrite=True)
            if expected_size is not None:
                got = dst.stat().st_size if dst.exists() else -1
                if got != expected_size:
                    raise RuntimeError(f"size mismatch for {dst}: got {got}, expected {expected_size}")
            return
        except Exception as e:
            last_err = e
            # Exponential-ish backoff
            time.sleep(min(2.0 * (2**i), 20.0))
    raise RuntimeError(f"failed to download {src} -> {dst}") from last_err


def _ensure_local_dir_from_blob(blob_dir: str, local_dir: Path, filenames: list[str]) -> Path:
    """
    Ensure `local_dir/<filename>` exists by copying from `blob_dir/<filename>`.
    Returns local_dir.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    for fn in filenames:
        dst = local_dir / fn
        src = f"{blob_dir}/{fn}"
        expected_size = None
        try:
            expected_size = bf.stat(src).size
        except Exception:
            expected_size = None

        if dst.exists() and dst.stat().st_size > 0 and (expected_size is None or dst.stat().st_size == expected_size):
            continue

        _download_with_retries(src, dst, expected_size=expected_size, retries=6)
    return local_dir


def _load_viz_data(url: str) -> dict:
    return torch.load(io.BytesIO(read_file_cached(url)), map_location="cpu", weights_only=True)


def _load_task_samples(viz_url: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (group0, group1) where each group is a tensor of shape [N, T].
    """
    viz = _load_viz_data(viz_url)
    ts = viz["importances"]["task_samples"]
    # ts is usually a tuple of tensors [2] each shaped [N, T].
    assert isinstance(ts, (tuple, list)) and len(ts) == 2
    g0, g1 = ts[0], ts[1]
    if not (isinstance(g0, torch.Tensor) and isinstance(g1, torch.Tensor)):
        raise TypeError(f"unexpected task_samples format: {[type(x) for x in ts]}")
    return g0.cpu(), g1.cpu()


def _collect_acts_matrix(
    model: torch.nn.Module,
    samples: torch.Tensor,  # [N, T]
    hook_key: str,
    *,
    device: str,
    use_all_positions: bool,
) -> torch.Tensor:
    """
    Collect activation matrix X for regression.
    If use_all_positions=True, stacks all non-padding positions across all samples (N_total x C).
    Else, uses only the last non-padding position from each sample (N x C).
    """
    xs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(samples.shape[0]):
            s = _truncate_zeros_1d(samples[i]).to(device)
            idx = s.unsqueeze(0)  # [1, T]
            with hook_recorder(regex=".*", interventions=None) as rec:
                model(idx)
            a = rec[hook_key]  # [1, T, C]
            assert a.ndim == 3 and a.shape[0] == 1
            # decide which positions to keep
            if use_all_positions:
                # keep all positions corresponding to non-pad tokens
                pos = torch.arange(a.shape[1], device=a.device)
                keep = pos < idx.shape[1]
                xs.append(a[0, keep, :].detach().cpu())
            else:
                xs.append(a[0, -1, :].detach().cpu().unsqueeze(0))
    return torch.cat(xs, dim=0).float()  # [M, C]


def _fit_ridge_bridge(
    Xs: torch.Tensor,  # [M, d_s]
    Yd: torch.Tensor,  # [M, d_d]
    *,
    lam: float = 1e-2,
) -> torch.Tensor:
    """
    Fit W minimizing ||Xs W - Yd||^2 + lam ||W||^2.
    Returns W of shape [d_s, d_d].
    """
    assert Xs.ndim == 2 and Yd.ndim == 2
    assert Xs.shape[0] == Yd.shape[0]
    X = Xs
    Y = Yd
    d_s = X.shape[1]
    # XtX: [d_s, d_s], XtY: [d_s, d_d]
    XtX = X.T @ X
    XtX = XtX + lam * torch.eye(d_s, dtype=XtX.dtype)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY)
    return W


def _pick_discriminative_channel(
    sparse_model: torch.nn.Module,
    group_src: torch.Tensor,  # [N,T]
    group_tgt: torch.Tensor,  # [N,T]
    hook_key: str,
    *,
    device: str,
) -> tuple[int, float]:
    """
    Choose channel j maximizing |mean_tgt - mean_src| / (std_src+std_tgt+eps).
    Returns (j, delta_scalar) where delta_scalar = mean_tgt[j] - mean_src[j].
    """
    Xs = _collect_acts_matrix(sparse_model, group_src, hook_key, device=device, use_all_positions=False)
    Xt = _collect_acts_matrix(sparse_model, group_tgt, hook_key, device=device, use_all_positions=False)
    mu_s = Xs.mean(dim=0)
    mu_t = Xt.mean(dim=0)
    std = Xs.std(dim=0) + Xt.std(dim=0) + 1e-6
    score = (mu_t - mu_s).abs() / std
    j = int(torch.argmax(score).item())
    delta = float((mu_t[j] - mu_s[j]).item())
    return j, delta


def _binary_prob_from_logits(last_logits: torch.Tensor, tok_a: int, tok_b: int) -> float:
    """
    Return P(tok_a | {tok_a, tok_b}) computed by softmax over just the two tokens.
    """
    x = last_logits[[tok_a, tok_b]].float()
    p = F.softmax(x, dim=-1)[0].item()
    return float(p)


def _binary_prob_from_logits_multi(
    last_logits: torch.Tensor, tok_as: list[int], tok_bs: list[int]
) -> float:
    """
    Return P(A | A∪B) where A and B are *sets* of token ids.
    Useful when the tokenizer may represent a "concept" with multiple tokens
    (e.g. \" vs \"\\n).
    """
    a = torch.logsumexp(last_logits[tok_as].float(), dim=0)
    b = torch.logsumexp(last_logits[tok_bs].float(), dim=0)
    return float(torch.softmax(torch.stack([a, b]), dim=0)[0].item())


def _compute_counterfactual_delta(
    sparse_model: torch.nn.Module,
    group_src: torch.Tensor,
    group_tgt: torch.Tensor,
    hook_key: str,
    *,
    device: str,
    topk: int | None = None,
) -> torch.Tensor:
    """
    Compute a sparse-space counterfactual delta as mean(tgt) - mean(src) at the last token position.
    Optionally keep only top-k abs components for stability / interpretability.
    Returns vector [d_s].
    """
    Xs = _collect_acts_matrix(sparse_model, group_src, hook_key, device=device, use_all_positions=False)
    Xt = _collect_acts_matrix(sparse_model, group_tgt, hook_key, device=device, use_all_positions=False)
    delta = (Xt.mean(dim=0) - Xs.mean(dim=0)).float()
    if topk is not None and topk > 0 and topk < delta.numel():
        vals, idx = torch.topk(delta.abs(), topk, sorted=False)
        keep = torch.zeros_like(delta)
        keep[idx] = delta[idx]
        delta = keep
    return delta

@dataclass(frozen=True)
class Figure9TaskSpec:
    name: str
    viz_url: str
    # Which hook to perturb / bridge
    hook_key: str
    # Token ids (target, alt) to compute binary probability
    tok_target: int
    tok_alt: int
    # How to identify which task_samples group is "source" vs "target"
    # (we steer source -> target)
    group_id_source: int
    group_id_target: int


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_curve(xs: list[float], ys: list[float], *, title: str, ylabel: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.yscale("log")
    plt.xlabel("Steering strength")
    plt.ylabel(ylabel)
    plt.title(title)
    # mimic figure9 baselines
    plt.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    plt.text(0.01, max(ys) * 0.7, "no steering", rotation=90, color="gray")
    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    plt.text(0.5, 0.55, "random", color="gray")
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=200)
    plt.close()

def _encode_and_pad(enc: Encoding, prompts: list[str]) -> torch.Tensor:
    toks = [torch.tensor(enc.encode(p), dtype=torch.long) for p in prompts]
    max_len = max(int(t.numel()) for t in toks)
    out = torch.zeros((len(toks), max_len), dtype=torch.long)
    for i, t in enumerate(toks):
        out[i, : t.numel()] = t
    return out


def _pad_2d_to_len(x: torch.Tensor, L: int) -> torch.Tensor:
    assert x.ndim == 2
    if x.shape[1] == L:
        return x
    out = torch.zeros((x.shape[0], L), dtype=x.dtype)
    out[:, : x.shape[1]] = x
    return out


def _pad_pair_to_same_len(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert a.ndim == 2 and b.ndim == 2
    L = max(int(a.shape[1]), int(b.shape[1]))
    return _pad_2d_to_len(a, L), _pad_2d_to_len(b, L)


def _make_quote_prompts(n: int, *, quote: str) -> list[str]:
    # Create prompts where a closing quote is strongly expected soon.
    # Using `print(<quote>...` encourages the model to close the string and then emit `)`.
    assert quote in ["'", '"']
    out: list[str] = []
    payloads = ["hello", "abc", "a_b_c", "path/to/file", "value=", "Station 1", "foo_bar"]
    for i in range(n):
        p = payloads[i % len(payloads)]
        out.append(f"print({quote}{p}")
    return out


def _make_while_return_prompts(n: int, *, kind: str) -> list[str]:
    # kind: "while" or "return"
    assert kind in ["while", "return"]
    out: list[str] = []
    for i in range(n):
        indent = "    " if (i % 2 == 0) else ""
        if kind == "while":
            out.append(f"def f():\n{indent}while True")
        else:
            out.append(f"def f():\n{indent}return True")
    return out


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    enc = Encoding(**tinypython.tinypython_2k())

    # --- Models ---
    dense_name = os.environ.get("FIG9_DENSE_MODEL", "dense1_4x")
    # For Figure 9 left, a 4-layer sparse model is important; `csp_bridges2` is loadable and 4-layer.
    sparse_quote_name = os.environ.get("FIG9_SPARSE_MODEL_QUOTE", "csp_bridges2")
    sparse_colon_name = os.environ.get("FIG9_SPARSE_MODEL_COLON", sparse_quote_name)

    # Download large model files robustly (avoid partial reads on Windows).
    local_models = Path(".cache") / "circuit_sparsity" / "models"
    dense_dir = _ensure_local_dir_from_blob(
        f"{MODEL_BASE_DIR}/models/{dense_name}",
        local_models / dense_name,
        ["beeg_config.json", "final_model.pt"],
    )
    sparse_quote_dir = _ensure_local_dir_from_blob(
        f"{MODEL_BASE_DIR}/models/{sparse_quote_name}",
        local_models / sparse_quote_name,
        ["beeg_config.json", "final_model.pt"],
    )
    sparse_colon_dir = _ensure_local_dir_from_blob(
        f"{MODEL_BASE_DIR}/models/{sparse_colon_name}",
        local_models / sparse_colon_name,
        ["beeg_config.json", "final_model.pt"],
    )

    dense = load_model(dense_dir.as_posix(), flash=True, grad_checkpointing=False, cuda=False)
    # If torch.load fails, force a re-download and retry once.
    try:
        sparse_quote = load_model(sparse_quote_dir.as_posix(), flash=True, grad_checkpointing=False, cuda=False)
    except Exception:
        _ensure_local_dir_from_blob(
            f"{MODEL_BASE_DIR}/models/{sparse_quote_name}",
            sparse_quote_dir,
            ["beeg_config.json", "final_model.pt"],
        )
        sparse_quote = load_model(sparse_quote_dir.as_posix(), flash=True, grad_checkpointing=False, cuda=False)

    try:
        sparse_colon = load_model(sparse_colon_dir.as_posix(), flash=True, grad_checkpointing=False, cuda=False)
    except Exception:
        _ensure_local_dir_from_blob(
            f"{MODEL_BASE_DIR}/models/{sparse_colon_name}",
            sparse_colon_dir,
            ["beeg_config.json", "final_model.pt"],
        )
        sparse_colon = load_model(sparse_colon_dir.as_posix(), flash=True, grad_checkpointing=False, cuda=False)

    # --- Task specs ---
    tok_single_ids = enc.encode(chr(39))  # "'"
    tok_double_ids = enc.encode(chr(34))  # '"'
    assert len(tok_single_ids) == 1 and len(tok_double_ids) == 1, (
        "Tokenizer does not have single-token quotes; Figure9-left replication assumes this."
    )
    tok_single = tok_single_ids[0]
    tok_double = tok_double_ids[0]
    tok_colon = enc.encode(":")[0]
    tok_nl = enc.encode("\n")[0]

    # Left (quote) is best reproduced with controlled prompts where the next token is the closing quote.
    # We'll use synthetic prompts by default to avoid dataset-version quirks.
    left_spec = Figure9TaskSpec(
        name="single_double_quote (synthetic prompts; steer toward single-close token)",
        viz_url="",
        hook_key="",  # filled dynamically after model load
        tok_target=tok_single,
        tok_alt=tok_double,
        group_id_source=0,
        group_id_target=1,
    )
    right_spec = Figure9TaskSpec(
        name="while_return_true_beeg (steer toward ':' via while True)",
        viz_url=f"{MODEL_BASE_DIR}/viz/csp_yolo2/while_return_true_beeg/prune_v4/k_optim/viz_data.pt",
        hook_key="",  # filled dynamically after model load
        tok_target=tok_colon,
        tok_alt=tok_nl,
        group_id_source=0,
        group_id_target=1,
    )

    # --- Load task samples (or synthesize prompts) ---
    fit_n_per_group = int(os.environ.get("FIG9_FIT_N_PER_GROUP", "8"))
    eval_n_per_group = int(os.environ.get("FIG9_EVAL_N_PER_GROUP", "16"))

    def _take_n(x: torch.Tensor, n: int) -> torch.Tensor:
        return x[: min(n, x.shape[0])].contiguous()

    # Left: use synthetic prompts (double -> single)
    left_n = max(eval_n_per_group, fit_n_per_group)
    left_src = _encode_and_pad(enc, _make_quote_prompts(left_n, quote='"'))
    left_tgt = _encode_and_pad(enc, _make_quote_prompts(left_n, quote="'"))

    try:
        g0_r, g1_r = _load_task_samples(right_spec.viz_url)

        # While/return task: decide by decoding a single sample and checking substring.
        def contains_substr(group: torch.Tensor, substr: str) -> bool:
            s = _truncate_zeros_1d(group[0])
            txt = enc.decode(s.tolist())
            return substr in txt

        if contains_substr(g0_r, "return True") and contains_substr(g1_r, "while True"):
            right_src, right_tgt = g0_r, g1_r  # return -> while
        elif contains_substr(g1_r, "return True") and contains_substr(g0_r, "while True"):
            right_src, right_tgt = g1_r, g0_r
        else:
            right_src, right_tgt = g0_r, g1_r
    except Exception:
        right_src = _encode_and_pad(enc, _make_while_return_prompts(eval_n_per_group, kind="return"))
        right_tgt = _encode_and_pad(enc, _make_while_return_prompts(eval_n_per_group, kind="while"))

    # Determine hook keys dynamically:
    # - Quote task: we hook the *input to attention* ("attn.act_in"), which is the residual stream
    #   after the block's LN, matching "input to the final attention layer" in the paper.
    # - While/return task: final MLP input channel ("mlp.act_in").
    dense_last = dense.config.n_layer - 1
    sparse_quote_last = sparse_quote.config.n_layer - 1
    sparse_colon_last = sparse_colon.config.n_layer - 1
    left_layer_sparse = min(dense_last, sparse_quote_last)
    left_hook_dense = f"{dense_last}.attn.act_in"
    left_hook_sparse = f"{left_layer_sparse}.attn.act_in"
    right_hook_dense = f"{dense_last}.mlp.act_in"
    right_hook_sparse = f"{sparse_colon_last}.mlp.act_in"

    # --- Fit bridges needed for each task ---
    # Use both source+target samples to fit a bridge local to the relevant hook.
    left_src_fit, left_tgt_fit = _pad_pair_to_same_len(_take_n(left_src, fit_n_per_group), _take_n(left_tgt, fit_n_per_group))
    right_src_fit, right_tgt_fit = _pad_pair_to_same_len(_take_n(right_src, fit_n_per_group), _take_n(right_tgt, fit_n_per_group))

    left_fit = torch.cat([left_src_fit, left_tgt_fit], dim=0)
    right_fit = torch.cat([right_src_fit, right_tgt_fit], dim=0)

    # For quote task, regress on the opening-quote token position (NOT the last token).
    def _quote_pos(tokens_1d: torch.Tensor) -> int:
        t = _truncate_zeros_1d(tokens_1d)
        # Robustly find the last token whose decoded text contains a quote character.
        # This handles tokenizers where quotes may appear in multi-character tokens (e.g. '="').
        last = None
        for i, tid in enumerate(t.tolist()):
            s = enc.decode_single_token_bytes(int(tid)).decode("utf-8", errors="replace")
            if ("\"" in s) or ("'" in s):
                last = i
        if last is not None:
            return int(last)
        return int(t.numel() - 1)

    def _acts_at_positions(model: torch.nn.Module, samples2d: torch.Tensor, hook_key: str, pos_fn) -> torch.Tensor:
        rows = []
        model.eval()
        with torch.no_grad():
            for i in range(samples2d.shape[0]):
                s = _truncate_zeros_1d(samples2d[i]).to(device)
                pos = min(pos_fn(s), s.numel() - 1)
                idx = s.unsqueeze(0)
                with hook_recorder(regex=".*", interventions=None) as rec:
                    model(idx)
                a = rec[hook_key]  # [1,T,C]
                rows.append(a[0, pos, :].detach().cpu().float().unsqueeze(0))
        return torch.cat(rows, dim=0)

    Xs_left = _acts_at_positions(sparse_quote, left_fit, left_hook_sparse, _quote_pos)
    Yd_left = _acts_at_positions(dense, left_fit, left_hook_dense, _quote_pos)
    W_left = _fit_ridge_bridge(Xs_left, Yd_left, lam=1e-1)

    # For while/return, last token position is fine.
    Xs_right = _collect_acts_matrix(sparse_colon, right_fit, right_hook_sparse, device=device, use_all_positions=True)
    Yd_right = _collect_acts_matrix(dense, right_fit, right_hook_dense, device=device, use_all_positions=True)
    W_right = _fit_ridge_bridge(Xs_right, Yd_right, lam=1e-1)

    # --- Pick a sparse "concept channel" and compute counterfactual delta ---
    topk_delta = os.environ.get("FIG9_DELTA_TOPK", "")
    topk_delta_i = int(topk_delta) if topk_delta.strip().isdigit() else None
    # delta for quote task should be computed at the quote token position
    with torch.no_grad():
        src_q = _acts_at_positions(sparse_quote, left_src_fit, left_hook_sparse, _quote_pos)
        tgt_q = _acts_at_positions(sparse_quote, left_tgt_fit, left_hook_sparse, _quote_pos)

    # Figure 9 (left) in the paper specifically steers a *single interpretable channel*
    # ("quote type classifier"). We approximate this by selecting the most discriminative
    # sparse channel at the quote token position and only perturbing that channel.
    left_use_single_channel = os.environ.get("FIG9_LEFT_SINGLE_CHANNEL", "1") != "0"
    if left_use_single_channel:
        mu_s = src_q.mean(dim=0)
        mu_t = tgt_q.mean(dim=0)
        std = src_q.std(dim=0) + tgt_q.std(dim=0) + 1e-6
        score = (mu_t - mu_s).abs() / std
        j = int(torch.argmax(score).item())
        delta_j = float((mu_t[j] - mu_s[j]).item())
        delta_left_vec = torch.zeros_like(mu_s).float()
        delta_left_vec[j] = delta_j
        print(f"[left] using single sparse channel j={j} delta={delta_j:+.4f}")
    else:
        delta_left_vec = (tgt_q.mean(dim=0) - src_q.mean(dim=0)).float()
        if topk_delta_i is not None and 0 < topk_delta_i < delta_left_vec.numel():
            _, idx = torch.topk(delta_left_vec.abs(), topk_delta_i, sorted=False)
            keep = torch.zeros_like(delta_left_vec)
            keep[idx] = delta_left_vec[idx]
            delta_left_vec = keep

    # Calibrate the mapped dense perturbation scale:
    # our fitted W may under-scale the dense-space effect (2048->1024 regression with few samples).
    # We compute the empirical dense delta between single vs double contexts at the quote position,
    # and choose a scalar so that (delta_sparse @ W) matches that delta in least-squares sense.
    with torch.no_grad():
        dense_src_q = _acts_at_positions(dense, left_src_fit, left_hook_dense, _quote_pos)
        dense_tgt_q = _acts_at_positions(dense, left_tgt_fit, left_hook_dense, _quote_pos)
        delta_dense_emp = (dense_tgt_q.mean(dim=0) - dense_src_q.mean(dim=0)).float()

    mapped_dense = (delta_left_vec @ W_left).float()
    denom = float((mapped_dense @ mapped_dense).item())
    if denom > 1e-12:
        left_scale = float((delta_dense_emp @ mapped_dense).item() / denom)
    else:
        left_scale = 1.0
    # allow manual override
    if os.environ.get("FIG9_LEFT_SCALE", "").strip():
        try:
            left_scale = float(os.environ["FIG9_LEFT_SCALE"])
        except Exception:
            pass
    print(
        f"[left] mapped_dense L2={float(mapped_dense.norm().item()):.4f} "
        f"emp_dense L2={float(delta_dense_emp.norm().item()):.4f} "
        f"scale={left_scale:+.4f}"
    )

    delta_right_vec = _compute_counterfactual_delta(
        sparse_colon, right_src_fit, right_tgt_fit, right_hook_sparse, device=device, topk=topk_delta_i
    )

    print(f"[left] delta_sparse L2={float(delta_left_vec.norm().item()):.4f} (topk={topk_delta_i})")
    print(f"[right] delta_sparse L2={float(delta_right_vec.norm().item()):.4f} (topk={topk_delta_i})")

    def run_sweep(
        *,
        dense_model: torch.nn.Module,
        hook_key: str,
        W: torch.Tensor,
        delta_sparse: torch.Tensor,  # [d_s]
        eval_samples: torch.Tensor,
        tok_target: int,
        tok_alt: int,
        tokset_target: list[int] | None = None,
        tokset_alt: list[int] | None = None,
        dense_scale: float = 1.0,
        strengths: list[float],
    ) -> list[float]:
        ys: list[float] = []
        base_dir_dense = (delta_sparse @ W).float() * float(dense_scale)  # [d_d]

        dense_model.eval()
        with torch.no_grad():
            for a in strengths:
                # perturbation vector in dense space
                dense_delta = a * base_dir_dense  # [d_d]
                dense_delta = dense_delta.to(device)

                # For quote task, edit at quote token position; for others, keep last token.
                def _intervene_factory(pos: int):
                    def _intervene(t: torch.Tensor) -> torch.Tensor:
                        out = t.clone()
                        out[:, pos, :] = out[:, pos, :] + dense_delta
                        return out
                    return _intervene

                ps: list[float] = []
                for i in range(eval_samples.shape[0]):
                    s = _truncate_zeros_1d(eval_samples[i]).to(device)
                    idx = s.unsqueeze(0)
                    if hook_key == left_hook_dense:
                        pos = min(_quote_pos(s), s.numel() - 1)
                    else:
                        pos = int(s.numel() - 1)
                    interventions = {hook_key: _intervene_factory(pos)}
                    with hook_recorder(regex=".*", interventions=interventions):
                        logits, _, _ = dense_model(idx)
                    last = logits[0, -1].cpu()
                    if tokset_target is not None and tokset_alt is not None:
                        ps.append(_binary_prob_from_logits_multi(last, tokset_target, tokset_alt))
                    else:
                        ps.append(_binary_prob_from_logits(last, tok_target, tok_alt))
                ys.append(float(sum(ps) / len(ps)))
        return ys

    strengths = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]

    # Evaluate on the *source* distribution (double-quoted prompts; return True prompts),
    # mirroring the paper's "counterfactual steering".
    ys_left = run_sweep(
        dense_model=dense,
        hook_key=left_hook_dense,
        W=W_left,
        delta_sparse=delta_left_vec,
        eval_samples=_take_n(left_src, eval_n_per_group),
        tok_target=left_spec.tok_target,
        tok_alt=left_spec.tok_alt,
        dense_scale=left_scale,
        strengths=strengths,
    )
    ys_right = run_sweep(
        dense_model=dense,
        hook_key=right_hook_dense,
        W=W_right,
        delta_sparse=delta_right_vec,
        eval_samples=_take_n(right_src, eval_n_per_group),
        tok_target=right_spec.tok_target,
        tok_alt=right_spec.tok_alt,
        strengths=strengths,
    )

    _write_csv(
        ARTIFACTS_DIR / "figure9_left.csv",
        [{"strength": a, "p_target": y} for a, y in zip(strengths, ys_left)],
    )
    _write_csv(
        ARTIFACTS_DIR / "figure9_right.csv",
        [{"strength": a, "p_target": y} for a, y in zip(strengths, ys_right)],
    )

    _plot_curve(
        strengths,
        ys_left,
        title="Figure 9 (left) - steer dense toward single quote",
        ylabel="P(single quote | {single,double})",
        out_png=ARTIFACTS_DIR / "figure9_left.png",
    )
    _plot_curve(
        strengths,
        ys_right,
        title="Figure 9 (right) - steer dense toward ':' via while True",
        ylabel="P(: | {:,\\n})",
        out_png=ARTIFACTS_DIR / "figure9_right.png",
    )

    print("Wrote:", (ARTIFACTS_DIR / "figure9_left.png").as_posix())
    print("Wrote:", (ARTIFACTS_DIR / "figure9_right.png").as_posix())


if __name__ == "__main__":
    main()



