"""
Reproduce Figure 8 (bracket-counting "context dilution") from:
  "Weight-sparse transformers have interpretable circuits" (Gao et al. 2025)
  https://arxiv.org/pdf/2511.13653

Figure 8 shows:
  (left) p(correct) for predicting the nested-list closing token `]]\n` drops as list length grows
  (right) activation magnitude of a key feature (paper: `2.attn.resid_delta[1249]`) drops with list length

This script follows the repo-native task definition used by
`circuit_sparsity/examples/bracket_counting_example.py`:
  correct token:  enc.encode("]]\n")[0]
  incorrect token: enc.encode("]\n")[0]

Run (CPU by default):
  python circuit_sparsity/examples/reproduce_figure8.py

Outputs:
  artifacts/figure8_left.csv,  artifacts/figure8_left.png
  artifacts/figure8_right.csv, artifacts/figure8_right.png

Notes:
  - We use the *unpruned* released model `csp_yolo2` by default, matching the paper claim.
  - Inputs are synthetic nested Python list literals with varying number of elements.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot(xs: list[int], ys: list[float], *, title: str, ylabel: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of elements in list")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=200)
    plt.close()


def _plot_left(xs: list[int], ys: list[float], *, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of elements in list")
    plt.ylabel("p(correct)")
    plt.title("Figure 8 (left) - context dilution on bracket_counting")
    # match paper: incorrect completion threshold at 0.5
    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    plt.text(xs[len(xs) // 2], 0.52, "incorrect completion threshold", color="gray")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=200)
    plt.close()


def _plot_right(xs: list[int], ys: list[float], *, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", color="orange")
    plt.xlabel("Number of elements in list")
    plt.ylabel("Activation 2.attn.resid_delta[1249]")
    plt.title("Figure 8 (right) - feature magnitude vs list length")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=200)
    plt.close()


_VIZ_PREFIX_CACHE: str | None = None


def _get_viz_prefix_text(enc: Encoding) -> str | None:
    """
    Try to build a prompt prefix from the released bracket_counting viz sample, to better
    match the paper's evaluation distribution.

    Strategy:
      - Load viz_data.pt for bracket_counting.
      - Take the first sample, decode to text.
      - Cut the text at the *last* occurrence of "[[" and keep everything up to and including it.
        Then our synthetic list body gets appended after that.
    """
    global _VIZ_PREFIX_CACHE
    if _VIZ_PREFIX_CACHE is not None:
        return _VIZ_PREFIX_CACHE

    try:
        import io

        viz_path = (
            f"{MODEL_BASE_DIR}/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
        )
        buf = io.BytesIO(read_file_cached(viz_path))
        viz = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception:
        return None

    try:
        ts = viz["importances"]["task_samples"]
        # task_samples is often a tuple of (group0, group1)
        g0 = ts[0]
        # g0 is a Tensor [n_samples, seq_len] or list of 1D tensors
        sample = g0[0]
        if isinstance(sample, torch.Tensor) and sample.ndim == 2:
            sample = sample[0]
        if not (isinstance(sample, torch.Tensor) and sample.ndim == 1):
            return None
        # truncate trailing zeros (padding)
        nz = sample.nonzero()
        if nz.numel() > 0:
            sample = sample[: nz[-1] + 1]
        text = enc.decode(sample.tolist())
        pos = text.rfind("[[")
        if pos == -1:
            return None
        prefix = text[: pos + 2]
        # Keep only a recent window of tokens to avoid extremely long contexts that
        # (a) shift the distribution away from the paper and (b) can be slow/unstable.
        # Default: 256 tokens, override with FIG8_PREFIX_TOKS.
        try:
            max_toks = int(os.environ.get("FIG8_PREFIX_TOKS", "256"))
        except Exception:
            max_toks = 256
        if max_toks > 0:
            ids = enc.encode(prefix)
            if len(ids) > max_toks:
                ids = ids[-max_toks:]
                prefix = enc.decode(ids)
                # ensure prefix still ends with "[[" (if it got cut, fall back to synthetic)
                if not prefix.endswith("[["):
                    pos2 = prefix.rfind("[[")
                    if pos2 == -1:
                        return None
                    prefix = prefix[: pos2 + 2]
        _VIZ_PREFIX_CACHE = prefix
        return _VIZ_PREFIX_CACHE
    except Exception:
        return None


def _make_nested_list_prompt(n: int, variant: int) -> str:
    """
    Make a nested list literal that ends right before the model should close it.
    We use a simple 2-level nested list: x = [[a, b, c, ...   (no closing brackets, no newline)
    """
    # Paper/viz samples look like numeric lists (e.g. "values =[[5, 3, 11, 3, 12 ...]").
    # Using non-numeric tokens (letters) makes the prompt distribution less realistic and
    # noticeably lowers the small-n baseline p(correct).
    rng = random.Random(variant)
    elems = [str(rng.randrange(0, 20)) for _ in range(n)]
    body = ", ".join(elems)

    # Optional: use a real-code prefix extracted from viz_data.pt (closer to paper distribution).
    # This relies on `_VIZ_PREFIX_CACHE` being populated once in `main()`.
    if _VIZ_PREFIX_CACHE:
        return f"{_VIZ_PREFIX_CACHE}{body}"

    # Prompt style controls how the list is introduced:
    # - "assign_nospace" (default): "values =[[1, 2, 3"
    # - "assign":         "values = [[1, 2, 3"
    # - "list":           "[[1, 2, 3"
    # - "split":          "values = [ [1, 2, 3"
    style = os.environ.get("FIG8_PROMPT_STYLE", "assign_nospace").strip().lower()
    if style == "assign":
        prefix = "values = [["
    elif style == "list":
        prefix = "[["
    elif style == "split":
        prefix = "values = [ ["
    else:
        prefix = "values =[["
    return f"{prefix}{body}"


def _seq_logprob_two_choice(
    model,
    *,
    prompt_ids: list[int],
    correct_ids: list[int],
    incorrect_ids: list[int],
    device: str,
) -> float:
    """
    Compute p(correct | {correct, incorrect}) using sequence log-probabilities, not just
    the first token's logits. This makes the metric robust when completions are multi-token.
    """

    def seq_logprob(comp_ids: list[int]) -> torch.Tensor:
        # need logits for each completion token, so run on prompt + comp[:-1]
        inp = prompt_ids + comp_ids[:-1]
        x = torch.tensor(inp, dtype=torch.long, device=device).unsqueeze(0)
        logits, _, _ = model(x)
        logp = 0.0
        # positions in logits that predict completion tokens:
        # first completion token is predicted at position len(prompt_ids)-1
        start = len(prompt_ids) - 1
        for j, tok in enumerate(comp_ids):
            pos = start + j
            lp = F.log_softmax(logits[0, pos].float(), dim=-1)[tok]
            logp = logp + lp
        return logp

    lp_c = seq_logprob(correct_ids)
    lp_i = seq_logprob(incorrect_ids)
    # softmax over the two candidates
    return float(torch.softmax(torch.stack([lp_c, lp_i]), dim=0)[0].item())


def _ensure_local_model(model_name: str) -> str:
    """
    Download model files locally so we can show clear progress and avoid silent streaming hangs.
    Returns local directory path suitable for `load_model`.
    """
    remote_dir = f"{MODEL_BASE_DIR}/models/{model_name}"
    local_dir = Path(".cache") / "circuit_sparsity" / "models" / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    for fn in ["beeg_config.json", "final_model.pt"]:
        src = f"{remote_dir}/{fn}"
        dst = local_dir / fn
        expected = None
        try:
            expected = bf.stat(src).size
        except Exception:
            expected = None
        ok = dst.exists() and dst.stat().st_size > 0 and (expected is None or dst.stat().st_size == expected)
        if ok:
            continue
        print(f"[figure8] downloading {model_name}/{fn} ...", flush=True)
        bf.copy(src, dst.as_posix(), overwrite=True)
        if expected is not None and dst.stat().st_size != expected:
            raise RuntimeError(f"download size mismatch for {dst}: got {dst.stat().st_size}, expected {expected}")
        print(f"[figure8] downloaded {model_name}/{fn} ({dst.stat().st_size} bytes)", flush=True)

    return local_dir.as_posix()


@dataclass(frozen=True)
class Figure8Config:
    model_name: str = "csp_yolo2"
    layer: int = 2
    channel: int = 1249
    # If provided, overrides (n_min, n_max, n_step)
    ns: tuple[int, ...] | None = None
    n_min: int = 5
    n_max: int = 67
    n_step: int = 5
    samples_per_n: int = 8
    device: str = "cpu"
    token_mode: str = "brackets_nl"  # "brackets" (]] vs ]) or "brackets_nl" (]]\n vs ]\n)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Paper-like sweep points (optional). Example: FIG8_NS="5,9,13,18,23,28,33,38,43,48,53,58,63,67"
    ns_env = os.environ.get("FIG8_NS", "").strip()
    ns_override = None
    if ns_env:
        ns_override = tuple(int(x) for x in ns_env.split(",") if x.strip())

    cfg = Figure8Config(
        model_name=os.environ.get("FIG8_MODEL", "csp_yolo2"),
        layer=int(os.environ.get("FIG8_LAYER", "2")),
        channel=int(os.environ.get("FIG8_CHANNEL", "1249")),
        ns=ns_override,
        n_min=int(os.environ.get("FIG8_N_MIN", "5")),
        n_max=int(os.environ.get("FIG8_N_MAX", "67")),
        n_step=int(os.environ.get("FIG8_N_STEP", "5")),
        samples_per_n=int(os.environ.get("FIG8_SAMPLES_PER_N", "8")),
        device=os.environ.get("FIG8_DEVICE", "cpu"),
        # Paper Figure 8 caption says "]]", but in this tokenizer the task is commonly defined on the
        # single tokens "]]\n" vs "]\n" (see bracket_counting_example.py), which gives a much cleaner
        # binary decision and matches the paper's high p(correct) at small n.
        token_mode=os.environ.get("FIG8_TOKEN_MODE", "brackets_nl").strip().lower(),
    )
    # Optional quick mode (faster but noisier): FIG8_QUICK=1
    if os.environ.get("FIG8_QUICK", "").strip() == "1":
        cfg = Figure8Config(**{**cfg.__dict__, "samples_per_n": 3, "n_step": 6})

    print(f"[figure8] config={cfg}", flush=True)

    enc = Encoding(**tinypython.tinypython_2k())
    print("[figure8] preparing model files...", flush=True)
    model_path = _ensure_local_model(cfg.model_name)
    print("[figure8] loading model into memory...", flush=True)
    model = load_model(model_path, flash=True, grad_checkpointing=False, cuda=(cfg.device == "cuda"))
    print("[figure8] model loaded.", flush=True)

    # Populate viz prefix cache once (default on). Falls back silently if unavailable.
    # NOTE: We do this AFTER loading the model weights. On Windows, doing a torch.load(viz_data.pt)
    # before torch.load(final_model.pt) has been observed to occasionally trigger a native crash
    # (0xC0000005) during model loading.
    if os.environ.get("FIG8_USE_VIZ_PREFIX", "1").strip() == "1":
        global _VIZ_PREFIX_CACHE
        _VIZ_PREFIX_CACHE = _get_viz_prefix_text(enc)
        if _VIZ_PREFIX_CACHE:
            print("[figure8] using viz-derived prefix for prompts (closer to paper).", flush=True)
        else:
            print("[figure8] viz-derived prefix unavailable; using synthetic prompt prefix.", flush=True)

    if cfg.token_mode == "brackets_nl":
        correct_str = "]]\n"
        incorrect_str = "]\n"
        tok_correct_str = "]]\\n"
        tok_incorrect_str = "]\\n"
    else:
        # Match Figure 8 caption wording ("correct completion of ]] on longer lists")
        correct_str = "]]"
        incorrect_str = "]"
        tok_correct_str = "]]"
        tok_incorrect_str = "]"

    correct_ids = enc.encode(correct_str)
    incorrect_ids = enc.encode(incorrect_str)
    # For logging only: show first token ids (may be multi-token completions)
    tok_correct = correct_ids[0]
    tok_incorrect = incorrect_ids[0]
    hook_key = f"{cfg.layer}.attn.resid_delta"
    print(
        f"[figure8] completion: correct='{tok_correct_str}' ids={correct_ids} | incorrect='{tok_incorrect_str}' ids={incorrect_ids}",
        flush=True,
    )
    print(f"[figure8] hook={hook_key} channel={cfg.channel}", flush=True)

    if cfg.ns is not None and len(cfg.ns) > 0:
        ns = list(cfg.ns)
    else:
        ns = list(range(cfg.n_min, cfg.n_max + 1, cfg.n_step))
    left_rows: list[dict] = []
    right_rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        for idx_n, n in enumerate(ns):
            t_n0 = time.time()
            ps: list[float] = []
            acts: list[float] = []
            for k in range(cfg.samples_per_n):
                prompt = _make_nested_list_prompt(n, k)
                prompt_ids = enc.encode(prompt)
                toks = torch.tensor(prompt_ids, dtype=torch.long, device=cfg.device).unsqueeze(0)
                # Recording all hooks (regex=".*") is extremely memory-heavy for long prompts.
                # We only need this single hook for Figure 8 (right).
                hook_regex = f"^{cfg.layer}\\.attn\\.resid_delta$"
                with hook_recorder(regex=hook_regex, interventions=None) as rec:
                    logits, _, _ = model(toks)

                # p(correct) over the two relevant candidates (sequence-level)
                # (uses a second forward pass internally only if needed)
                if len(correct_ids) == 1 and len(incorrect_ids) == 1:
                    last = logits[0, -1]
                    x2 = torch.stack([last[correct_ids[0]], last[incorrect_ids[0]]]).float()
                    p = float(F.softmax(x2, dim=0)[0].item())
                else:
                    p = _seq_logprob_two_choice(
                        model,
                        prompt_ids=prompt_ids,
                        correct_ids=correct_ids,
                        incorrect_ids=incorrect_ids,
                        device=cfg.device,
                    )
                ps.append(p)
                # activation magnitude at last position
                a = rec[hook_key][0, -1, cfg.channel]
                # Paper plot appears to use raw activation (positive, decreasing).
                acts.append(float(a.item()))
                if k == 0:
                    print(f"[figure8] n={n} sample {k+1}/{cfg.samples_per_n} ok; p≈{p:.4f} act≈{acts[-1]:.4f}", flush=True)
            left_rows.append({"n": n, "p_correct_mean": sum(ps) / len(ps)})
            right_rows.append({"n": n, "activation_mean": sum(acts) / len(acts)})
            dt = time.time() - t_n0
            done = idx_n + 1
            eta = (len(ns) - done) * dt
            print(f"[figure8] done n={n} ({done}/{len(ns)}) p_correct={left_rows[-1]['p_correct_mean']:.4f} act={right_rows[-1]['activation_mean']:.4f} dt={dt:.1f}s eta≈{eta/60:.1f}min", flush=True)

    _write_csv(ARTIFACTS_DIR / "figure8_left.csv", left_rows)
    _write_csv(ARTIFACTS_DIR / "figure8_right.csv", right_rows)

    _plot_left([r["n"] for r in left_rows], [r["p_correct_mean"] for r in left_rows], out_png=ARTIFACTS_DIR / "figure8_left.png")
    # Use raw activation mean (closer to paper) on the right
    _plot_right([r["n"] for r in right_rows], [r["activation_mean"] for r in right_rows], out_png=ARTIFACTS_DIR / "figure8_right.png")

    print("Wrote:", (ARTIFACTS_DIR / "figure8_left.png").as_posix(), flush=True)
    print("Wrote:", (ARTIFACTS_DIR / "figure8_right.png").as_posix(), flush=True)
    print(f"[figure8] total time {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

