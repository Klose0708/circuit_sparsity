#!/usr/bin/env python3
"""
Minimal per-token activation visualizer for viz_data.pkl files.

What it does
- Loads a viz_data.pkl
- Attempts to find tokens and per-token activation arrays
- Lets you select a candidate array + node index
- Renders a per-token heatmap over the code/tokens

Run
  STREAMLIT_SERVER_FILEWATCHER_TYPE=none \
  streamlit run per_token_viz_demo.py

Notes
- This is schema-adaptive: it scans viz_data["samples"] for arrays shaped
  like [num_nodes, seq_len] or [seq_len, num_nodes]. You can pick the right one.
- If your data stores token ids only, it can decode them via tiktoken if installed
  and the tokenizer name exists in viz_data["importances"]["beeg_model_config"].
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st

os.environ.setdefault("STREAMLIT_SERVER_FILEWATCHER_TYPE", "none")


def _to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _iter_items(obj: Any, prefix: tuple[str, ...] = ()):  # yields (path, value)
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_items(v, prefix + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_items(v, prefix + (f"[{i}]",))
    else:
        yield prefix, obj


@dataclass
class Candidate:
    path: tuple[str, ...]
    arr: np.ndarray
    orientation: str  # "nodes_tokens" or "tokens_nodes"


def find_candidates(samples: dict[str, Any], seq_len: int, max_candidates: int = 50) -> list[Candidate]:
    cands: list[Candidate] = []
    for path, v in _iter_items(samples):
        try:
            a = _to_numpy(v)
        except Exception:
            continue
        if not isinstance(a, np.ndarray):
            continue
        if a.ndim != 2:
            continue
        h, w = a.shape
        if w == seq_len:  # [nodes, tokens]
            cands.append(Candidate(path, a, "nodes_tokens"))
        elif h == seq_len:  # [tokens, nodes]
            cands.append(Candidate(path, a, "tokens_nodes"))
        if len(cands) >= max_candidates:
            break
    return cands


def try_get_tokens(viz_data: dict[str, Any]) -> list[str] | None:
    samples = viz_data.get("samples", {}) or {}
    # Direct string tokens
    for key in ("tokens", "code_tokens", "str_tokens", "text_tokens"):
        toks = samples.get(key)
        if isinstance(toks, (list, tuple)) and toks and isinstance(toks[0], str):
            return list(toks)
    # Token IDs + tiktoken
    tok_ids = None
    for key in ("token_ids", "input_ids", "ids"):
        ids = samples.get(key)
        if isinstance(ids, (list, tuple, np.ndarray)):
            tok_ids = list(ids)
            break
    if tok_ids is not None:
        try:
            import tiktoken

            encname = (
                viz_data.get("importances", {})
                .get("beeg_model_config", None)
                .tokenizer_name
            )
            enc = tiktoken.get_encoding(encname)
            return [enc.decode_single_token_bytes(int(t)).decode("utf-8", errors="replace") for t in tok_ids]
        except Exception:
            return [str(int(t)) for t in tok_ids]
    return None


def color_spans(tokens: list[str], values: np.ndarray, center_zero: bool = True):
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("coolwarm")
    v = values
    if center_zero:
        m = float(np.max(np.abs(v))) if v.size > 0 else 1.0
        lo, hi = -m, m
    else:
        lo, hi = float(v.min(initial=0.0)), float(v.max(initial=1.0))
    norm = (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-9)
    html = '<div style="white-space: pre-wrap; color:black; background:white; font-family:monospace; border: 1px solid #AAA; padding:4px">'
    for tok, r in zip(tokens, norm, strict=False):
        rgba = cmap(r)
        R, G, B, A = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), rgba[3]
        style = f"background: rgba({R},{G},{B},{A:.3f});"
        tok = "\n" if tok == "\n" else tok
        html += f'<span style="{style}" title="{float(v[len(html)%len(v)]):.4f}">{tok}</span>'
    html += "</div>"
    return html


def main():
    st.set_page_config(page_title="Per-token viz demo", layout="wide")
    st.config.set_option("server.fileWatcherType", "none")

    p = st.text_input("viz_data.pkl path", value="")
    if not p:
        st.info("Provide a path to viz_data.pkl to begin.")
        return
    if not os.path.exists(p):
        st.error("Path does not exist")
        return
    with open(p, "rb") as fh:
        viz_data = pickle.load(fh)

    samples = viz_data.get("samples", {}) or {}
    tokens = try_get_tokens(viz_data)
    if not tokens:
        st.error("Could not find or decode tokens in viz_data['samples'].")
        st.write("Available sample keys:", list(samples.keys()))
        return

    st.caption(f"Detected {len(tokens)} tokens")
    candidates = find_candidates(samples, seq_len=len(tokens))
    if not candidates:
        st.error("No per-token arrays found under viz_data['samples']. Provide a key with shape [nodes, seq] or [seq, nodes].")
        st.write("Available sample keys:", list(samples.keys()))
        return

    cand_labels = ["/".join(c.path) + f"  shape={tuple(c.arr.shape)}  as={c.orientation}" for c in candidates]
    ci = st.selectbox("choose activation array", options=list(range(len(candidates))), format_func=lambda i: cand_labels[i])
    cand = candidates[ci]

    orient = st.radio("orientation", options=["nodes_tokens", "tokens_nodes"], index=0 if cand.orientation == "nodes_tokens" else 1, horizontal=True)
    arr = cand.arr if orient == "nodes_tokens" else cand.arr.T
    n_nodes, seq_len = arr.shape
    st.caption(f"Using array with shape [nodes={n_nodes}, seq={seq_len}]")

    idx = st.slider("node idx", 0, max(0, n_nodes - 1), 0, step=1)
    vals = _to_numpy(arr[idx])  # [seq]

    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown("**Per-token activations**")
        st.html(color_spans(tokens, vals))
    with cols[1]:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 2))
        sns.heatmap(vals[None, :], cmap="coolwarm", cbar=True, center=0, ax=ax)
        ax.set_yticks([])
        ax.set_xlabel("token position")
        st.pyplot(fig, use_container_width=True)

    with st.expander("debug: list of candidate arrays"):
        for i, c in enumerate(candidates):
            st.text(f"[{i}] path=/{'/'.join(c.path)}  shape={tuple(c.arr.shape)}  as={c.orientation}")


if __name__ == "__main__":
    main()

