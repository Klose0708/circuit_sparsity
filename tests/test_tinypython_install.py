#!/usr/bin/env python3
"""
Test that tinypython_2k is discoverable via tiktoken and functions minimally.

This script is a thin wrapper around install_tinypython.verify_install().

Usage
  python alignment/repe/repe/x/dan/cs_viz/test_tinypython_install.py \
    --tok-dir ~/data/tinypython_tok
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tok-dir", type=str, default=None)
    args = parser.parse_args(argv)

    # Ensure we can import the installer module regardless of CWD
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    from install_tinypython import ensure_tiktoken_ext_on_path, set_tok_dir, verify_install

    ensure_tiktoken_ext_on_path()
    set_tok_dir(args.tok_dir)
    return verify_install(verbose=True)


if __name__ == "__main__":
    raise SystemExit(main())

