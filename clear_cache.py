#!/usr/bin/env python3
"""
Delete the cache directories used by the visualizer utilities.

- ~/data/dev/shm/cache : CacheHelper in circuit_sparsity/viz.py
- tiktoken cache       : TIKTOKEN_CACHE_DIR, DATA_GYM_CACHE_DIR, or tmp/data-gym-cache
"""

import os
import shutil
import tempfile
from pathlib import Path


def remove_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted {path}")
    else:
        print(f"Skipped {path} (not found)")


def main():
    paths: set[Path] = {
        Path(os.path.expanduser("~/data/dev/shm/cache")),
    }

    # tiktoken cache location matches the library's search order
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        paths.add(Path(os.environ["TIKTOKEN_CACHE_DIR"]))
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        paths.add(Path(os.environ["DATA_GYM_CACHE_DIR"]))
    else:
        paths.add(Path(tempfile.gettempdir()) / "data-gym-cache")

    for p in sorted(paths):
        remove_dir(p)


if __name__ == "__main__":
    main()
