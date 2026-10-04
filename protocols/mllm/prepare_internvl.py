#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    configured = os.environ.get("INTERNVL25_MODEL_PATH")
    if configured and Path(configured).is_dir():
        print(configured)
        return
    cache_dir = Path(
        os.environ.get(
            "INTERNVL25_CACHE_DIR", "/root/autodl-tmp/internvl25_models"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download

    model_path = snapshot_download(
        "OpenGVLab/InternVL2_5-8B",
        cache_dir=str(cache_dir),
    )
    if not Path(model_path).is_dir():
        raise RuntimeError(f"InternVL download did not produce a directory: {model_path}")
    print(model_path)


if __name__ == "__main__":
    main()
