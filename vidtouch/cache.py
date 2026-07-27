from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .data import VIDEO_EXTS, load_video_cv2, video_cache_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache VidTouch video tensors")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_video_paths(data_root: str | Path) -> list[Path]:
    tac_dir = Path(data_root) / "TACs"
    if not tac_dir.exists():
        raise FileNotFoundError(tac_dir)
    return sorted(path for path in tac_dir.iterdir() if path.suffix.lower() in VIDEO_EXTS)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root) if args.cache_root else data_root / ".cache"
    videos = iter_video_paths(data_root)
    cache_dir = cache_root / f"video_f{args.frames}_s{args.size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    started = time.time()
    written = 0
    skipped = 0
    for path in tqdm(videos, desc="cache videos", dynamic_ncols=True):
        out_path = video_cache_file(cache_root, path, args.frames, args.size)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        tensor = load_video_cv2(str(path), args.frames, args.size).to(dtype=dtype)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(tensor, tmp_path)
        tmp_path.replace(out_path)
        written += 1
    elapsed = time.time() - started
    print(
        f"Cached {written} videos, skipped {skipped}, "
        f"cache_root={cache_root}, elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
