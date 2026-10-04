#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from evaluate_benchmark import (
    EXPECTED_ASSIGNMENT_SHA256,
    EXPECTED_LABELS_SHA256,
    EXPECTED_SPLIT_SHA256,
    sha256,
)


DATASET_ROOT = Path(os.environ.get("VIDTOUCH_DATASET_ROOT", "/root/VBTSINT_DATASET"))
SPLIT_PATH = DATASET_ROOT / "benchmark/splits/fabric_common_v2.json"
LABEL_PATH = DATASET_ROOT / "label.txt"
RGB_DIR = DATASET_ROOT / "RGBs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the remote benchmark setup")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Validate data and protocol files without requiring CUDA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_hash = sha256(SPLIT_PATH)
    labels_hash = sha256(LABEL_PATH)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Wrong split SHA256: {split_hash}")
    if labels_hash != EXPECTED_LABELS_SHA256:
        raise RuntimeError(f"Wrong label.txt SHA256: {labels_hash}")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split["metadata"]["assignment_sha256"] != EXPECTED_ASSIGNMENT_SHA256:
        raise RuntimeError("Wrong Fabric assignment SHA256")
    partition_ids = {
        fabric_id
        for partition in ("train_ids", "val_ids", "test_ids")
        for fabric_id in split[partition]
    }
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    rgb_paths = [
        path
        for path in RGB_DIR.iterdir()
        if path.is_file() and path.suffix.casefold() in image_suffixes
    ]
    rgb_count = sum(
        any(
            path.stem == fabric_id or path.name.startswith(f"{fabric_id} ")
            for fabric_id in partition_ids
        )
        for path in rgb_paths
    )
    missing_rgb_ids = [
        fabric_id
        for fabric_id in sorted(partition_ids)
        if not any(
            path.stem == fabric_id or path.name.startswith(f"{fabric_id} ")
            for path in rgb_paths
        )
    ]
    if missing_rgb_ids:
        raise RuntimeError(f"Frozen Fabric IDs without RGB images: {missing_rgb_ids}")
    if rgb_count != 435:
        raise RuntimeError(
            f"Expected 435 RGB files for frozen Fabric IDs, found {rgb_count}"
        )
    free_gib = shutil.disk_usage("/root/autodl-tmp").free / (1024**3)
    if not args.data_only and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Switch the AutoDL instance from no-card mode "
            "to GPU mode before launching the benchmark."
        )
    device = torch.cuda.get_device_properties(0) if not args.data_only else None
    report = {
        "valid": True,
        "mode": "data-only" if args.data_only else "full",
        "gpu": device.name if device else None,
        "gpu_memory_gib": (
            round(device.total_memory / (1024**3), 2) if device else None
        ),
        "data_disk_free_gib": round(free_gib, 2),
        "split_sha256": split_hash,
        "labels_sha256": labels_hash,
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "partition_sizes": {
            "train": len(split["train_ids"]),
            "val": len(split["val_ids"]),
            "test": len(split["test_ids"]),
        },
        "rgb_count": rgb_count,
        "ignored_rgb_count": len(rgb_paths) - rgb_count,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
