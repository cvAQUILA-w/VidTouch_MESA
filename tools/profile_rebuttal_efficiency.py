#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from vidtouch.data import (
    LabelVocab,
    VidTouchPairDataset,
    build_record_partitions_from_ids,
    scan_vidtouch,
)
from vidtouch.model import VidTouchMaterialModel
from vidtouch.train import make_loader
from vidtouch.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VidTouch checkpoint inference")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def build(checkpoint_path: Path, data_root: Path) -> tuple[torch.nn.Module, dict, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = copy.deepcopy(checkpoint["config"])
    vocab = LabelVocab.from_state_dict(checkpoint["vocab"])
    records = scan_vidtouch(data_root, min_pairs_per_fabric=cfg["split"].get("min_pairs_per_fabric", 1))
    split_path = Path(cfg["split"]["file"])
    if not split_path.is_absolute() and not split_path.exists():
        split_path = Path(__file__).resolve().parent.parent / split_path
    state = load_json(split_path)
    partitions, _ = build_record_partitions_from_ids(
        records,
        {name.removesuffix("_ids"): ids for name, ids in state.items() if name.endswith("_ids")},
    )
    video_cache_root = cfg["data"].get("video_cache_root")
    if isinstance(video_cache_root, str) and video_cache_root.lower() in {
        "auto", "auto_f16", "auto_float16"
    }:
        video_cache_root = str(data_root / ".cache")
    dataset = VidTouchPairDataset(
        partitions["val"],
        vocab,
        [record.fabric_id for record in partitions["val"]],
        image_size=cfg["data"]["image_size"],
        video_size=cfg["data"]["video_size"],
        video_frames=cfg["data"]["video_frames"],
        training=False,
        split_mode=cfg["split"]["mode"],
        repeats_per_epoch=1,
        deterministic_val_pairs=True,
        image_views_per_sample=cfg["data"].get("val_image_views_per_sample", 2),
        video_views_per_sample=cfg["data"].get("val_video_views_per_sample", 2),
        sample_with_replacement=cfg["data"].get("sample_with_replacement", True),
        video_cache_root=video_cache_root,
        val_fabric_sets=cfg["data"].get("val_fabric_sets", False),
    )
    loader = make_loader(dataset, 1, 0, training=False)
    batch = next(iter(loader))
    sizes = vocab.sizes
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg.setdefault("num_material_components", sizes["material_components"])
    model_cfg.setdefault("num_material_primaries", sizes["material_primaries"])
    model_cfg.setdefault("num_usage_supercategories", sizes["usage_supercategories"])
    component_matrix, primary_indices = vocab.material_semantic_layout()
    model_cfg.setdefault("material_component_matrix", component_matrix)
    model_cfg.setdefault("material_primary_indices", primary_indices)
    model = VidTouchMaterialModel(
        num_weaves=sizes["weave"],
        num_materials=sizes["material"],
        num_usages=sizes["usage"],
        num_features=sizes["features"],
        **model_cfg,
    )
    model.load_state_dict(checkpoint["model"])
    return model, cfg, batch


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the rebuttal efficiency profile")
    device = torch.device("cuda")
    results = {}
    for specification in args.checkpoint:
        name, raw_path = specification.split("=", 1)
        model, cfg, batch = build(Path(raw_path), args.data_root)
        model.eval().to(device)
        image = batch["image"].to(device)
        video = batch["video"].to(device)
        use_amp = bool(cfg["train"].get("amp", False))

        with torch.inference_mode():
            for _ in range(args.warmup):
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    model(image, video)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(args.repeats):
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    model(image, video)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

        results[name] = {
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "model_only_latency_ms_per_fabric": 1000.0 * elapsed / args.repeats,
            "throughput_fabrics_per_sec_batch1": args.repeats / elapsed,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "evidence": {
                "rgb_views": int(image.shape[1]) if image.ndim == 5 else 1,
                "tactile_views": int(video.shape[1]) if video.ndim == 6 else 1,
                "frames_per_tactile_view": int(video.shape[-4]),
            },
        }
        del model, image, video
        torch.cuda.empty_cache()

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "timed_repeats": args.repeats,
        "scope": "model forward only; excludes media decoding and host-to-device transfer",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
