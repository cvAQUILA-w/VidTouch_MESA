from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .data import (
    LabelVocab,
    VidTouchPairDataset,
    build_record_partitions_from_ids,
    build_record_splits,
    build_record_splits_from_ids,
    scan_vidtouch,
)
from .metrics import (
    material_knowledge_discovery_score,
    mean_legacy_score,
    mean_main_score,
)
from .model import VidTouchMaterialModel
from .train import evaluate, make_loader
from .utils import load_json, unwrap_for_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VidTouch checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-threshold", type=float, default=None)
    parser.add_argument("--partition", choices=("val", "test"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = copy.deepcopy(ckpt["config"])
    if args.feature_threshold is not None:
        cfg["eval"]["feature_threshold"] = float(args.feature_threshold)
    vocab = LabelVocab.from_state_dict(ckpt["vocab"])
    records = scan_vidtouch(args.data_root, min_pairs_per_fabric=cfg["split"].get("min_pairs_per_fabric", 1))
    split_file = cfg["split"].get("file")
    if split_file:
        split_path = Path(split_file)
        if not split_path.is_absolute() and not split_path.exists():
            split_path = Path(__file__).resolve().parent.parent / split_path
        state = load_json(split_path)
        partition_ids = {
            name.removesuffix("_ids"): list(ids)
            for name, ids in state.items()
            if name.endswith("_ids") and isinstance(ids, list)
        }
        partitions, _ = build_record_partitions_from_ids(records, partition_ids)
        eval_partition = args.partition or str(cfg["split"].get("eval_partition", "val"))
        if eval_partition not in partitions:
            raise ValueError(f"Split does not define partition: {eval_partition}")
        if eval_partition == "test":
            if args.feature_threshold is None:
                raise ValueError(
                    "Test evaluation requires --feature-threshold calibrated on Validation"
                )
            cfg["eval"]["calibrate_feature_threshold"] = False
        val_records = partitions[eval_partition]
    else:
        split_seed = int(cfg["split"].get("seed", cfg["seed"]))
        _, val_records, _ = build_record_splits(
            records,
            cfg["split"]["mode"],
            cfg["split"]["val_ratio"],
            split_seed,
        )
    video_cache_root = cfg["data"].get("video_cache_root")
    if isinstance(video_cache_root, str):
        cache_mode = video_cache_root.lower()
        if cache_mode in {"auto", "auto_f16", "auto_float16"}:
            video_cache_root = str(Path(args.data_root) / ".cache")
        elif cache_mode in {"auto_f32", "auto_float32"}:
            video_cache_root = str(Path(args.data_root) / ".cache_float32")
    val_ds = VidTouchPairDataset(
        val_records,
        vocab,
        [r.fabric_id for r in val_records],
        image_size=cfg["data"]["image_size"],
        video_size=cfg["data"]["video_size"],
        video_frames=cfg["data"]["video_frames"],
        training=False,
        split_mode=cfg["split"]["mode"],
        repeats_per_epoch=1,
        deterministic_val_pairs=cfg["data"].get("deterministic_val_pairs", True),
        image_views_per_sample=cfg["data"].get(
            "val_image_views_per_sample",
            cfg["data"].get("image_views_per_sample", 1),
        ),
        video_views_per_sample=cfg["data"].get(
            "val_video_views_per_sample",
            cfg["data"].get("video_views_per_sample", 1),
        ),
        sample_with_replacement=cfg["data"].get("sample_with_replacement", True),
        video_cache_root=video_cache_root,
        val_fabric_sets=cfg["data"].get("val_fabric_sets", False),
    )
    loader = make_loader(val_ds, cfg["train"]["batch_size"], cfg["data"]["num_workers"], training=False)
    sizes = vocab.sizes
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg.setdefault("num_material_components", sizes["material_components"])
    model_cfg.setdefault("num_material_primaries", sizes["material_primaries"])
    model_cfg.setdefault("num_usage_supercategories", sizes["usage_supercategories"])
    material_component_matrix, material_primary_indices = vocab.material_semantic_layout()
    model_cfg.setdefault("material_component_matrix", material_component_matrix)
    model_cfg.setdefault("material_primary_indices", material_primary_indices)
    model = VidTouchMaterialModel(
        num_weaves=sizes["weave"],
        num_materials=sizes["material"],
        num_usages=sizes["usage"],
        num_features=sizes["features"],
        **model_cfg,
    )
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    evaluated = evaluate(
        model,
        loader,
        device,
        cfg,
        return_predictions=args.predictions_output is not None,
    )
    if args.predictions_output:
        metrics, predictions = evaluated
    else:
        metrics = evaluated
        predictions = None
    metrics["legacy_main_score"] = mean_legacy_score(metrics)
    metrics["main_score"] = mean_main_score(metrics)
    metrics["mkds"] = material_knowledge_discovery_score(metrics)
    if split_file:
        metrics["partition"] = eval_partition
    text = json.dumps(unwrap_for_json(metrics), ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.predictions_output:
        prediction_path = Path(args.predictions_output)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in predictions or []:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
