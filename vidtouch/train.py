from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    LabelVocab,
    VidTouchPairDataset,
    build_common_label_filter,
    build_record_partitions_from_ids,
    build_record_splits,
    build_record_splits_from_ids,
    scan_vidtouch,
)
from .losses import VidTouchCriterion
from .metrics import (
    best_global_multilabel_threshold,
    material_knowledge_discovery_score,
    mean_legacy_score,
    mean_main_score,
    multiclass_macro_metrics,
    multilabel_f1,
    retrieval_recall,
    top1_accuracy,
)
from .model import VidTouchMaterialModel
from .utils import AverageMeter, load_json, load_yaml, save_json, save_yaml, seed_everything, unwrap_for_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VidTouch material-aware model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    return parser.parse_args()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def make_loader(
    dataset: VidTouchPairDataset,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
        persistent_workers=(num_workers > 0),
    )


def build_label_statistics(records: list[Any], vocab: LabelVocab) -> dict[str, torch.Tensor]:
    sizes = vocab.sizes
    stats = {
        "weave": torch.zeros(sizes["weave"], dtype=torch.float32),
        "material": torch.zeros(sizes["material"], dtype=torch.float32),
        "usage": torch.zeros(sizes["usage"], dtype=torch.float32),
        "features": torch.zeros(sizes["features"], dtype=torch.float32),
        "num_records": torch.tensor(float(len(records)), dtype=torch.float32),
        "num_feature_records": torch.tensor(0.0, dtype=torch.float32),
    }
    for record in records:
        labels = vocab.encode(record)
        if int(labels["weave"]) >= 0:
            stats["weave"][int(labels["weave"])] += 1.0
        if int(labels["material"]) >= 0:
            stats["material"][int(labels["material"])] += 1.0
        if int(labels["usage"]) >= 0:
            stats["usage"][int(labels["usage"])] += 1.0
        features = labels["features"].float()
        stats["features"] += features
        if features.sum() > 0:
            stats["num_feature_records"] += 1.0
    return stats


def build_everything(cfg: dict[str, Any], data_root: str) -> tuple[
    VidTouchMaterialModel,
    VidTouchCriterion,
    LabelVocab,
    DataLoader,
    DataLoader,
    dict[str, Any],
]:
    records = scan_vidtouch(data_root, min_pairs_per_fabric=cfg["split"].get("min_pairs_per_fabric", 1))
    filter_cfg = cfg.get("label_filter") or {}
    filter_source = str(filter_cfg.get("count_source", "train_records"))
    allowed_labels = None
    label_filter_state = None
    split_file = cfg["split"].get("file")
    fixed_split = None
    split_path = None
    if split_file:
        split_path = Path(split_file)
        if not split_path.is_absolute() and not split_path.exists():
            split_path = Path(__file__).resolve().parent.parent / split_path
        fixed_split = load_json(split_path)

    if filter_source == "split_manifest":
        if fixed_split is None:
            raise ValueError("label_filter.count_source=split_manifest requires a fixed split file")
        label_filter_state = fixed_split.get("metadata", {}).get("label_filter")
        if not isinstance(label_filter_state, dict) or not isinstance(
            label_filter_state.get("allowed_labels"), dict
        ):
            raise ValueError("Fixed split does not contain metadata.label_filter.allowed_labels")
        allowed_labels = {
            name: list(labels)
            for name, labels in label_filter_state["allowed_labels"].items()
        }
    elif filter_source == "all_records":
        allowed_labels, label_filter_state = build_common_label_filter(
            records,
            filter_cfg,
            count_source="all_records",
        )

    if fixed_split is not None:
        partition_ids = {
            name.removesuffix("_ids"): list(ids)
            for name, ids in fixed_split.items()
            if name.endswith("_ids") and isinstance(ids, list)
        }
        partitions, split_state = build_record_partitions_from_ids(records, partition_ids)
        train_partition_names = list(cfg["split"].get("train_partitions", ["train"]))
        eval_partition = str(cfg["split"].get("eval_partition", "val"))
        unknown_partitions = (set(train_partition_names) | {eval_partition}) - set(partitions)
        if unknown_partitions:
            raise ValueError(f"Unknown fixed split partitions: {sorted(unknown_partitions)}")
        if eval_partition in train_partition_names:
            raise ValueError("Evaluation partition cannot also be a training partition")
        train_records = [
            record
            for name in train_partition_names
            for record in partitions[name]
        ]
        val_records = partitions[eval_partition]
        split_state["train_partitions"] = train_partition_names
        split_state["eval_partition"] = eval_partition
        split_state["file"] = str(split_path)
        split_state["metadata"] = fixed_split.get("metadata", {})
    else:
        split_seed = int(cfg["split"].get("seed", cfg["seed"]))
        train_records, val_records, split_state = build_record_splits(
            records,
            mode=cfg["split"]["mode"],
            val_ratio=float(cfg["split"]["val_ratio"]),
            seed=split_seed,
        )
        split_state["seed"] = split_seed

    if filter_source == "train_records":
        allowed_labels, label_filter_state = build_common_label_filter(
            train_records,
            filter_cfg,
            count_source="train_records",
        )
    elif filter_source not in {"all_records", "split_manifest"}:
        raise ValueError(f"Unsupported label_filter.count_source: {filter_source}")
    vocab = LabelVocab(records, allowed_labels=allowed_labels)
    if label_filter_state is not None:
        split_state["label_filter"] = label_filter_state
    video_cache_root = cfg["data"].get("video_cache_root")
    if isinstance(video_cache_root, str):
        cache_mode = video_cache_root.lower()
        if cache_mode in {"auto", "auto_f16", "auto_float16"}:
            video_cache_root = str(Path(data_root) / ".cache")
        elif cache_mode in {"auto_f32", "auto_float32"}:
            video_cache_root = str(Path(data_root) / ".cache_float32")
    train_ds = VidTouchPairDataset(
        train_records,
        vocab,
        [r.fabric_id for r in train_records],
        image_size=cfg["data"]["image_size"],
        video_size=cfg["data"]["video_size"],
        video_frames=cfg["data"]["video_frames"],
        training=True,
        split_mode=cfg["split"]["mode"],
        repeats_per_epoch=cfg["data"]["train_repeats_per_epoch"],
        deterministic_val_pairs=cfg["data"].get("deterministic_val_pairs", True),
        image_views_per_sample=cfg["data"].get("image_views_per_sample", 1),
        video_views_per_sample=cfg["data"].get("video_views_per_sample", 1),
        pair_recombination=cfg["data"].get("pair_recombination", True),
        sample_with_replacement=cfg["data"].get("sample_with_replacement", True),
        video_cache_root=video_cache_root,
        val_fabric_sets=cfg["data"].get("val_fabric_sets", False),
    )
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
        image_views_per_sample=cfg["data"].get("val_image_views_per_sample", cfg["data"].get("image_views_per_sample", 1)),
        video_views_per_sample=cfg["data"].get("val_video_views_per_sample", cfg["data"].get("video_views_per_sample", 1)),
        pair_recombination=cfg["data"].get("pair_recombination", True),
        sample_with_replacement=cfg["data"].get("sample_with_replacement", True),
        video_cache_root=video_cache_root,
        val_fabric_sets=cfg["data"].get("val_fabric_sets", False),
    )
    train_loader = make_loader(
        train_ds,
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"],
        training=True,
    )
    val_loader = make_loader(
        val_ds,
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"],
        training=False,
    )
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
    loss_cfg = copy.deepcopy(cfg["loss"])
    loss_cfg.pop("schedule", None)
    criterion = VidTouchCriterion(**loss_cfg)
    criterion.set_class_statistics(build_label_statistics(train_records, vocab))
    return model, criterion, vocab, train_loader, val_loader, split_state


def get_scaler(
    enabled: bool,
    init_scale: float = 1024.0,
    growth_interval: int = 2000,
) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
            init_scale=init_scale,
            growth_interval=growth_interval,
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
            init_scale=init_scale,
            growth_interval=growth_interval,
        )


def apply_loss_schedule(
    criterion: VidTouchCriterion,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    schedule = cfg.get("loss", {}).get("schedule")
    active: dict[str, Any] | None = None
    active_index = -1
    if isinstance(schedule, list):
        for index, stage in enumerate(schedule):
            if not isinstance(stage, dict):
                continue
            start = int(stage.get("start_epoch", 1))
            end_value = stage.get("end_epoch", stage.get("until_epoch"))
            end = int(end_value) if end_value is not None else None
            if epoch >= start and (end is None or epoch <= end):
                active = stage
                active_index = index
                break
    if active is not None:
        for key in ("temperature", "contrastive_weight", "cls_weight", "relation_weight", "queue_loss_weight"):
            if key in active:
                setattr(criterion, key, float(active[key]))
        if "memory_queue_size" in active:
            criterion.memory_queue_size = int(active["memory_queue_size"])
        if isinstance(active.get("positive_weights"), dict):
            criterion.positive_weights = dict(active["positive_weights"])
        if isinstance(active.get("classification"), dict):
            updated = dict(criterion.classification)
            updated.update(active["classification"])
            criterion.classification = updated

    return {
        "stage": active.get("name", f"stage_{active_index + 1}") if active is not None else "static",
        "temperature": float(criterion.temperature),
        "contrastive_weight": float(criterion.contrastive_weight),
        "cls_weight": float(criterion.cls_weight),
        "relation_weight": float(criterion.relation_weight),
        "queue_loss_weight": float(criterion.queue_loss_weight),
    }


def train_one_epoch(
    model: VidTouchMaterialModel,
    criterion: VidTouchCriterion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    if bool(cfg["train"].get("freeze_frozen_batchnorm", False)):
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
    warmup_epochs = int(cfg["train"].get("backbone_warmup_epochs", 0))
    if epoch <= warmup_epochs:
        model.visual.backbone.eval()
        tactile_backbone = getattr(model.tactile, "backbone", None)
        if tactile_backbone is not None:
            tactile_backbone.eval()
    meters: dict[str, AverageMeter] = {}
    fp32_retries = 0
    amp_overflow_steps = 0
    nonfinite_gradient_steps = 0
    max_batches = cfg["train"].get("max_train_batches")
    pbar = tqdm(loader, desc=f"train {epoch}", dynamic_ncols=True)
    for step, batch in enumerate(pbar, 1):
        if max_batches is not None and step > int(max_batches):
            break
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(batch["image"], batch["video"])
            loss, logs = criterion(output, batch, model)
        if (
            not torch.isfinite(loss).all()
            and amp_enabled
            and bool(cfg["train"].get("amp_retry_on_nonfinite", True))
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                output = model(batch["image"], batch["video"])
                loss, logs = criterion(output, batch, model)
            fp32_retries += 1
        if not torch.isfinite(loss).all():
            bad_components = [
                key for key, value in logs.items() if not torch.isfinite(value).all()
            ]
            bad_outputs = []
            for name, value in {
                "image_emb": output.image_emb,
                "tactile_emb": output.tactile_emb,
                "fused": output.fused,
                **{f"logits.{key}": value for key, value in output.logits.items()},
                **{
                    f"semantic_logits.{key}": value
                    for key, value in output.semantic_logits.items()
                },
            }.items():
                if not torch.isfinite(value).all():
                    bad_outputs.append(name)
            message = (
                f"Non-finite loss at epoch={epoch}, step={step}: {float(loss.detach())}; "
                f"components={bad_components}; outputs={bad_outputs}; fp32_retry={amp_enabled}"
            )
            if bool(cfg["train"].get("fail_on_nonfinite", True)):
                raise FloatingPointError(message)
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        gradients_finite = True
        if cfg["train"].get("grad_clip"):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["train"]["grad_clip"]),
            )
            gradients_finite = bool(torch.isfinite(grad_norm))
        else:
            gradients_finite = all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.update()
            nonfinite_gradient_steps += 1
            continue
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before_step:
            amp_overflow_steps += 1
        criterion.update_queue(output, batch)
        logs["memory_queue_items"] = criterion.queue_image_emb.shape[0]
        bs = batch["fabric"].shape[0]
        for key, value in logs.items():
            meters.setdefault(key, AverageMeter()).update(float(value), bs)
        if step % int(cfg["train"]["log_every"]) == 0:
            pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in meters.items() if k in {"loss", "loss_contrastive", "loss_cls"}})
    result = {k: v.avg for k, v in meters.items()}
    result["amp_fp32_retries"] = float(fp32_retries)
    result["amp_overflow_steps"] = float(amp_overflow_steps)
    result["nonfinite_gradient_steps"] = float(nonfinite_gradient_steps)
    return result


@torch.no_grad()
def evaluate(
    model: VidTouchMaterialModel,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    max_batches = cfg["train"].get("max_val_batches")
    all_logits: dict[str, list[torch.Tensor]] = {"weave": [], "material": [], "usage": [], "features": []}
    all_targets: dict[str, list[torch.Tensor]] = {"weave": [], "material": [], "usage": [], "features": [], "fabric": []}
    image_embs: list[torch.Tensor] = []
    tactile_embs: list[torch.Tensor] = []
    all_unimodal_logits: dict[str, dict[str, list[torch.Tensor]]] = {}
    modality_weights: list[torch.Tensor] = []
    fabric_ids: list[str] = []
    for step, batch in enumerate(tqdm(loader, desc="val", dynamic_ncols=True), 1):
        if max_batches is not None and step > int(max_batches):
            break
        fabric_ids.extend(str(value) for value in batch["fabric_id"])
        batch = to_device(batch, device)
        output = model(batch["image"], batch["video"])
        for key in all_logits:
            all_logits[key].append(output.logits[key].detach().cpu())
        for key in all_targets:
            all_targets[key].append(batch[key].detach().cpu())
        image_embs.append(output.image_emb.detach().cpu())
        tactile_embs.append(output.tactile_emb.detach().cpu())
        for modality, modality_logits in output.unimodal_logits.items():
            store = all_unimodal_logits.setdefault(
                modality,
                {"weave": [], "material": [], "usage": [], "features": []},
            )
            for key, value in modality_logits.items():
                store[key].append(value.detach().cpu())
        if output.modality_weights is not None:
            modality_weights.append(output.modality_weights.detach().cpu())
    logits = {k: torch.cat(v, dim=0) for k, v in all_logits.items()}
    targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
    metrics = {
        "weave_acc": top1_accuracy(logits["weave"], targets["weave"]),
        "material_acc": top1_accuracy(logits["material"], targets["material"]),
        "usage_acc": top1_accuracy(logits["usage"], targets["usage"]),
        "weave_eval_count": float((targets["weave"] >= 0).sum().item()),
        "material_eval_count": float((targets["material"] >= 0).sum().item()),
        "usage_eval_count": float((targets["usage"] >= 0).sum().item()),
    }
    for attr in ("weave", "material", "usage"):
        macro = multiclass_macro_metrics(logits[attr], targets[attr])
        metrics[f"{attr}_balanced_acc"] = macro["balanced_acc"]
        metrics[f"{attr}_f1_macro"] = macro["f1_macro"]
    feature_valid = targets["features"].sum(dim=1) > 0
    metrics["feature_eval_count"] = float(feature_valid.sum().item())
    metrics.update(
        multilabel_f1(
            logits["features"][feature_valid],
            targets["features"][feature_valid],
            cfg["eval"]["feature_threshold"],
        )
    )
    if bool(cfg["eval"].get("calibrate_feature_threshold", True)):
        calibrated_threshold, calibrated = best_global_multilabel_threshold(
            logits["features"][feature_valid],
            targets["features"][feature_valid],
        )
        metrics["feature_best_global_threshold"] = calibrated_threshold
        metrics["feature_f1_micro_calibrated"] = calibrated["feature_f1_micro"]
        metrics["feature_f1_macro_calibrated"] = calibrated["feature_f1_macro"]
    for modality, stored in all_unimodal_logits.items():
        modality_logits = {key: torch.cat(values, dim=0) for key, values in stored.items()}
        for attr in ("weave", "material", "usage"):
            metrics[f"{modality}_{attr}_acc"] = top1_accuracy(modality_logits[attr], targets[attr])
        modality_feature = multilabel_f1(
            modality_logits["features"][feature_valid],
            targets["features"][feature_valid],
            cfg["eval"]["feature_threshold"],
        )
        metrics[f"{modality}_feature_f1_micro"] = modality_feature["feature_f1_micro"]
    if modality_weights:
        weights = torch.cat(modality_weights, dim=0).mean(dim=0)
        token_names = getattr(model, "modality_token_names", ("image", "tactile", "interaction"))
        for attr_index, attr in enumerate(("weave", "material", "usage", "features")):
            for token_index, token in enumerate(token_names):
                metrics[f"attention_{attr}_{token}"] = float(weights[attr_index, token_index])
    for attr, gate in getattr(model, "attribute_tactile_gates", {}).items():
        metrics[f"attribute_tactile_gate_{attr}"] = float(torch.sigmoid(gate.detach()).cpu())
    image_emb = torch.cat(image_embs, dim=0)
    tactile_emb = torch.cat(tactile_embs, dim=0)
    metrics.update(retrieval_recall(image_emb, tactile_emb, targets["fabric"], cfg["eval"]["retrieval_topk"]))
    metrics["legacy_main_score"] = mean_legacy_score(metrics)
    metrics["main_score"] = mean_main_score(metrics)
    metrics["mkds"] = material_knowledge_discovery_score(metrics)
    if return_predictions:
        feature_pred = (
            logits["features"].sigmoid() >= float(cfg["eval"]["feature_threshold"])
        ).to(torch.int64)
        rows: list[dict[str, Any]] = []
        for index, fabric_id in enumerate(fabric_ids):
            rows.append(
                {
                    "fabric_id": fabric_id,
                    "target": {
                        "weave": int(targets["weave"][index]),
                        "material": int(targets["material"][index]),
                        "usage": int(targets["usage"][index]),
                        "features": targets["features"][index].to(torch.int64).tolist(),
                    },
                    "prediction": {
                        "weave": int(logits["weave"][index].argmax()),
                        "material": int(logits["material"][index].argmax()),
                        "usage": int(logits["usage"][index].argmax()),
                        "features": feature_pred[index].tolist(),
                    },
                }
            )
        return metrics, rows
    return metrics


def save_checkpoint(
    path: Path,
    model: VidTouchMaterialModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_macro_score: float,
    best_legacy_score: float,
    cfg: dict[str, Any],
    vocab: LabelVocab,
    checkpoint_role: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_macro_score,
            "best_macro_score": best_macro_score,
            "best_legacy_score": best_legacy_score,
            "checkpoint_role": checkpoint_role,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "vocab": vocab.state_dict(),
        },
        path,
    )


def update_checkpoint_alias(source: Path, alias: Path) -> None:
    alias.unlink(missing_ok=True)
    try:
        os.link(source, alias)
    except OSError:
        shutil.copy2(source, alias)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed_everything(int(cfg["seed"]))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config_resolved.yaml")

    model, criterion, vocab, train_loader, val_loader, split_state = build_everything(cfg, args.data_root)
    save_json(vocab.state_dict(), out_dir / "vocab.json")
    save_json(split_state, out_dir / "split.json")

    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        core_vocab_keys = ("fabric_ids", "weaves", "materials", "usages", "features")
        init_vocab = init_ckpt.get("vocab") or {}
        current_vocab = vocab.state_dict()
        if any(init_vocab.get(key) != current_vocab.get(key) for key in core_vocab_keys):
            raise ValueError("Initialization checkpoint vocabulary does not match this run")
        incompatible = model.load_state_dict(init_ckpt["model"], strict=False)
        print(f"Initialized from: {args.init_checkpoint}")
        print(f"Missing initialization keys: {incompatible.missing_keys}")
        print(f"Unexpected initialization keys: {incompatible.unexpected_keys}")

    trainable_prefixes = tuple(cfg["train"].get("trainable_parameter_prefixes", ()))
    if trainable_prefixes:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names:
            raise ValueError(f"No parameters match trainable prefixes: {trainable_prefixes}")
        print(f"Trainable parameter prefixes: {trainable_prefixes}")
        print(f"Trainable parameters: {trainable_names}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    criterion.to(device)
    base_lr = float(cfg["train"]["lr"])
    backbone_lr_mult = float(cfg["train"].get("backbone_lr_mult", 1.0))
    visual_backbone_parameters = []
    tactile_backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual.backbone."):
            visual_backbone_parameters.append(parameter)
        elif name.startswith("tactile.backbone.") or name.startswith(
            "attribute_tactile.backbone."
        ):
            tactile_backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    parameter_groups = [{"params": other_parameters, "lr": base_lr, "group_name": "main"}]
    if visual_backbone_parameters:
        parameter_groups.append(
            {
                "params": visual_backbone_parameters,
                "lr": base_lr * backbone_lr_mult,
                "group_name": "visual_backbone",
            }
        )
    tactile_backbone_lr_mult = float(
        cfg["train"].get("tactile_backbone_lr_mult", backbone_lr_mult)
    )
    if tactile_backbone_parameters:
        parameter_groups.append(
            {
                "params": tactile_backbone_parameters,
                "lr": base_lr * tactile_backbone_lr_mult,
                "group_name": "tactile_backbone",
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["train"]["epochs"])))
    scaler = get_scaler(
        bool(cfg["train"]["amp"]) and device.type == "cuda",
        init_scale=float(cfg["train"].get("amp_init_scale", 1024.0)),
        growth_interval=int(cfg["train"].get("amp_growth_interval", 2000)),
    )

    start_epoch = 1
    best_score = -math.inf
    best_legacy_score = -math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_macro_score", ckpt.get("best_score", best_score)))
        best_legacy_score = float(ckpt.get("best_legacy_score", best_legacy_score))

    warmup_parameters = visual_backbone_parameters + tactile_backbone_parameters

    metrics_path = out_dir / "metrics.jsonl"
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Vocab sizes: {vocab.sizes}")
    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        backbone_warmup_active = epoch <= int(cfg["train"].get("backbone_warmup_epochs", 0))
        for parameter in warmup_parameters:
            parameter.requires_grad_(not backbone_warmup_active)
        schedule_state = apply_loss_schedule(criterion, cfg, epoch)
        t0 = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        group_lrs = {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, cfg, epoch)
        val_metrics = evaluate(model, val_loader, device, cfg)
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "backbone_lr": group_lrs.get("visual_backbone", 0.0),
            "visual_backbone_lr": group_lrs.get("visual_backbone", 0.0),
            "tactile_backbone_lr": group_lrs.get("tactile_backbone", 0.0),
            "backbone_warmup_active": backbone_warmup_active,
            "time_sec": time.time() - t0,
            **{f"schedule_{k}": v for k, v in schedule_state.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(unwrap_for_json(row), ensure_ascii=False) + "\n")
        print(json.dumps(unwrap_for_json(row), ensure_ascii=False, indent=2))
        scheduler.step()
        score = val_metrics["main_score"]
        legacy_score = val_metrics["legacy_main_score"]
        is_best_macro = score > best_score
        is_best_legacy = legacy_score > best_legacy_score
        if is_best_macro:
            best_score = score
        if is_best_legacy:
            best_legacy_score = legacy_score
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            best_legacy_score,
            cfg,
            vocab,
            "last",
        )
        if is_best_macro:
            best_macro_path = out_dir / "best_macro.pt"
            save_checkpoint(
                best_macro_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                best_legacy_score,
                cfg,
                vocab,
                "best_macro",
            )
            update_checkpoint_alias(best_macro_path, out_dir / "best.pt")
        if is_best_legacy:
            save_checkpoint(
                out_dir / "best_legacy.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                best_legacy_score,
                cfg,
                vocab,
                "best_legacy",
            )
        if epoch % int(cfg["train"]["save_every"]) == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                best_legacy_score,
                cfg,
                vocab,
                f"epoch_{epoch:03d}",
            )


if __name__ == "__main__":
    main()
