from __future__ import annotations

import math

import torch


def top1_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> float:
    valid = target != ignore_index
    if not valid.any():
        return 0.0
    logits = logits[valid]
    target = target[valid]
    pred = logits.argmax(dim=-1)
    return (pred == target).float().mean().item()


def multiclass_macro_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -1,
) -> dict[str, float]:
    valid = target != ignore_index
    if not valid.any():
        return {"balanced_acc": 0.0, "f1_macro": 0.0}
    pred = logits[valid].argmax(dim=-1)
    target = target[valid]
    recalls = []
    f1s = []
    for cls in target.unique(sorted=True):
        true_cls = target == cls
        pred_cls = pred == cls
        tp = (true_cls & pred_cls).sum().float()
        fn = (true_cls & ~pred_cls).sum().float()
        fp = (~true_cls & pred_cls).sum().float()
        recalls.append(tp / (tp + fn).clamp_min(1.0))
        f1s.append(2 * tp / (2 * tp + fp + fn).clamp_min(1.0))
    return {
        "balanced_acc": torch.stack(recalls).mean().item(),
        "f1_macro": torch.stack(f1s).mean().item(),
    }


def multilabel_f1(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float | torch.Tensor = 0.5,
) -> dict[str, float]:
    if logits.numel() == 0:
        return {"feature_f1_micro": 0.0, "feature_f1_macro": 0.0}
    pred = (logits.sigmoid() >= threshold).float()
    target = target.float()
    tp = (pred * target).sum(dim=0)
    fp = (pred * (1.0 - target)).sum(dim=0)
    fn = ((1.0 - pred) * target).sum(dim=0)
    f1_per = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-8)
    valid = target.sum(dim=0) > 0
    macro = f1_per[valid].mean().item() if valid.any() else 0.0
    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    micro = (2 * tp_micro / (2 * tp_micro + fp_micro + fn_micro).clamp_min(1e-8)).item()
    return {"feature_f1_micro": micro, "feature_f1_macro": macro}


def best_global_multilabel_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = thresholds or [round(value, 2) for value in torch.arange(0.1, 0.91, 0.05).tolist()]
    scored = [(threshold, multilabel_f1(logits, target, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["feature_f1_macro"], item[1]["feature_f1_micro"]))


def retrieval_recall(
    image_emb: torch.Tensor,
    tactile_emb: torch.Tensor,
    fabric: torch.Tensor,
    topk: list[int],
) -> dict[str, float]:
    sim = image_emb @ tactile_emb.t()
    out: dict[str, float] = {}
    max_k = min(max(topk), sim.shape[0])
    i2t = sim.topk(max_k, dim=1).indices
    t2i = sim.t().topk(max_k, dim=1).indices
    fabric = fabric.cpu()
    for k in topk:
        kk = min(k, max_k)
        ok_i2t = []
        ok_t2i = []
        for i in range(sim.shape[0]):
            ok_i2t.append((fabric[i2t[i, :kk].cpu()] == fabric[i]).any().item())
            ok_t2i.append((fabric[t2i[i, :kk].cpu()] == fabric[i]).any().item())
        out[f"retrieval_i2t_r{k}"] = float(sum(ok_i2t) / len(ok_i2t))
        out[f"retrieval_t2i_r{k}"] = float(sum(ok_t2i) / len(ok_t2i))
        out[f"retrieval_mean_r{k}"] = 0.5 * (out[f"retrieval_i2t_r{k}"] + out[f"retrieval_t2i_r{k}"])
    return out


def mean_main_score(metrics: dict[str, float]) -> float:
    keys = ["weave_balanced_acc", "material_balanced_acc", "usage_balanced_acc"]
    values = [metrics[k] for k in keys if k in metrics]
    feature_key = (
        "feature_f1_macro_calibrated"
        if "feature_f1_macro_calibrated" in metrics
        else "feature_f1_macro"
    )
    if feature_key in metrics:
        values.append(metrics[feature_key])
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def material_knowledge_discovery_score(metrics: dict[str, float]) -> float:
    """Harmonic aggregate of the four equal-class material-semantic axes."""
    feature_key = (
        "feature_f1_macro_calibrated"
        if "feature_f1_macro_calibrated" in metrics
        else "feature_f1_macro"
    )
    keys = [
        "weave_balanced_acc",
        "material_balanced_acc",
        "usage_balanced_acc",
        feature_key,
    ]
    if any(key not in metrics for key in keys):
        return 0.0
    values = [float(metrics[key]) for key in keys]
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        return 0.0
    return float(len(values) / sum(1.0 / value for value in values))


def mean_legacy_score(metrics: dict[str, float]) -> float:
    keys = ["weave_acc", "material_acc", "usage_acc", "feature_f1_micro"]
    values = [metrics[k] for k in keys if k in metrics]
    if not values:
        return 0.0
    return float(sum(values) / len(values))
