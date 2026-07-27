from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import ModelOutput, VidTouchMaterialModel


def _feature_overlap(features: torch.Tensor) -> torch.Tensor:
    inter = features @ features.t()
    denom = features.sum(dim=1, keepdim=True).clamp_min(1.0)
    return inter / denom


def _feature_overlap_cross(query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
    inter = query_features @ candidate_features.t()
    denom = query_features.sum(dim=1, keepdim=True).clamp_min(1.0)
    return inter / denom


def label_aware_target_matrix(
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    candidate_batch: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    device = batch["fabric"].device
    candidate_batch = batch if candidate_batch is None else candidate_batch
    n = batch["fabric"].shape[0]
    m = candidate_batch["fabric"].shape[0]
    target = torch.zeros((n, m), device=device, dtype=torch.float32)
    target += weights.get("fabric", 1.0) * (
        batch["fabric"][:, None] == candidate_batch["fabric"][None, :]
    ).float()
    weave_match = (
        (batch["weave"][:, None] >= 0)
        & (candidate_batch["weave"][None, :] >= 0)
        & (batch["weave"][:, None] == candidate_batch["weave"][None, :])
    )
    target += weights.get("weave", 0.0) * (
        weave_match.float()
    )
    material_match = (
        (batch["material"][:, None] >= 0)
        & (candidate_batch["material"][None, :] >= 0)
        & (batch["material"][:, None] == candidate_batch["material"][None, :])
    )
    target += weights.get("material", 0.0) * (
        material_match.float()
    )
    usage_match = (
        (batch["usage"][:, None] >= 0)
        & (candidate_batch["usage"][None, :] >= 0)
        & (batch["usage"][:, None] == candidate_batch["usage"][None, :])
    )
    target += weights.get("usage", 0.0) * (
        usage_match.float()
    )
    if weights.get("feature", 0.0) > 0 and batch["features"].numel() > 0:
        target += weights.get("feature", 0.0) * _feature_overlap_cross(
            batch["features"],
            candidate_batch["features"],
        )
    pair_weight = float(weights.get("pair", 0.0) or 0.0)
    if pair_weight > 0 and m >= n:
        target[:, :n] += pair_weight * torch.eye(n, device=device)
    elif candidate_batch is batch:
        eye = torch.eye(n, device=device)
        target = torch.maximum(target, eye)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return target


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(target.float() * log_probs).sum(dim=-1).mean()


def symmetric_label_aware_contrastive(
    image_emb: torch.Tensor,
    tactile_emb: torch.Tensor,
    batch: dict[str, torch.Tensor],
    temperature: float,
    positive_weights: dict[str, float],
) -> torch.Tensor:
    logits = image_emb.float() @ tactile_emb.float().t() / temperature
    targets = label_aware_target_matrix(batch, positive_weights)
    loss_i2t = soft_cross_entropy(logits, targets)
    loss_t2i = soft_cross_entropy(logits.t(), targets.t())
    return 0.5 * (loss_i2t + loss_t2i)


def directional_label_aware_contrastive(
    query_emb: torch.Tensor,
    candidate_emb: torch.Tensor,
    query_batch: dict[str, torch.Tensor],
    candidate_batch: dict[str, torch.Tensor],
    temperature: float,
    positive_weights: dict[str, float],
) -> torch.Tensor:
    logits = query_emb.float() @ candidate_emb.float().t() / temperature
    targets = label_aware_target_matrix(query_batch, positive_weights, candidate_batch)
    return soft_cross_entropy(logits, targets)


def _effective_number_weight(counts: torch.Tensor, beta: float) -> torch.Tensor:
    counts = counts.float()
    weights = torch.zeros_like(counts)
    valid = counts > 0
    if not valid.any():
        return weights
    if beta <= 0:
        weights[valid] = 1.0
    else:
        valid_counts = counts[valid].clamp_min(1.0)
        weights[valid] = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=counts.device), valid_counts))
    weights[valid] = weights[valid] / weights[valid].mean().clamp_min(1e-8)
    return weights


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    label_smoothing: float,
    gamma: float,
) -> torch.Tensor:
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits[valid].float()
    target = target[valid]
    weight = weight.float() if weight is not None else None
    ce = F.cross_entropy(
        logits,
        target,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if gamma <= 0:
        return ce.mean()
    pt = logits.softmax(dim=-1).gather(1, target[:, None]).squeeze(1).clamp_min(1e-6)
    return (((1.0 - pt) ** gamma) * ce).mean()


def asymmetric_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None,
    gamma_pos: float,
    gamma_neg: float,
) -> torch.Tensor:
    valid = target.sum(dim=1) > 0
    if logits.numel() == 0 or not valid.any():
        return logits.sum() * 0.0
    logits = logits[valid].float()
    target = target[valid].float()
    pos_weight = pos_weight.float() if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    if gamma_pos <= 0 and gamma_neg <= 0:
        return bce.mean()
    prob = logits.sigmoid()
    pt = torch.where(target > 0, prob, 1.0 - prob).clamp_min(1e-6)
    gamma = torch.where(
        target > 0,
        torch.full_like(target, gamma_pos),
        torch.full_like(target, gamma_neg),
    )
    return (((1.0 - pt) ** gamma) * bce).mean()


def balanced_softmax_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_count: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0.0
    prior = class_count.to(device=logits.device, dtype=torch.float32).clamp_min(1.0).log()
    adjusted_logits = logits[valid].float() + prior.unsqueeze(0)
    return F.cross_entropy(
        adjusted_logits,
        target[valid],
        label_smoothing=float(label_smoothing),
    )


class VidTouchCriterion(nn.Module):
    def __init__(
        self,
        temperature: float,
        contrastive_weight: float,
        cls_weight: float,
        relation_weight: float,
        positive_weights: dict[str, float],
        memory_queue_size: int = 0,
        queue_loss_weight: float = 0.0,
        classification: dict[str, object] | None = None,
        unimodal_aux_weight: float = 0.0,
        unimodal_aux_attributes: list[str] | tuple[str, ...] | None = None,
        semantic_aux_weight: float = 0.0,
        semantic_aux_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.cls_weight = cls_weight
        self.relation_weight = relation_weight
        self.positive_weights = positive_weights
        self.memory_queue_size = int(memory_queue_size)
        self.queue_loss_weight = float(queue_loss_weight)
        self.classification = classification or {}
        self.unimodal_aux_weight = float(unimodal_aux_weight)
        self.semantic_aux_weight = float(semantic_aux_weight)
        self.semantic_aux_weights = semantic_aux_weights or {}
        self.unimodal_aux_attributes = tuple(
            unimodal_aux_attributes or ("weave", "material", "usage", "features")
        )
        unknown_aux_attributes = set(self.unimodal_aux_attributes) - {
            "weave",
            "material",
            "usage",
            "features",
        }
        if unknown_aux_attributes:
            raise ValueError(f"Unknown unimodal auxiliary attributes: {sorted(unknown_aux_attributes)}")
        self.register_buffer("queue_image_emb", torch.empty(0, 0), persistent=False)
        self.register_buffer("queue_tactile_emb", torch.empty(0, 0), persistent=False)
        self.register_buffer("queue_fabric", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("queue_weave", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("queue_material", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("queue_usage", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("queue_features", torch.empty(0, 0), persistent=False)
        self.register_buffer("weave_class_weight", torch.empty(0), persistent=False)
        self.register_buffer("material_class_weight", torch.empty(0), persistent=False)
        self.register_buffer("usage_class_weight", torch.empty(0), persistent=False)
        self.register_buffer("feature_pos_weight", torch.empty(0), persistent=False)
        self.register_buffer("weave_class_count", torch.empty(0), persistent=False)
        self.register_buffer("material_class_count", torch.empty(0), persistent=False)
        self.register_buffer("usage_class_count", torch.empty(0), persistent=False)

    def set_class_statistics(self, stats: dict[str, torch.Tensor]) -> None:
        self.weave_class_count = stats["weave"].float()
        self.material_class_count = stats["material"].float()
        self.usage_class_count = stats["usage"].float()
        beta = float(self.classification.get("class_balance_beta", 0.0) or 0.0)
        self.weave_class_weight = _effective_number_weight(stats["weave"], beta)
        self.material_class_weight = _effective_number_weight(stats["material"], beta)
        self.usage_class_weight = _effective_number_weight(stats["usage"], beta)
        if bool(self.classification.get("feature_pos_weight", False)):
            pos = stats["features"].float()
            total = float(
                stats.get("num_feature_records", stats.get("num_records", torch.tensor(0))).item()
            )
            neg = torch.full_like(pos, total) - pos
            max_weight = float(self.classification.get("feature_pos_weight_max", 20.0))
            self.feature_pos_weight = (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=max_weight)
        else:
            self.feature_pos_weight = torch.empty(0)

    def _queue_batch(self) -> dict[str, torch.Tensor]:
        return {
            "fabric": self.queue_fabric,
            "weave": self.queue_weave,
            "material": self.queue_material,
            "usage": self.queue_usage,
            "features": self.queue_features,
        }

    def _candidate_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.queue_image_emb.shape[0] == 0:
            return {key: batch[key] for key in ("fabric", "weave", "material", "usage", "features")}
        queue = self._queue_batch()
        return {
            key: torch.cat([batch[key], queue[key].to(batch[key].device)], dim=0)
            for key in ("fabric", "weave", "material", "usage", "features")
        }

    def _contrastive_loss(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        in_batch = symmetric_label_aware_contrastive(
            output.image_emb,
            output.tactile_emb,
            batch,
            self.temperature,
            self.positive_weights,
        )
        if self.memory_queue_size <= 0 or self.queue_loss_weight <= 0 or self.queue_image_emb.shape[0] == 0:
            return in_batch
        candidate_batch = self._candidate_batch(batch)
        image_candidates = torch.cat([output.image_emb, self.queue_image_emb.to(output.image_emb.device)], dim=0)
        tactile_candidates = torch.cat(
            [output.tactile_emb, self.queue_tactile_emb.to(output.tactile_emb.device)],
            dim=0,
        )
        loss_i2t = directional_label_aware_contrastive(
            output.image_emb,
            tactile_candidates,
            batch,
            candidate_batch,
            self.temperature,
            self.positive_weights,
        )
        loss_t2i = directional_label_aware_contrastive(
            output.tactile_emb,
            image_candidates,
            batch,
            candidate_batch,
            self.temperature,
            self.positive_weights,
        )
        queue_loss = 0.5 * (loss_i2t + loss_t2i)
        return in_batch + self.queue_loss_weight * queue_loss

    def _weight_or_none(self, name: str) -> torch.Tensor | None:
        weight = getattr(self, f"{name}_class_weight")
        if weight.numel() == 0:
            return None
        return weight

    def _classification_losses(
        self,
        logits: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        smoothing = float(self.classification.get("label_smoothing", 0.0) or 0.0)
        focal_cfg = self.classification.get("focal_gamma", {})
        if not isinstance(focal_cfg, dict):
            focal_cfg = {}
        feature_pos_weight = self.feature_pos_weight if self.feature_pos_weight.numel() > 0 else None
        multiclass_loss = str(self.classification.get("multiclass_loss", "focal")).lower()
        if multiclass_loss == "balanced_softmax":
            weave_loss = balanced_softmax_cross_entropy(
                logits["weave"], batch["weave"], self.weave_class_count, smoothing
            )
            material_loss = balanced_softmax_cross_entropy(
                logits["material"], batch["material"], self.material_class_count, smoothing
            )
            usage_loss = balanced_softmax_cross_entropy(
                logits["usage"], batch["usage"], self.usage_class_count, smoothing
            )
        elif multiclass_loss == "focal":
            weave_loss = focal_cross_entropy(
                logits["weave"], batch["weave"], self._weight_or_none("weave"), smoothing,
                float(focal_cfg.get("weave", 0.0) or 0.0),
            )
            material_loss = focal_cross_entropy(
                logits["material"], batch["material"], self._weight_or_none("material"), smoothing,
                float(focal_cfg.get("material", 0.0) or 0.0),
            )
            usage_loss = focal_cross_entropy(
                logits["usage"], batch["usage"], self._weight_or_none("usage"), smoothing,
                float(focal_cfg.get("usage", 0.0) or 0.0),
            )
        else:
            raise ValueError(f"Unsupported multiclass loss: {multiclass_loss}")
        return {
            "weave": weave_loss,
            "material": material_loss,
            "usage": usage_loss,
            "features": asymmetric_binary_loss(
                logits["features"],
                batch["features"],
                feature_pos_weight,
                float(self.classification.get("feature_focal_gamma_pos", 0.0) or 0.0),
                float(self.classification.get("feature_focal_gamma_neg", 0.0) or 0.0),
            ),
        }

    def _weighted_classification_total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        attr_weights = self.classification.get("attribute_weights", {})
        if not isinstance(attr_weights, dict):
            attr_weights = {}
        return sum(
            float(attr_weights.get(name, 1.0)) * losses[name]
            for name in ("weave", "material", "usage", "features")
        )

    def _semantic_auxiliary_loss(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = output.fused.sum() * 0.0
        losses = {
            "material_components": zero,
            "material_primary": zero,
            "usage_supercategory": zero,
        }
        logits = output.semantic_logits
        if "material_components" in logits:
            valid = batch["material_semantic_valid"] > 0
            if valid.any():
                losses["material_components"] = F.binary_cross_entropy_with_logits(
                    logits["material_components"][valid].float(),
                    batch["material_components"][valid].float(),
                )
        if "material_primary" in logits:
            valid = batch["material_primary"] >= 0
            if valid.any():
                losses["material_primary"] = F.cross_entropy(
                    logits["material_primary"][valid].float(),
                    batch["material_primary"][valid],
                )
        if "usage_supercategory" in logits:
            valid = batch["usage_supercategory"] >= 0
            if valid.any():
                losses["usage_supercategory"] = F.cross_entropy(
                    logits["usage_supercategory"][valid].float(),
                    batch["usage_supercategory"][valid],
                )
        total = sum(
            float(self.semantic_aux_weights.get(name, 1.0)) * value
            for name, value in losses.items()
        )
        return total, losses

    @torch.no_grad()
    def update_queue(self, output: ModelOutput, batch: dict[str, torch.Tensor]) -> None:
        if self.memory_queue_size <= 0:
            return
        image_emb = output.image_emb.detach()
        tactile_emb = output.tactile_emb.detach()
        labels = {
            "fabric": batch["fabric"].detach(),
            "weave": batch["weave"].detach(),
            "material": batch["material"].detach(),
            "usage": batch["usage"].detach(),
            "features": batch["features"].detach(),
        }
        if self.queue_image_emb.shape[0] == 0:
            self.queue_image_emb = image_emb[-self.memory_queue_size :]
            self.queue_tactile_emb = tactile_emb[-self.memory_queue_size :]
            self.queue_fabric = labels["fabric"][-self.memory_queue_size :]
            self.queue_weave = labels["weave"][-self.memory_queue_size :]
            self.queue_material = labels["material"][-self.memory_queue_size :]
            self.queue_usage = labels["usage"][-self.memory_queue_size :]
            self.queue_features = labels["features"][-self.memory_queue_size :]
            return
        self.queue_image_emb = torch.cat([self.queue_image_emb, image_emb], dim=0)[-self.memory_queue_size :]
        self.queue_tactile_emb = torch.cat([self.queue_tactile_emb, tactile_emb], dim=0)[-self.memory_queue_size :]
        self.queue_fabric = torch.cat([self.queue_fabric, labels["fabric"]], dim=0)[-self.memory_queue_size :]
        self.queue_weave = torch.cat([self.queue_weave, labels["weave"]], dim=0)[-self.memory_queue_size :]
        self.queue_material = torch.cat([self.queue_material, labels["material"]], dim=0)[-self.memory_queue_size :]
        self.queue_usage = torch.cat([self.queue_usage, labels["usage"]], dim=0)[-self.memory_queue_size :]
        self.queue_features = torch.cat([self.queue_features, labels["features"]], dim=0)[-self.memory_queue_size :]

    def forward(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
        model: VidTouchMaterialModel,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cls = self._classification_losses(output.logits, batch)
        cls_total = self._weighted_classification_total(cls)
        zero = cls_total.new_zeros(())
        contrastive = zero
        if self.contrastive_weight > 0:
            contrastive = self._contrastive_loss(output, batch)
        unimodal_aux = cls_total.new_zeros(())
        if self.unimodal_aux_weight > 0 and output.unimodal_logits:
            attr_weights = self.classification.get("attribute_weights", {})
            aux_totals = []
            for logits in output.unimodal_logits.values():
                aux_losses = self._classification_losses(logits, batch)
                aux_totals.append(
                    sum(
                        float(attr_weights.get(name, 1.0)) * aux_losses[name]
                        for name in self.unimodal_aux_attributes
                    )
                )
            unimodal_aux = torch.stack(aux_totals).mean()
        relation = zero
        if self.relation_weight > 0:
            relation_pred = model.relation_prediction(output.logits).float()
            fused = F.normalize(output.fused.float(), dim=-1, eps=1e-6)
            relation_pred = F.normalize(relation_pred, dim=-1, eps=1e-6)
            relation = 1.0 - F.cosine_similarity(fused, relation_pred, dim=-1).mean()
        semantic_aux = zero
        semantic_losses = {
            "material_components": zero,
            "material_primary": zero,
            "usage_supercategory": zero,
        }
        if self.semantic_aux_weight > 0:
            semantic_aux, semantic_losses = self._semantic_auxiliary_loss(output, batch)
        total = (
            self.contrastive_weight * contrastive
            + self.cls_weight * cls_total
            + self.relation_weight * relation
            + self.unimodal_aux_weight * unimodal_aux
            + self.semantic_aux_weight * semantic_aux
        )
        logs = {
            "loss": total.detach(),
            "loss_contrastive": contrastive.detach(),
            "loss_cls": cls_total.detach(),
            "loss_relation": relation.detach(),
            "loss_weave": cls["weave"].detach(),
            "loss_material": cls["material"].detach(),
            "loss_usage": cls["usage"].detach(),
            "loss_features": cls["features"].detach(),
            "loss_unimodal_aux": unimodal_aux.detach(),
            "loss_semantic_aux": semantic_aux.detach(),
            "loss_material_components": semantic_losses["material_components"].detach(),
            "loss_material_primary": semantic_losses["material_primary"].detach(),
            "loss_usage_supercategory": semantic_losses["usage_supercategory"].detach(),
        }
        return total, logs
