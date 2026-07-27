from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


@dataclass
class ModelOutput:
    image_emb: torch.Tensor
    tactile_emb: torch.Tensor
    fused: torch.Tensor
    logits: dict[str, torch.Tensor]
    attr_features: dict[str, torch.Tensor] = field(default_factory=dict)
    unimodal_logits: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    semantic_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    modality_weights: torch.Tensor | None = None


def _safe_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    # FP16 cannot represent the default 1e-12 epsilon used by F.normalize.
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def _make_resnet18(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, dim


def _make_resnet50(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, dim


class DINOv2TextureBackbone(nn.Module):
    def __init__(self, repo_path: str | None, pretrained: bool) -> None:
        super().__init__()
        repo = repo_path or os.environ.get("VIDTOUCH_DINOV2_REPO")
        if repo:
            if not Path(repo).exists():
                raise FileNotFoundError(
                    f"DINOv2 repository not found at {repo}. Set "
                    "model.image_backbone_path or VIDTOUCH_DINOV2_REPO."
                )
            self.model = torch.hub.load(
                repo,
                "dinov2_vits14",
                source="local",
                pretrained=pretrained,
            )
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=pretrained,
            )
        self.embed_dim = int(self.model.embed_dim)
        self.unfrozen_block_count = 0

    def unfreeze_last_blocks(self, count: int) -> None:
        count = max(0, min(int(count), len(self.model.blocks)))
        self.model.requires_grad_(False)
        self.unfrozen_block_count = count
        if count == 0:
            return
        for block in self.model.blocks[-count:]:
            block.requires_grad_(True)
        self.model.norm.requires_grad_(True)

    def set_partial_train_mode(self, mode: bool) -> None:
        self.model.eval()
        if self.unfrozen_block_count == 0:
            return
        for block in self.model.blocks[-self.unfrozen_block_count:]:
            block.train(mode)
        self.model.norm.train(mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(image)
        cls = features["x_norm_clstoken"]
        patches = features["x_norm_patchtokens"]
        patch_mean = patches.mean(dim=1)
        patch_std = patches.std(dim=1, unbiased=False)
        return torch.cat([cls, patch_mean, patch_std], dim=-1)


def build_image_backbone(
    name: str,
    pretrained: bool,
    backbone_path: str | None = None,
) -> tuple[nn.Module, int]:
    name = name.lower()
    if name == "resnet18":
        return _make_resnet18(pretrained)
    if name == "resnet50":
        return _make_resnet50(pretrained)
    if name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, dim
    if name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        dim = model.head.in_features
        model.head = nn.Identity()
        return model, dim
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        dim = model.heads.head.in_features
        model.heads = nn.Identity()
        return model, dim
    if name == "dinov2_vits14":
        model = DINOv2TextureBackbone(backbone_path, pretrained)
        return model, model.embed_dim * 3
    raise ValueError(f"Unsupported image backbone: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViewSetPool(nn.Module):
    def __init__(self, embed_dim: int, mode: str = "mean") -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"mean", "attention"}:
            raise ValueError(f"Unsupported view set pooling mode: {mode}")
        hidden = max(32, embed_dim // 2)
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() != 3:
            raise ValueError(f"Expected B,D or B,V,D features, got shape {tuple(x.shape)}")
        if x.shape[1] == 1:
            return x[:, 0]
        if self.mode == "mean":
            return x.mean(dim=1)
        weights = self.scorer(x).softmax(dim=1)
        pooled = (weights * x).sum(dim=1)
        return self.out_norm(pooled)


class TextureStatEncoder(nn.Module):
    def __init__(self, embed_dim: int, channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        stat_dim = channels * channels + channels * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, max(embed_dim, channels * 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embed_dim, channels * 4), embed_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Covariance accumulation can overflow under AMP even when the stem output
        # itself is finite. Keep this small statistics branch in FP32.
        x = self.stem(image).float()
        b, c, h, w = x.shape
        tokens = x.reshape(b, c, h * w)
        mean = tokens.mean(dim=-1)
        std = tokens.std(dim=-1, unbiased=False)
        centered = tokens - mean.unsqueeze(-1)
        cov = centered @ centered.transpose(1, 2) / max(1, h * w - 1)
        stats = torch.cat([mean, std, cov.flatten(1)], dim=-1)
        return self.proj(stats)


class TextureAwareVisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        embed_dim: int,
        local_grid: int = 2,
        local_weight: float = 0.5,
        freeze_backbone: bool = False,
        unfreeze_last_blocks: int = 0,
        backbone_path: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = build_image_backbone(backbone_name, pretrained, backbone_path)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            if unfreeze_last_blocks > 0:
                if not isinstance(self.backbone, DINOv2TextureBackbone):
                    raise ValueError("visual_unfreeze_last_blocks is only supported for DINOv2")
                self.backbone.unfreeze_last_blocks(unfreeze_last_blocks)
        self.backbone_has_trainable_parameters = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        self.local_grid = local_grid
        self.local_weight = local_weight
        self.proj = MLP(feat_dim * 2, max(embed_dim, feat_dim), embed_dim)

    def train(self, mode: bool = True) -> "TextureAwareVisualEncoder":
        super().train(mode)
        if self.freeze_backbone:
            if isinstance(self.backbone, DINOv2TextureBackbone):
                self.backbone.set_partial_train_mode(mode)
            else:
                self.backbone.eval()
        return self

    def _local_crops(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_grid <= 1:
            return x[:, None]
        b, c, h, w = x.shape
        crops = []
        gh = h // self.local_grid
        gw = w // self.local_grid
        for yi in range(self.local_grid):
            for xi in range(self.local_grid):
                top = yi * gh
                left = xi * gw
                crop = x[:, :, top : top + gh, left : left + gw]
                crop = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
                crops.append(crop)
        return torch.stack(crops, dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone and not self.backbone_has_trainable_parameters:
            with torch.no_grad():
                global_feat = self.backbone(image)
                if self.local_grid <= 1:
                    local_feat = global_feat
                else:
                    crops = self._local_crops(image)
                    b, k, c, h, w = crops.shape
                    local_feat = self.backbone(crops.reshape(b * k, c, h, w)).reshape(b, k, -1).mean(dim=1)
        else:
            global_feat = self.backbone(image)
            if self.local_grid <= 1:
                local_feat = global_feat
            else:
                crops = self._local_crops(image)
                b, k, c, h, w = crops.shape
                local_feat = self.backbone(crops.reshape(b * k, c, h, w)).reshape(b, k, -1).mean(dim=1)
        mixed = torch.cat([global_feat, self.local_weight * local_feat], dim=-1)
        return self.proj(mixed)


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == (1, 1, 1)
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class ContactDynamicsTactileEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        base_channels: int = 32,
        use_delta: bool = True,
        use_spectral: bool = False,
        spectral_grid: int = 4,
        spectral_fusion: str = "mlp",
        spectral_gate_init: float = -4.0,
        background_subtract: bool = False,
        phase_aware: bool = False,
        phase_fusion: str = "replace",
        phase_gate_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.use_delta = use_delta
        self.use_spectral = use_spectral
        self.spectral_grid = spectral_grid
        self.spectral_fusion_mode = spectral_fusion.lower()
        self.background_subtract = bool(background_subtract)
        self.phase_aware = bool(phase_aware)
        self.phase_fusion_mode = phase_fusion.lower()
        in_ch = 6 if use_delta else 3
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, c, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(c),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            Conv3DBlock(c, c * 2, stride=(1, 2, 2)),
            Conv3DBlock(c * 2, c * 4, stride=(2, 2, 2)),
            Conv3DBlock(c * 4, c * 4, stride=(1, 2, 2)),
        )
        feat_dim = c * 4
        self.proj = MLP(feat_dim, max(embed_dim, feat_dim), embed_dim)
        if self.phase_aware:
            if self.phase_fusion_mode == "replace":
                self.phase_fusion = MLP(embed_dim * 3, embed_dim * 2, embed_dim)
            elif self.phase_fusion_mode == "gated_residual":
                self.phase_delta = MLP(embed_dim * 2, embed_dim * 2, embed_dim)
                self.phase_gate = nn.Parameter(torch.tensor(float(phase_gate_init)))
            else:
                raise ValueError(f"Unsupported tactile phase fusion: {phase_fusion}")
        if self.use_spectral:
            spectral_dim = 3 * spectral_grid * spectral_grid
            self.spectral_proj = MLP(spectral_dim, max(embed_dim, spectral_dim * 2), embed_dim)
            if self.spectral_fusion_mode == "gated_residual":
                self.spectral_delta = MLP(embed_dim, embed_dim * 2, embed_dim)
                self.spectral_gate = nn.Parameter(torch.tensor(float(spectral_gate_init)))
            elif self.spectral_fusion_mode == "mlp":
                self.spectral_fusion = MLP(embed_dim * 2, embed_dim * 2, embed_dim)
            else:
                raise ValueError(f"Unsupported spectral_fusion: {spectral_fusion}")

    def _add_delta(self, x: torch.Tensor) -> torch.Tensor:
        # x is B, C, T, H, W.
        delta = torch.zeros_like(x)
        delta[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return torch.cat([x, delta], dim=1)

    def _spectral_features(self, video: torch.Tensor) -> torch.Tensor:
        # video is B, T, C, H, W. Use pooled temporal energy as a compact tactile texture descriptor.
        b, t, c, h, w = video.shape
        gray = video.float().mean(dim=2).reshape(b * t, 1, h, w)
        pooled = F.adaptive_avg_pool2d(gray, (self.spectral_grid, self.spectral_grid))
        pooled = pooled.reshape(b, t, self.spectral_grid * self.spectral_grid)
        if t > 1:
            pooled = pooled[:, 1:] - pooled[:, :-1]
        spectrum = torch.fft.rfft(pooled, dim=1).abs()
        mean = spectrum.mean(dim=1)
        std = spectrum.std(dim=1, unbiased=False)
        peak = spectrum.amax(dim=1)
        return torch.cat([mean, std, peak], dim=-1)

    def _phase_scores(self, video: torch.Tensor, target_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        residual = video.float() - video[:, :1].float()
        contact = residual.abs().mean(dim=(2, 3, 4))
        contact_change = torch.zeros_like(contact)
        contact_change[:, 1:] = (contact[:, 1:] - contact[:, :-1]).clamp_min(0.0)
        motion = torch.zeros_like(contact)
        motion[:, 1:] = (video[:, 1:].float() - video[:, :-1].float()).abs().mean(dim=(2, 3, 4))
        contact_scale = contact / contact.amax(dim=1, keepdim=True).clamp_min(1e-6)
        press_score = contact_change + 0.05 * contact
        stroke_score = motion * (0.25 + contact_scale)

        def resize(score: torch.Tensor) -> torch.Tensor:
            resized = F.interpolate(
                score.unsqueeze(1),
                size=target_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            scale = resized.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
            return torch.softmax(resized / scale, dim=1)

        return resize(press_score), resize(stroke_score)

    def forward_with_phases(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input is B, T, C, H, W. Phase scores are computed before optional background subtraction.
        raw_video = video
        if self.background_subtract:
            video = video - video[:, :1]
        x = video.permute(0, 2, 1, 3, 4).contiguous()
        if self.use_delta:
            x = self._add_delta(x)
        x = self.stem(x)
        x = self.blocks(x)
        temporal = x.mean(dim=(3, 4)).transpose(1, 2)
        base_emb = self.proj(temporal.mean(dim=1))
        press_emb = base_emb
        stroke_emb = base_emb
        if self.phase_aware:
            press_weight, stroke_weight = self._phase_scores(raw_video, temporal.shape[1])
            press_emb = self.proj((press_weight.unsqueeze(-1) * temporal).sum(dim=1))
            stroke_emb = self.proj((stroke_weight.unsqueeze(-1) * temporal).sum(dim=1))
            if self.phase_fusion_mode == "gated_residual":
                phase_delta = self.phase_delta(torch.cat([press_emb, stroke_emb], dim=-1))
                conv_emb = base_emb + torch.sigmoid(self.phase_gate) * phase_delta
            else:
                conv_emb = self.phase_fusion(torch.cat([base_emb, press_emb, stroke_emb], dim=-1))
        else:
            conv_emb = base_emb
        if self.use_spectral:
            spectral_emb = self.spectral_proj(self._spectral_features(raw_video))
            if self.spectral_fusion_mode == "gated_residual":
                gate = torch.sigmoid(self.spectral_gate)
                conv_emb = conv_emb + gate * self.spectral_delta(spectral_emb)
            else:
                conv_emb = self.spectral_fusion(torch.cat([conv_emb, spectral_emb], dim=-1))
        return conv_emb, press_emb, stroke_emb

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.forward_with_phases(video)[0]


class R3D18TactileEncoder(nn.Module):
    """Torchvision video encoder adapted to normalized DIGIT clips."""

    def __init__(
        self,
        embed_dim: int,
        pretrained: bool = False,
        backbone_path: str | None = None,
        freeze_backbone: bool = False,
        unfreeze_last_blocks: int = 0,
        architecture: str = "r3d18",
        input_crop_size: int | None = None,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        if self.architecture == "r3d18":
            weights = (
                models.video.R3D_18_Weights.KINETICS400_V1
                if pretrained and not backbone_path
                else None
            )
            self.backbone = models.video.r3d_18(weights=weights)
        elif self.architecture == "mc3_18":
            weights = (
                models.video.MC3_18_Weights.KINETICS400_V1
                if pretrained and not backbone_path
                else None
            )
            self.backbone = models.video.mc3_18(weights=weights)
        else:
            raise ValueError(f"Unsupported torchvision video architecture: {architecture}")
        if backbone_path:
            state = torch.load(backbone_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)
        feat_dim = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Identity()
        self.proj = MLP(feat_dim, max(embed_dim, feat_dim), embed_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.input_crop_size = int(input_crop_size) if input_crop_size else None
        self.unfrozen_block_count = max(0, min(int(unfreeze_last_blocks), 4))
        self._stages = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            if self.unfrozen_block_count > 0:
                for stage in self._stages[-self.unfrozen_block_count :]:
                    stage.requires_grad_(True)

        # VidTouch caches use ImageNet normalization. Convert to the normalization
        # expected by torchvision's Kinetics-400 R3D-18 weights in the forward pass.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "kinetics_mean",
            torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "kinetics_std",
            torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "R3D18TactileEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            if self.unfrozen_block_count > 0:
                for stage in self._stages[-self.unfrozen_block_count :]:
                    stage.train(mode)
        return self

    def forward_with_phases(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = video * self.imagenet_std + self.imagenet_mean
        if self.input_crop_size is not None:
            height, width = raw.shape[-2:]
            crop = min(self.input_crop_size, height, width)
            top = max(0, (height - crop) // 2)
            left = max(0, (width - crop) // 2)
            raw = raw[..., top : top + crop, left : left + crop]
        normalized = (raw - self.kinetics_mean) / self.kinetics_std
        x = normalized.permute(0, 2, 1, 3, 4).contiguous()
        features = self.backbone(x)
        embedding = self.proj(features)
        return embedding, embedding, embedding

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.forward_with_phases(video)[0]


class VidTouchMaterialModel(nn.Module):
    def __init__(
        self,
        num_weaves: int,
        num_materials: int,
        num_usages: int,
        num_features: int,
        image_backbone: str = "resnet18",
        image_pretrained: bool = False,
        image_backbone_path: str | None = None,
        freeze_visual_backbone: bool = False,
        visual_unfreeze_last_blocks: int = 0,
        visual_local_grid: int = 2,
        visual_local_weight: float = 0.5,
        visual_texture_stats: bool = False,
        visual_texture_channels: int = 32,
        tactile_backbone: str = "contact3d",
        tactile_pretrained: bool = False,
        tactile_backbone_path: str | None = None,
        freeze_tactile_backbone: bool = False,
        tactile_unfreeze_last_blocks: int = 0,
        tactile_input_crop_size: int | None = None,
        tactile_base_channels: int = 32,
        tactile_use_delta: bool = True,
        tactile_use_spectral: bool = False,
        tactile_spectral_grid: int = 4,
        tactile_spectral_fusion: str = "mlp",
        tactile_spectral_gate_init: float = -4.0,
        tactile_background_subtract: bool = False,
        tactile_phase_aware: bool = False,
        tactile_phase_fusion: str = "replace",
        tactile_phase_gate_init: float = -3.0,
        attribute_tactile_backbone: str | None = None,
        attribute_tactile_pretrained: bool = False,
        attribute_tactile_backbone_path: str | None = None,
        freeze_attribute_tactile_backbone: bool = True,
        attribute_tactile_unfreeze_last_blocks: int = 0,
        attribute_tactile_attributes: list[str] | tuple[str, ...] | None = None,
        attribute_tactile_gate_init: float = -2.0,
        attribute_tactile_fusion: str = "feature_residual",
        embed_dim: int = 256,
        fusion_hidden_dim: int = 512,
        fusion_operator: str = "interaction",
        visual_set_pooling: str = "mean",
        tactile_set_pooling: str = "mean",
        fusion_mode: str = "multimodal",
        attribute_fusion: str = "shared",
        attribute_query_heads: int = 4,
        attribute_query_residual: bool = False,
        attribute_query_gate_init: float = -3.0,
        attribute_query_include_interaction: bool = True,
        attribute_query_normalize_memory: bool = False,
        attribute_query_attributes: list[str] | tuple[str, ...] | None = None,
        modality_dropout: float = 0.0,
        unimodal_aux_heads: bool = False,
        material_semantic_aux: bool = False,
        usage_hierarchy_aux: bool = False,
        num_material_components: int = 0,
        num_material_primaries: int = 0,
        num_usage_supercategories: int = 0,
        material_compositional_logits: bool = False,
        material_component_matrix: list[list[float]] | None = None,
        material_primary_indices: list[int] | None = None,
        material_component_gate_init: float = -2.5,
        material_primary_gate_init: float = -2.5,
    ) -> None:
        super().__init__()
        self.visual = TextureAwareVisualEncoder(
            image_backbone,
            image_pretrained,
            embed_dim,
            local_grid=visual_local_grid,
            local_weight=visual_local_weight,
            freeze_backbone=freeze_visual_backbone,
            unfreeze_last_blocks=visual_unfreeze_last_blocks,
            backbone_path=image_backbone_path,
        )
        self.visual_texture = (
            TextureStatEncoder(embed_dim, channels=visual_texture_channels)
            if visual_texture_stats
            else None
        )
        self.visual_texture_fusion = (
            MLP(embed_dim * 2, embed_dim * 2, embed_dim)
            if visual_texture_stats
            else None
        )
        tactile_backbone = tactile_backbone.lower()
        if tactile_backbone == "contact3d":
            self.tactile = ContactDynamicsTactileEncoder(
                embed_dim,
                base_channels=tactile_base_channels,
                use_delta=tactile_use_delta,
                use_spectral=tactile_use_spectral,
                spectral_grid=tactile_spectral_grid,
                spectral_fusion=tactile_spectral_fusion,
                spectral_gate_init=tactile_spectral_gate_init,
                background_subtract=tactile_background_subtract,
                phase_aware=tactile_phase_aware,
                phase_fusion=tactile_phase_fusion,
                phase_gate_init=tactile_phase_gate_init,
            )
            self.tactile_phase_aware = bool(tactile_phase_aware)
        elif tactile_backbone in {"r3d18", "mc3_18"}:
            self.tactile = R3D18TactileEncoder(
                embed_dim,
                pretrained=tactile_pretrained,
                backbone_path=tactile_backbone_path,
                freeze_backbone=freeze_tactile_backbone,
                unfreeze_last_blocks=tactile_unfreeze_last_blocks,
                architecture=tactile_backbone,
                input_crop_size=tactile_input_crop_size,
            )
            self.tactile_phase_aware = False
        else:
            raise ValueError(f"Unsupported tactile backbone: {tactile_backbone}")
        self.attribute_tactile_attributes = tuple(
            sorted(set(attribute_tactile_attributes or ("material", "usage")))
        )
        self.attribute_tactile_fusion = str(attribute_tactile_fusion).lower()
        if self.attribute_tactile_fusion not in {"feature_residual", "logit_residual"}:
            raise ValueError(
                f"Unsupported attribute tactile fusion: {attribute_tactile_fusion}"
            )
        unknown_attribute_tactile = set(self.attribute_tactile_attributes) - {
            "weave",
            "material",
            "usage",
            "features",
        }
        if unknown_attribute_tactile:
            raise ValueError(
                f"Unknown attribute tactile attributes: {sorted(unknown_attribute_tactile)}"
            )
        attribute_tactile_name = (attribute_tactile_backbone or "none").lower()
        if attribute_tactile_name == "none":
            self.attribute_tactile = None
        elif attribute_tactile_name == "r3d18":
            self.attribute_tactile = R3D18TactileEncoder(
                embed_dim,
                pretrained=attribute_tactile_pretrained,
                backbone_path=attribute_tactile_backbone_path,
                freeze_backbone=freeze_attribute_tactile_backbone,
                unfreeze_last_blocks=attribute_tactile_unfreeze_last_blocks,
            )
        else:
            raise ValueError(f"Unsupported attribute tactile backbone: {attribute_tactile_backbone}")
        self.visual_set_pool = ViewSetPool(embed_dim, visual_set_pooling)
        self.tactile_set_pool = ViewSetPool(embed_dim, tactile_set_pooling)
        self.visual_head = nn.Linear(embed_dim, embed_dim)
        self.tactile_head = nn.Linear(embed_dim, embed_dim)
        self.attribute_tactile_set_pool = (
            ViewSetPool(embed_dim, tactile_set_pooling)
            if self.attribute_tactile is not None
            else None
        )
        self.attribute_tactile_head = (
            nn.Linear(embed_dim, embed_dim)
            if self.attribute_tactile is not None
            else None
        )
        self.fusion_mode = fusion_mode.lower()
        if self.fusion_mode not in {"multimodal", "image_only", "tactile_only"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.attribute_fusion = attribute_fusion.lower()
        if self.attribute_fusion not in {"shared", "query"}:
            raise ValueError(f"Unsupported attribute_fusion: {attribute_fusion}")
        self.modality_dropout = float(modality_dropout)
        self.attr_names = ("weave", "material", "usage", "features")
        self.attribute_query_residual = bool(attribute_query_residual)
        self.attribute_query_include_interaction = bool(attribute_query_include_interaction)
        self.attribute_query_normalize_memory = bool(attribute_query_normalize_memory)
        self.attribute_query_attributes = (
            set(attribute_query_attributes)
            if attribute_query_attributes is not None
            else set(self.attr_names)
        )
        unknown_query_attrs = self.attribute_query_attributes - set(self.attr_names)
        if unknown_query_attrs:
            raise ValueError(f"Unknown attribute_query_attributes: {sorted(unknown_query_attrs)}")
        self.fusion_operator = fusion_operator.lower()
        if self.fusion_operator not in {"concat", "interaction"}:
            raise ValueError(f"Unsupported fusion operator: {fusion_operator}")
        fused_dim = embed_dim * (2 if self.fusion_operator == "concat" else 4)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                "weave": nn.Linear(embed_dim, num_weaves),
                "material": nn.Linear(embed_dim, num_materials),
                "usage": nn.Linear(embed_dim, num_usages),
                "features": nn.Linear(embed_dim, num_features),
            }
        )
        if self.attribute_tactile is not None:
            if self.attribute_tactile_fusion == "feature_residual":
                self.attribute_tactile_adapters = nn.ModuleDict(
                    {
                        name: MLP(embed_dim * 4, fusion_hidden_dim, embed_dim)
                        for name in sorted(self.attribute_tactile_attributes)
                    }
                )
            else:
                output_sizes = {
                    "weave": num_weaves,
                    "material": num_materials,
                    "usage": num_usages,
                    "features": num_features,
                }
                self.attribute_tactile_logit_heads = nn.ModuleDict(
                    {
                        name: nn.Linear(embed_dim, output_sizes[name])
                        for name in sorted(self.attribute_tactile_attributes)
                    }
                )
            self.attribute_tactile_gates = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.tensor(float(attribute_tactile_gate_init)))
                    for name in sorted(self.attribute_tactile_attributes)
                }
            )
        if self.attribute_fusion == "query":
            if embed_dim % int(attribute_query_heads) != 0:
                raise ValueError("embed_dim must be divisible by attribute_query_heads")
            self.attr_queries = nn.Parameter(torch.randn(len(self.attr_names), embed_dim) * 0.02)
            self.attr_attention = nn.MultiheadAttention(
                embed_dim,
                num_heads=int(attribute_query_heads),
                dropout=0.1,
                batch_first=True,
            )
            self.attr_query_norm = nn.LayerNorm(embed_dim)
            if self.attribute_query_normalize_memory:
                self.attr_memory_norm = nn.LayerNorm(embed_dim)
            if self.attribute_query_residual:
                self.attr_query_gate = nn.Parameter(
                    torch.full((len(self.attr_names),), float(attribute_query_gate_init))
                )
        self.unimodal_aux_heads = bool(unimodal_aux_heads)
        if self.unimodal_aux_heads:
            head_sizes = {
                "weave": num_weaves,
                "material": num_materials,
                "usage": num_usages,
                "features": num_features,
            }
            self.image_aux_heads = nn.ModuleDict(
                {name: nn.Linear(embed_dim, size) for name, size in head_sizes.items()}
            )
            self.tactile_aux_heads = nn.ModuleDict(
                {name: nn.Linear(embed_dim, size) for name, size in head_sizes.items()}
            )
        self.material_semantic_aux = bool(material_semantic_aux)
        self.usage_hierarchy_aux = bool(usage_hierarchy_aux)
        self.material_compositional_logits = bool(material_compositional_logits)
        if self.material_compositional_logits and not self.material_semantic_aux:
            raise ValueError("Compositional material logits require material_semantic_aux")
        if self.material_semantic_aux:
            if num_material_components <= 0 or num_material_primaries <= 0:
                raise ValueError("Material semantic auxiliary heads require non-empty vocabularies")
            self.material_component_head = nn.Linear(embed_dim, num_material_components)
            self.material_primary_head = nn.Linear(embed_dim, num_material_primaries)
            if self.material_compositional_logits:
                component_matrix = torch.tensor(
                    material_component_matrix or (), dtype=torch.float32
                )
                primary_indices = torch.tensor(
                    material_primary_indices or (), dtype=torch.long
                )
                if component_matrix.shape != (num_materials, num_material_components):
                    raise ValueError(
                        "Material component matrix shape does not match material vocabulary"
                    )
                if primary_indices.shape != (num_materials,):
                    raise ValueError(
                        "Material primary indices do not match material vocabulary"
                    )
                self.register_buffer(
                    "material_component_matrix", component_matrix, persistent=False
                )
                self.register_buffer(
                    "material_primary_indices", primary_indices, persistent=False
                )
                self.material_component_gate = nn.Parameter(
                    torch.tensor(float(material_component_gate_init))
                )
                self.material_primary_gate = nn.Parameter(
                    torch.tensor(float(material_primary_gate_init))
                )
        if self.usage_hierarchy_aux:
            if num_usage_supercategories <= 0:
                raise ValueError("Usage hierarchy auxiliary head requires a non-empty vocabulary")
            self.usage_supercategory_head = nn.Linear(embed_dim, num_usage_supercategories)
        self.attr_relation = nn.Sequential(
            nn.Linear(num_weaves + num_materials + num_usages + num_features, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, embed_dim),
        )

    def _encode_image_flat(self, image: torch.Tensor) -> torch.Tensor:
        visual = self.visual(image)
        if self.visual_texture is None or self.visual_texture_fusion is None:
            return visual
        texture = self.visual_texture(image)
        return self.visual_texture_fusion(torch.cat([visual, texture], dim=-1))

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            b, views, c, h, w = image.shape
            features = self._encode_image_flat(image.reshape(b * views, c, h, w)).reshape(b, views, -1)
            return self.visual_set_pool(features)
        return self._encode_image_flat(image)

    @staticmethod
    def _encode_video_branch(
        video: torch.Tensor,
        encoder: nn.Module,
        pool: ViewSetPool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if video.dim() == 6:
            b, views, t, c, h, w = video.shape
            encoded = encoder.forward_with_phases(video.reshape(b * views, t, c, h, w))
            pooled = tuple(
                pool(features.reshape(b, views, -1))
                for features in encoded
            )
            return pooled
        return encoder.forward_with_phases(video)

    def _encode_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encode_video_branch(video, self.tactile, self.tactile_set_pool)

    def forward(self, image: torch.Tensor, video: torch.Tensor) -> ModelOutput:
        hv = self._encode_image(image)
        ht, ht_press, ht_stroke = self._encode_video(video)
        zv = _safe_l2_normalize(self.visual_head(hv))
        zt = _safe_l2_normalize(self.tactile_head(ht))
        zt_press = _safe_l2_normalize(self.tactile_head(ht_press))
        zt_stroke = _safe_l2_normalize(self.tactile_head(ht_stroke))
        attribute_zt = None
        if self.attribute_tactile is not None:
            assert self.attribute_tactile_set_pool is not None
            assert self.attribute_tactile_head is not None
            attribute_ht = self._encode_video_branch(
                video,
                self.attribute_tactile,
                self.attribute_tactile_set_pool,
            )[0]
            attribute_zt = _safe_l2_normalize(self.attribute_tactile_head(attribute_ht))
        fusion_zv = zv
        fusion_zt = zt
        if self.fusion_mode == "image_only":
            fusion_zt = torch.zeros_like(zt)
        elif self.fusion_mode == "tactile_only":
            fusion_zv = torch.zeros_like(zv)
        if self.fusion_operator == "concat":
            fused_input = torch.cat([fusion_zv, fusion_zt], dim=-1)
        else:
            fused_input = torch.cat(
                [fusion_zv, fusion_zt, torch.abs(fusion_zv - fusion_zt), fusion_zv * fusion_zt],
                dim=-1,
            )
        shared_fused = self.fusion(fused_input)

        modality_weights = None
        if self.attribute_fusion == "query":
            if self.tactile_phase_aware:
                memory_parts = [zv, zt_press, zt_stroke]
                token_names = ["image", "press", "stroke"]
            else:
                memory_parts = [zv, zt]
                token_names = ["image", "tactile"]
            if self.attribute_query_include_interaction:
                memory_parts.append(shared_fused)
                token_names.append("interaction")
            memory = torch.stack(memory_parts, dim=1)
            if self.attribute_query_normalize_memory:
                memory = self.attr_memory_norm(memory)
            self.modality_token_names = tuple(token_names)
            key_padding_mask = torch.zeros(
                (memory.shape[0], memory.shape[1]),
                dtype=torch.bool,
                device=memory.device,
            )
            if self.fusion_mode == "image_only":
                for token_index, token_name in enumerate(token_names):
                    if token_name != "image":
                        key_padding_mask[:, token_index] = True
            elif self.fusion_mode == "tactile_only":
                for token_index, token_name in enumerate(token_names):
                    if token_name in {"image", "interaction"}:
                        key_padding_mask[:, token_index] = True
            elif self.training and self.modality_dropout > 0:
                choice = torch.rand(memory.shape[0], device=memory.device)
                drop_visual = choice < self.modality_dropout * 0.5
                drop_tactile = (choice >= self.modality_dropout * 0.5) & (choice < self.modality_dropout)
                for token_index, token_name in enumerate(token_names):
                    if token_name == "image":
                        key_padding_mask[drop_visual, token_index] = True
                    elif token_name == "interaction":
                        key_padding_mask[drop_visual | drop_tactile, token_index] = True
                    else:
                        key_padding_mask[drop_tactile, token_index] = True
            queries = self.attr_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
            attr_delta, modality_weights = self.attr_attention(
                queries,
                memory,
                memory,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            if self.attribute_query_residual:
                gate = torch.sigmoid(self.attr_query_gate).view(1, -1, 1)
                attr_tokens = shared_fused.unsqueeze(1) + gate * attr_delta
            else:
                attr_tokens = self.attr_query_norm(attr_delta + queries)
            attr_features = {}
            for index, name in enumerate(self.attr_names):
                attr_features[name] = (
                    attr_tokens[:, index]
                    if name in self.attribute_query_attributes
                    else shared_fused
                )
            fused = attr_tokens.mean(dim=1)
        else:
            fused = shared_fused
            attr_features = {name: fused for name in self.attr_names}

        if attribute_zt is not None and self.attribute_tactile_fusion == "feature_residual":
            for name in self.attribute_tactile_attributes:
                base = attr_features[name]
                adapter_input = torch.cat(
                    [base, attribute_zt, torch.abs(base - attribute_zt), base * attribute_zt],
                    dim=-1,
                )
                delta = self.attribute_tactile_adapters[name](adapter_input)
                gate = torch.sigmoid(self.attribute_tactile_gates[name])
                attr_features[name] = base + gate * delta

        logits = {name: self.heads[name](attr_features[name]) for name in self.attr_names}
        if attribute_zt is not None and self.attribute_tactile_fusion == "logit_residual":
            for name in self.attribute_tactile_attributes:
                gate = torch.sigmoid(self.attribute_tactile_gates[name])
                logits[name] = logits[name] + gate * self.attribute_tactile_logit_heads[name](
                    attribute_zt
                )

        semantic_logits: dict[str, torch.Tensor] = {}
        if self.material_semantic_aux:
            semantic_logits["material_components"] = self.material_component_head(
                attr_features["material"]
            )
            semantic_logits["material_primary"] = self.material_primary_head(
                attr_features["material"]
            )
        if self.usage_hierarchy_aux:
            semantic_logits["usage_supercategory"] = self.usage_supercategory_head(
                attr_features["usage"]
            )
        if self.material_compositional_logits:
            component_logits = semantic_logits["material_components"].float()
            component_matrix = self.material_component_matrix
            component_score = (
                F.logsigmoid(component_logits) @ component_matrix.t()
                + F.logsigmoid(-component_logits) @ (1.0 - component_matrix).t()
            ) / float(component_matrix.shape[1])
            primary_score = F.log_softmax(
                semantic_logits["material_primary"].float(), dim=-1
            )[:, self.material_primary_indices]
            component_score = component_score - component_score.mean(dim=-1, keepdim=True)
            primary_score = primary_score - primary_score.mean(dim=-1, keepdim=True)
            logits["material"] = (
                logits["material"].float()
                + torch.sigmoid(self.material_component_gate) * component_score
                + torch.sigmoid(self.material_primary_gate) * primary_score
            )

        unimodal_logits: dict[str, dict[str, torch.Tensor]] = {}
        if self.unimodal_aux_heads:
            unimodal_logits = {
                "image": {name: self.image_aux_heads[name](zv) for name in self.attr_names},
                "tactile": {name: self.tactile_aux_heads[name](zt) for name in self.attr_names},
            }
        return ModelOutput(
            image_emb=zv,
            tactile_emb=zt,
            fused=fused,
            logits=logits,
            attr_features=attr_features,
            unimodal_logits=unimodal_logits,
            semantic_logits=semantic_logits,
            modality_weights=modality_weights,
        )

    def relation_prediction(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            logits["weave"].softmax(dim=-1),
            logits["material"].softmax(dim=-1),
            logits["usage"].softmax(dim=-1),
            logits["features"].sigmoid(),
        ]
        return self.attr_relation(torch.cat(parts, dim=-1))
