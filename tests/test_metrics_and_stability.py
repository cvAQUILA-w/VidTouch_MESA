from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from vidtouch.data import USAGE_SUPERCATEGORY, split_material_components
from vidtouch.losses import (
    VidTouchCriterion,
    balanced_softmax_cross_entropy,
    symmetric_label_aware_contrastive,
)
from vidtouch.metrics import (
    best_global_multilabel_threshold,
    material_knowledge_discovery_score,
    mean_legacy_score,
    mean_main_score,
    multilabel_f1,
)
from vidtouch.model import ModelOutput, VidTouchMaterialModel, _safe_l2_normalize
from vidtouch.train import save_checkpoint, update_checkpoint_alias


class MetricProtocolTest(unittest.TestCase):
    def test_main_score_uses_long_tail_metrics(self) -> None:
        metrics = {
            "weave_acc": 0.9,
            "material_acc": 0.8,
            "usage_acc": 0.7,
            "feature_f1_micro": 0.6,
            "weave_balanced_acc": 0.4,
            "material_balanced_acc": 0.3,
            "usage_balanced_acc": 0.2,
            "feature_f1_macro_calibrated": 0.1,
        }
        self.assertAlmostEqual(mean_main_score(metrics), 0.25)
        self.assertAlmostEqual(mean_legacy_score(metrics), 0.75)
        expected_mkds = 4.0 / (1.0 / 0.4 + 1.0 / 0.3 + 1.0 / 0.2 + 1.0 / 0.1)
        self.assertAlmostEqual(material_knowledge_discovery_score(metrics), expected_mkds)

    def test_mkds_returns_zero_when_an_axis_is_missing_or_zero(self) -> None:
        complete = {
            "weave_balanced_acc": 0.4,
            "material_balanced_acc": 0.3,
            "usage_balanced_acc": 0.2,
            "feature_f1_macro": 0.1,
        }
        self.assertGreater(material_knowledge_discovery_score(complete), 0.0)
        complete["usage_balanced_acc"] = 0.0
        self.assertEqual(material_knowledge_discovery_score(complete), 0.0)
        complete.pop("feature_f1_macro")
        self.assertEqual(material_knowledge_discovery_score(complete), 0.0)

    def test_threshold_selection_is_macro_first(self) -> None:
        logits = torch.tensor(
            [[4.0, -0.4], [4.0, -0.4], [4.0, 0.4], [-4.0, 0.4]]
        )
        target = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        )
        thresholds = [0.4, 0.6]
        expected = max(
            ((t, multilabel_f1(logits, target, t)) for t in thresholds),
            key=lambda item: (item[1]["feature_f1_macro"], item[1]["feature_f1_micro"]),
        )
        actual = best_global_multilabel_threshold(logits, target, thresholds)
        self.assertEqual(actual[0], expected[0])


class StabilityTest(unittest.TestCase):
    def test_checkpoint_tracks_both_selection_scores_and_alias(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        vocab = Mock()
        vocab.state_dict.return_value = {"features": ["soft"]}
        with TemporaryDirectory() as tmp:
            macro_path = Path(tmp) / "best_macro.pt"
            alias_path = Path(tmp) / "best.pt"
            save_checkpoint(
                macro_path,
                model,
                optimizer,
                scheduler,
                epoch=3,
                best_macro_score=0.28,
                best_legacy_score=0.40,
                cfg={"seed": 42},
                vocab=vocab,
                checkpoint_role="best_macro",
            )
            update_checkpoint_alias(macro_path, alias_path)
            checkpoint = torch.load(alias_path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["best_score"], 0.28)
        self.assertEqual(checkpoint["best_macro_score"], 0.28)
        self.assertEqual(checkpoint["best_legacy_score"], 0.40)
        self.assertEqual(checkpoint["checkpoint_role"], "best_macro")

    def test_material_decomposition_preserves_primary_order(self) -> None:
        self.assertEqual(
            split_material_components("cottonpolyesterspandex"),
            ("cotton", "polyester", "spandex"),
        )
        self.assertEqual(
            split_material_components("polyestercotton"),
            ("polyester", "cotton"),
        )
        self.assertEqual(USAGE_SUPERCATEGORY["shirt"], "upper-body")
        self.assertEqual(USAGE_SUPERCATEGORY["coat"], "outerwear")

    def test_half_precision_zero_vectors_normalize_finitely(self) -> None:
        normalized = _safe_l2_normalize(torch.zeros(4, 8, dtype=torch.float16))
        self.assertTrue(torch.isfinite(normalized).all())

    def test_contrastive_loss_upcasts_sensitive_math(self) -> None:
        image = torch.zeros(3, 8, dtype=torch.float16)
        tactile = torch.zeros(3, 8, dtype=torch.float16)
        batch = {
            "fabric": torch.tensor([0, 1, 2]),
            "weave": torch.tensor([0, 0, 1]),
            "material": torch.tensor([0, 1, 1]),
            "usage": torch.tensor([0, 1, 2]),
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        }
        loss = symmetric_label_aware_contrastive(
            image,
            tactile,
            batch,
            temperature=0.07,
            positive_weights={"fabric": 1.0, "weave": 0.25},
        )
        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(loss.dtype, torch.float32)

    def test_zero_weight_objectives_are_not_evaluated(self) -> None:
        criterion = VidTouchCriterion(
            temperature=0.07,
            contrastive_weight=0.0,
            cls_weight=1.0,
            relation_weight=0.0,
            positive_weights={"fabric": 1.0},
            semantic_aux_weight=0.0,
        )
        output = ModelOutput(
            image_emb=torch.full((2, 4), float("nan")),
            tactile_emb=torch.full((2, 4), float("nan")),
            fused=torch.zeros(2, 4),
            logits={
                "weave": torch.zeros(2, 2),
                "material": torch.zeros(2, 2),
                "usage": torch.zeros(2, 2),
                "features": torch.zeros(2, 2),
            },
        )
        batch = {
            "weave": torch.tensor([0, 1]),
            "material": torch.tensor([0, 1]),
            "usage": torch.tensor([0, 1]),
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        model = Mock()
        model.relation_prediction.side_effect = AssertionError("disabled relation loss was evaluated")
        with patch.object(
            criterion,
            "_contrastive_loss",
            side_effect=AssertionError("disabled contrastive loss was evaluated"),
        ), patch.object(
            criterion,
            "_semantic_auxiliary_loss",
            side_effect=AssertionError("disabled semantic loss was evaluated"),
        ):
            total, logs = criterion(output, batch, model)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(float(logs["loss_contrastive"]), 0.0)
        self.assertEqual(float(logs["loss_relation"]), 0.0)
        self.assertEqual(float(logs["loss_semantic_aux"]), 0.0)

    def test_balanced_softmax_uses_train_class_prior(self) -> None:
        logits = torch.zeros(2, 2)
        target = torch.tensor([0, 1])
        count = torch.tensor([9.0, 1.0])
        loss = balanced_softmax_cross_entropy(logits, target, count)
        expected = torch.nn.functional.cross_entropy(
            torch.log(count).expand(2, -1), target
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), float(expected))

    def test_attribute_tactile_logit_residual_has_attribute_only_heads(self) -> None:
        model = VidTouchMaterialModel(
            num_weaves=3,
            num_materials=4,
            num_usages=5,
            num_features=6,
            image_backbone="resnet18",
            image_pretrained=False,
            attribute_tactile_backbone="r3d18",
            attribute_tactile_pretrained=False,
            freeze_attribute_tactile_backbone=True,
            attribute_tactile_attributes=("material", "usage"),
            attribute_tactile_fusion="logit_residual",
            embed_dim=32,
            fusion_hidden_dim=64,
        )
        self.assertEqual(set(model.attribute_tactile_logit_heads), {"material", "usage"})
        self.assertFalse(hasattr(model, "attribute_tactile_adapters"))
        self.assertEqual(model.attribute_tactile_logit_heads["material"].out_features, 4)
        self.assertEqual(model.attribute_tactile_logit_heads["usage"].out_features, 5)

    def test_semantic_auxiliary_heads_have_expected_output_sizes(self) -> None:
        model = VidTouchMaterialModel(
            num_weaves=3,
            num_materials=4,
            num_usages=5,
            num_features=6,
            image_backbone="resnet18",
            image_pretrained=False,
            material_semantic_aux=True,
            usage_hierarchy_aux=True,
            num_material_components=8,
            num_material_primaries=6,
            num_usage_supercategories=5,
            embed_dim=32,
            fusion_hidden_dim=64,
        )
        self.assertEqual(model.material_component_head.out_features, 8)
        self.assertEqual(model.material_primary_head.out_features, 6)
        self.assertEqual(model.usage_supercategory_head.out_features, 5)

    def test_compositional_material_classifier_validates_layout(self) -> None:
        model = VidTouchMaterialModel(
            num_weaves=3,
            num_materials=4,
            num_usages=5,
            num_features=6,
            image_backbone="resnet18",
            image_pretrained=False,
            material_semantic_aux=True,
            material_compositional_logits=True,
            num_material_components=3,
            num_material_primaries=2,
            material_component_matrix=[
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            material_primary_indices=[0, 0, 1, 1],
            embed_dim=32,
            fusion_hidden_dim=64,
        )
        self.assertEqual(tuple(model.material_component_matrix.shape), (4, 3))
        self.assertEqual(tuple(model.material_primary_indices.shape), (4,))
        gate = torch.sigmoid(model.material_component_gate).detach().item()
        self.assertAlmostEqual(gate, 0.075858, places=5)


if __name__ == "__main__":
    unittest.main()
