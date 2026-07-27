from pathlib import Path

from vidtouch.utils import load_yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "configs"


def test_main_config_matches_frozen_mesa_protocol() -> None:
    config = load_yaml(CONFIG_ROOT / "mesa.yaml")
    assert config["split"]["file"] == "splits/fabric_common_v2.json"
    assert config["data"]["pair_recombination"] is True
    assert config["model"]["image_backbone"] == "dinov2_vits14"
    assert config["model"]["visual_unfreeze_last_blocks"] == 1
    assert config["model"]["tactile_backbone"] == "r3d18"
    assert config["model"]["tactile_unfreeze_last_blocks"] == 0
    assert config["loss"]["contrastive_weight"] == 1.5
    assert config["loss"]["relation_weight"] == 0.01


def test_ablation_configs_are_single_factor_overrides() -> None:
    full = load_yaml(CONFIG_ROOT / "mesa.yaml")
    expected = {
        "no_fpr": lambda cfg: cfg["data"]["pair_recombination"] is False,
        "fabric_only": lambda cfg: sum(
            cfg["loss"]["positive_weights"][key]
            for key in ("weave", "material", "usage", "feature")
        )
        == 0.0,
        "no_texture": lambda cfg: cfg["model"]["visual_texture_stats"] is False,
        "no_balance": lambda cfg: (
            cfg["loss"]["classification"]["class_balance_beta"] == 0.0
            and cfg["loss"]["classification"]["feature_pos_weight"] is False
        ),
        "no_alignment": lambda cfg: cfg["loss"]["contrastive_weight"] == 0.0,
        "image_only": lambda cfg: cfg["model"]["fusion_mode"] == "image_only",
        "tactile_only": lambda cfg: cfg["model"]["fusion_mode"] == "tactile_only",
    }
    for name, check in expected.items():
        config = load_yaml(CONFIG_ROOT / "ablation" / f"{name}.yaml")
        assert config["split"] == full["split"]
        assert config["model"]["tactile_backbone"] == "r3d18"
        assert config["loss"]["relation_weight"] == 0.01
        assert check(config)


def test_lowshot_configs_preserve_validation_and_test_protocol() -> None:
    for size in (25, 50):
        config = load_yaml(CONFIG_ROOT / f"lowshot{size}.yaml")
        assert config["split"]["file"] == f"splits/fabric_common_v2_lowshot{size}.json"
        assert config["split"]["eval_partition"] == "val"
