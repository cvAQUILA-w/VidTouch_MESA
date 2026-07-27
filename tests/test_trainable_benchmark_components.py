import torch

from vidtouch.losses import asymmetric_binary_loss
from vidtouch.model import R3D18TactileEncoder, VidTouchMaterialModel, build_image_backbone
from vidtouch.train import build_label_statistics
from vidtouch.data import FabricRecord, LabelVocab


def test_resnet50_backbone_is_available() -> None:
    backbone, dim = build_image_backbone("resnet50", pretrained=False)
    assert dim == 2048
    assert backbone.fc.__class__.__name__ == "Identity"


def test_mc3_tactile_encoder_forward() -> None:
    encoder = R3D18TactileEncoder(
        embed_dim=32,
        pretrained=False,
        architecture="mc3_18",
        input_crop_size=56,
    ).eval()
    video = torch.randn(1, 4, 3, 64, 64)
    with torch.no_grad():
        embedding = encoder(video)
    assert embedding.shape == (1, 32)


def test_plain_concat_fusion_has_expected_input_width() -> None:
    model = VidTouchMaterialModel(
        num_weaves=3,
        num_materials=4,
        num_usages=5,
        num_features=6,
        image_backbone="resnet18",
        image_pretrained=False,
        tactile_backbone="contact3d",
        tactile_base_channels=4,
        visual_local_grid=1,
        embed_dim=32,
        fusion_hidden_dim=64,
        fusion_operator="concat",
    )
    assert model.fusion[0].in_features == 64


def test_feature_loss_ignores_rows_without_retained_labels() -> None:
    logits = torch.tensor([[8.0, 8.0], [0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    loss = asymmetric_binary_loss(logits, target, None, 0.0, 0.0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[1:], target[1:])
    assert torch.allclose(loss, expected)


def test_feature_statistics_count_only_valid_rows() -> None:
    records = [
        FabricRecord("a", "plain", "cotton", "shirt", ("soft",), (), ()),
        FabricRecord("b", "plain", "cotton", "shirt", ("rare",), (), ()),
    ]
    vocab = LabelVocab(records, allowed_labels={
        "weave": ["plain"],
        "material": ["cotton"],
        "usage": ["shirt"],
        "features": ["soft"],
    })
    stats = build_label_statistics(records, vocab)
    assert stats["num_records"].item() == 2
    assert stats["num_feature_records"].item() == 1
