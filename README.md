# VidTouch MESA

Official PyTorch implementation of **Material Evidence Semantic Alignment
(MESA)** for specimen-level RGB and tactile-video material understanding on
VidTouch.

MESA is tailored to the structure of VidTouch:

- **Fabric-aware Pair Recombination (FPR)** samples valid RGB/tactile
  combinations from observations sharing a Fabric ID.
- **Texture and contact-dynamics encoding** combines a partially trainable
  DINOv2 ViT-S/14 branch, explicit texture statistics, and a frozen
  Kinetics-pretrained R3D-18 tactile backbone.
- **Semantic alignment and attribute-aware learning** uses Fabric identity and
  structured label overlap as cross-modal supervision, then predicts weave,
  material, usage, and multi-label features with task-specific masks.
- **Long-tail-aware supervision** uses effective-number weighting for
  single-label heads and asymmetric focusing for the feature head.

The repository contains the frozen 100/22/22 Fabric-ID split, low-shot subsets,
training/evaluation code, paper-aligned configurations, and matched ablations.
Dataset media and pretrained checkpoints are not stored in Git.

Audited three-seed numbers corresponding to this release are recorded in
[RESULTS.md](RESULTS.md).

## Repository Layout

```text
configs/                 MESA, low-shot, and single-factor ablation configs
scripts/                 Portable three-seed training and evaluation helpers
splits/                  Frozen Fabric-ID partitions and allowed vocabularies
tests/                   Metric, stability, and configuration tests
tools/                   Split validation and result summarization
vidtouch/                Dataset, model, loss, training, and evaluation code
```

## Installation

Python 3.10 or newer is recommended. Install a CUDA-compatible PyTorch build
for your system first, then install this package:

```bash
git clone https://github.com/cvAQUILA-w/VidTouch_MESA.git
cd VidTouch_MESA

# Follow https://pytorch.org/get-started/locally/ for the CUDA wheel you need.
pip install -e ".[dev]"
```

The default configuration obtains DINOv2 through Torch Hub. For an offline
machine, clone DINOv2 on a machine with network access, transfer it to the
server, and set:

```bash
export VIDTOUCH_DINOV2_REPO=/path/to/dinov2
```

TorchVision similarly downloads the Kinetics-400 R3D-18 weights on first use.
For offline use, set `model.tactile_backbone_path` in a local config to the
downloaded `r3d_18-b3b3357e.pth` file.

## Data

Expected layout:

```text
VidTouch/
  RGBs/
    <FabricID>*.jpg
  TACs/
    <FabricID>*.mp4
  label.txt
```

Each non-comment `label.txt` line is whitespace-delimited:

```text
FabricID weave material usage feature_1 feature_2 ...
```

The complete release is projected to a frozen common-label benchmark without
globally deleting rare-label fabrics. A target outside a head's retained
vocabulary is masked for that head only. See [DATASET.md](DATASET.md) and
[splits/README.md](splits/README.md).

Validate a dataset copy before training:

```bash
python tools/validate_benchmark_split.py \
  --data-root /path/to/VidTouch \
  --split splits/fabric_common_v2.json
```

## Cache Tactile Videos

Caching the uniformly sampled tactile tensors substantially reduces repeated
video-decoding overhead:

```bash
python -m vidtouch.cache \
  --data-root /path/to/VidTouch \
  --frames 12 \
  --size 160 \
  --dtype float16
```

The default `auto_float16` cache is stored under
`<data-root>/.cache`.

## Train MESA

One seed:

```bash
python -m vidtouch.train \
  --config configs/mesa.yaml \
  --seed 42 \
  --data-root /path/to/VidTouch \
  --output-dir runs/mesa_seed42
```

All three paper seeds:

```bash
bash scripts/train_three_seeds.sh /path/to/VidTouch runs/mesa
```

Each run retains:

- `best_macro.pt`: selected by Validation Macro Main;
- `best_legacy.pt`: selected by Validation Legacy Main;
- `best.pt`: compatibility alias of `best_macro.pt`;
- `last.pt`: final epoch;
- resolved configuration, split, vocabulary, and per-epoch metrics.

Training seeds change optimization only. They never regenerate the split or
label vocabulary.

## Validation and Test

Feature-threshold calibration and checkpoint selection must use Validation.
Test evaluation requires the frozen Validation threshold explicitly:

```bash
bash scripts/evaluate_checkpoint.sh \
  runs/mesa_seed42/best_macro.pt \
  /path/to/VidTouch \
  runs/mesa_seed42/evaluation
```

The evaluator reports:

- balanced accuracy for weave, material, and usage;
- macro-F1 for multi-label features;
- **Macro Main**, the arithmetic mean of those four equal-class axes;
- **MKDS**, their harmonic mean;
- ordinary accuracy/micro-F1 and **Legacy Main**;
- bidirectional image-to-touch and touch-to-image Recall@1/5.

## Ablations and Low-Shot Training

Run the seven matched single-factor ablations over seeds 42, 7, and 123:

```bash
bash scripts/run_ablations.sh /path/to/VidTouch runs/ablations
```

The available variants remove FPR, semantic positives, texture statistics,
class balancing, contrastive alignment, or one inference modality. Low-shot
training uses deterministic nested subsets while preserving the same Validation
and Test IDs:

```bash
python -m vidtouch.train \
  --config configs/lowshot50.yaml \
  --seed 42 \
  --data-root /path/to/VidTouch \
  --output-dir runs/lowshot50_seed42
```

## Reproducibility Notes

- The benchmark split is specimen-disjoint; no Fabric ID crosses partitions.
- Feature-positive validity masks are applied consistently in training and
  evaluation.
- cuDNN/CUBLAS deterministic settings are enabled where PyTorch supports them.
- Normalization and loss calculations use FP32 under AMP-sensitive paths.
- Test must not be used to select an epoch, threshold, seed, ablation, or
  ensemble.

## Citation

The paper is under review. Please use [CITATION.cff](CITATION.cff); publication
metadata will be updated after acceptance.

## License

The code and split manifests are released under the [MIT License](LICENSE).
VidTouch media, annotations, pretrained backbones, and third-party code may
have separate terms and are not redistributed by this repository.
