# Post-Review Evidence Artifact

This directory contains the compact, checkpoint-free evidence package for the
post-review analyses. All trainable comparisons use seeds 7, 42, and 123,
Validation-Macro checkpoint selection, Validation-only feature-threshold
calibration, two RGB plus two tactile observations, and Fabric-ID-disjoint
evaluation unless a subdirectory explicitly states otherwise.

## Headline Results

| Comparison | Macro Main | MKDS | Legacy Main |
|---|---:|---:|---:|
| Five-split MESA minus matched-simple mean gap | +2.65 | +2.91 | +0.45 |
| Split-level descriptive 95% t-CI | [1.65, 3.66] | [2.42, 3.40] | [-5.40, 6.29] |
| Positive split count | 5/5 | 5/5 | 2/5 |

The full five-split, nested 25/50/75/100-ID, and semantic-alignment tables are
in [RESULTS.md](RESULTS.md). Paired intervals use 10,000 Fabric-ID bootstrap
replicates and are stored beside the corresponding predictions.

## Directory Map

- `matched_controls/`: submitted MESA, capacity-matched simple controls, the
  simple-plus-FPR increment, and the without-replacement FPR diagnostic.
- `split_sensitivity/`: five fixed 100/22/22 assignments and paired results.
- `lowshot/`: deterministic nested 25/50/75-ID runs. The 100-ID point is the
  anchor comparison in `matched_controls/`.
- `semantic_alignment/`: no-alignment, Fabric-ID-only, and full semantic
  alignment comparisons.
- `frame_ablation/`: matched 6/12/24-frame results and paired intervals.
- `diagnostics/`: efficiency, recurrent errors, frozen-fusion comparison,
  relation ablation, and grouped-split summaries.
- `mllm/`: model-wise metrics, raw prediction records, invalid-output counts,
  and material-class coverage. Executable prompt/parser code is under
  `../../protocols/mllm/`.

Each run directory retains its resolved configuration, split, vocabulary,
epoch metrics, Validation/Test metrics, and one prediction row per scored
Fabric ID. Model checkpoints and dataset media are intentionally excluded.

## Integrity and Scope

`SHA256SUMS` covers every file in this directory except itself. Paths use `/`
separators and hashes are computed over exact file bytes.

The grouped splits assess sensitivity within the same 144-fabric collection.
They do not establish cross-sensor, cross-operator, non-fabric, or open-world
generalization. Bootstrap intervals quantify uncertainty over the scored Test
Fabric IDs and do not turn this collection into a population sample.
