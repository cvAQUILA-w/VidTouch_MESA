# Rebuttal Expansion Results (2026-10-04)
All new runs use seeds 7/42/123, Validation-Macro checkpoint selection,
Validation-only feature-threshold calibration, 2 RGB + 2 tactile evidence,
and Fabric-ID-disjoint evaluation. The queue completed 36/36 runs with exit
code 0.

## Five-Split Matched Comparison

Values are three-seed Test means. Gaps are MESA minus matched simple.

| Split | Simple Macro | MESA Macro | Gap | Simple MKDS | MESA MKDS | Gap | Legacy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anchor | 25.74 | 28.84 | +3.09 | 24.82 | 27.58 | +2.77 | +7.70 |
| A | 24.68 | 28.19 | +3.51 | 23.78 | 26.69 | +2.91 | -2.96 |
| B | 21.40 | 24.36 | +2.96 | 18.98 | 22.31 | +3.32 | -3.99 |
| C | 17.74 | 19.19 | +1.46 | 15.61 | 17.93 | +2.32 | -0.78 |
| D | 22.52 | 24.77 | +2.24 | 20.55 | 23.75 | +3.21 | +2.27 |

Across the five preset assignments, the mean Macro gap is +2.65 (descriptive
95% t-CI [1.65, 3.66]) and the mean MKDS gap is +2.91 ([2.42, 3.40]); both are
positive on 5/5 splits. The Legacy gap is +0.45 ([-5.40, 6.29]) and is positive
on only 2/5. These intervals describe split-level variation and do not replace
the wide per-split paired Fabric bootstrap intervals.

## Nested Data Scale

Validation/Test IDs and vocabulary are fixed. The Train sets are deterministic
nested subsets: 25 subset 50 subset 75 subset 100.

| Train IDs | Simple Macro | MESA Macro | Macro gap | Simple MKDS | MESA MKDS | MKDS gap | Legacy gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 25.19 | 26.82 | +1.63 | 21.66 | 22.95 | +1.29 | -1.61 |
| 50 | 21.16 | 28.35 | +7.20 | 20.64 | 26.74 | +6.09 | +5.86 |
| 75 | 23.66 | 29.07 | +5.41 | 22.26 | 28.02 | +5.77 | +6.01 |
| 100 | 25.74 | 28.84 | +3.09 | 24.82 | 27.58 | +2.77 | +7.70 |

The expected monotonic "less data, larger MESA advantage" trend is not
supported. The advantage is largest under moderate scarcity (50/75 IDs) but
shrinks at the extreme 25-ID point. This should be reported as a non-monotonic
moderate-scarcity result.

## Alignment Decomposition

| Variant | Macro | MKDS | Legacy | Weave | Material | Usage | Features | Retrieval R@1 | Retrieval R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No alignment | 29.77 | 28.49 | 40.57 | 27.00 | 25.64 | 38.59 | 27.86 | 1.52 | 15.15 |
| Fabric-ID-only | 27.38 | 25.18 | 41.50 | 20.00 | 29.49 | 33.54 | 26.48 | 42.42 | 71.97 |
| Full semantic | 28.84 | 27.58 | 42.15 | 36.00 | 24.36 | 27.47 | 27.51 | 19.70 | 53.79 |

Cross-modal alignment is strongly supported for retrieval. Relative to
Fabric-ID-only alignment, semantic positives improve Macro by +1.46, MKDS by
+2.40, Weave by +16.00, and Features by +1.03, while reducing Material and
Usage. No alignment retains the highest Macro/MKDS. The defensible conclusion
is therefore a retrieval/attribute trade-off, not uniform recognition gain.

## Files

- `summary.json`: complete mean/SD summaries, including retrieval.
- `split_sensitivity/*paired_bootstrap.json`: split C/D paired intervals.
- `lowshot/*paired_bootstrap.json`: 25/50/75 paired intervals.
- `semantic_alignment/paired_*.json`: classification intervals for alignment.
- Every run directory contains resolved config, epoch log, Validation/Test
  metrics, and per-Fabric predictions. Checkpoints remain on the server.
