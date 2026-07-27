# Audited MESA Results

The values below are percentages over seeds 42, 7, and 123 on the frozen
100/22/22 Fabric-ID split. Checkpoints and the multi-label feature threshold are
selected using Validation only. Values are mean +/- sample standard deviation.

## Full Method on Test

| Metric | Result |
|---|---:|
| MKDS | 28.25 |
| Macro Main | 28.84 +/- 3.70 |
| Legacy Main | 42.15 +/- 4.44 |
| Weave balanced accuracy | 36.00 +/- 7.57 |
| Material balanced accuracy | 24.36 +/- 5.88 |
| Usage balanced accuracy | 27.47 +/- 3.50 |
| Features macro-F1 | 27.51 +/- 8.23 |
| Weave accuracy | 49.02 +/- 8.99 |
| Material accuracy | 29.17 +/- 7.22 |
| Usage accuracy | 47.06 +/- 0.00 |
| Features micro-F1 | 43.37 +/- 4.26 |
| Mean bidirectional R@1 | 19.70 +/- 3.47 |
| Mean bidirectional R@5 | 53.79 +/- 9.19 |

MKDS is computed as the harmonic mean of the four reported mean equal-class
metrics. Macro Main is their arithmetic mean. Legacy Main uses ordinary
accuracy for weave/material/usage and micro-F1 for features.

## Matched Validation Ablation

All rows use the same split, seeds, budget, and Validation Macro checkpoint
selection.

| Variant | Macro Main | Delta | Legacy Main |
|---|---:|---:|---:|
| Full MESA | 30.94 +/- 2.91 | 0.00 | 40.79 +/- 3.66 |
| w/o FPR | 26.50 +/- 1.25 | -4.45 | 35.45 +/- 2.94 |
| Fabric-ID positives only | 29.00 +/- 2.19 | -1.95 | 39.58 +/- 1.46 |
| w/o texture statistics | 28.41 +/- 2.72 | -2.53 | 39.03 +/- 1.71 |
| w/o class balancing | 27.15 +/- 3.30 | -3.79 | 37.54 +/- 1.37 |
| w/o contrastive alignment | 25.91 +/- 3.33 | -5.04 | 36.52 +/- 3.78 |
| Image-only fusion | 27.04 +/- 1.46 | -3.90 | 34.33 +/- 3.64 |
| Tactile-only fusion | 22.77 +/- 0.95 | -8.17 | 33.96 +/- 2.54 |

These ablations are reported on Validation because the method and components
were selected there. Test is reserved for the frozen full-method estimate.
