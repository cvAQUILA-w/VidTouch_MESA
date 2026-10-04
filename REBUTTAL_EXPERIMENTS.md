# Rebuttal Experiment Protocol

This file freezes the additional experiment definitions before inspecting new
Test results. The standard `fabric_common_v2` split, seeds 7/42/123,
Validation-only checkpoint/threshold selection, and best-Macro checkpoint role
remain unchanged.

## Priority 0: Strictly Matched Control

The reviewers requested a control that differs from MESA in method components,
not in encoders, evidence budget, adaptation policy, or training budget.

`configs/rebuttal/simple_matched.yaml` preserves:

- DINOv2 ViT-S/14 with only the final block adapted after the same warm-up;
- fully frozen Kinetics-pretrained R3D-18;
- two RGB and two tactile observations per encounter;
- the 100/22/22 Fabric-ID split, 60 epochs, optimizer, seeds, and checkpoint rule.

It uses mean set pooling, feature concatenation, a shared fusion MLP, ordinary
cross-entropy/BCE, and removes FPR, texture statistics, contrastive alignment,
class balancing, and relation regularization.

`configs/rebuttal/simple_matched_fpr.yaml` changes only one factor: it enables
FPR. This provides a clean first increment from the strict control. Existing
matched experiments supply the remaining comparisons, including identity-only
positives, full semantic positives, and relation regularization.

Run from the repository root:

```bash
bash scripts/run_rebuttal_matched_controls.sh \
  /root/VBTSINT_DATASET \
  /root/autodl-tmp/rebuttal_matched_controls \
  cuda
```

On the earlier 4090 server, the full 60-epoch configuration took about 21
minutes per seed. Nine sequential runs, including the FPR diagnostic below,
should therefore be budgeted at roughly 3--3.5 hours including evaluation and
overhead; the exact time depends
on GPU, cache state, and storage throughput.

## Priority 0: FPR Repeated-View Diagnostic

The submitted configuration samples with replacement. From the actual 100
Train IDs, the probability that a 2-RGB/2-tactile encounter repeats at least
one view within a modality is 55.41% (RGB 33.12%; tactile 33.33%). These are not
cross-Fabric false positives, but they reduce view diversity.

`configs/rebuttal/mesa_fpr_unique.yaml` changes only this choice and samples
without replacement. All Train fabrics contain at least three observations per
modality, so each 2+2 encounter can use four within-modality-unique observations.
The run script evaluates this diagnostic over all three seeds. Report it as a
post-review robustness analysis, not as though it were the submitted method.

## Priority 0: Fabric-Level Uncertainty

After training, export Validation/Test predictions and compute the paired
Fabric-ID bootstrap:

```bash
bash scripts/evaluate_rebuttal_matched_controls.sh \
  /root/VBTSINT_DATASET \
  /root/autodl-tmp/rebuttal_matched_controls \
  cuda
```

The evaluator calibrates the feature threshold on Validation for each seed,
freezes it for Test, and writes one prediction row per Fabric ID. The bootstrap
uses identical resampled Test IDs for both methods, averages each metric across
the three fixed seeds, and reports percentile 95% intervals for the scores and
paired score differences. It does not turn the small Test set into population
evidence; the intervals should be used to qualify, not inflate, conclusions.

The same prediction export should be run for the three retained full-MESA
best-Macro checkpoints. Then use `tools/paired_fabric_bootstrap.py` with MESA
and the strict control in the same invocation. This is the comparison that
should appear in the rebuttal.

## Priority 1: Analyses Without New Training

1. Report the exact prompt, JSON schema, retry policy, and invalid-output rate
   for each zero-shot MLLM.
2. Add per-class support and confusion summaries, especially material and
   usage collapse and rare/common performance.
3. Report FPR's repeated-view draw rate. Sampling uses replacement; a repeated
   view reduces encounter diversity but is not a false cross-Fabric positive.
4. Report trainable parameter counts, GPU model, wall-clock training time, and
   inference time from the resolved configurations and logs.

## Priority 2: Only After the Matched Control

- Three or more alternative grouped Train/Validation/Test splits, each with
  three seeds. These are sensitivity analyses and must not replace or be mixed
  into the frozen benchmark ranking.
- Sequence-length/sampling diagnostics, such as 4/8/12 uniformly sampled
  frames. These address tactile temporal content but are less central than the
  fairness and uncertainty concerns shared by three reviewers.
- Rare-label few-shot/open-set experiments. These require a carefully defined
  protocol and should not be improvised from Test labels during rebuttal.

## Post-Review Expansion Frozen on 2026-10-03

The following definitions were fixed before inspecting their Test results.
They use seeds 7/42/123, Validation-Macro checkpoint selection, Validation-only
feature-threshold calibration, and the same 2-RGB/2-tactile evidence budget.

### Five-Split Matched Comparison

Report five 100/22/22 Fabric-ID-disjoint assignments: the frozen anchor split,
the existing sensitivity splits A/B, and new splits C/D generated with search
seeds 20261002/20261003. Every assignment must cover all 144 usable Fabric IDs
exactly once and retain the benchmark vocabulary in Train/Validation/Test.
Within each split, compare MESA and `simple_matched` over all three seeds. Keep
the original anchor ranking unchanged; the five-split result is a sensitivity
analysis. Report each split and the distribution of paired split-level gaps.

Frozen assignment SHA-256 values are: anchor `c96c6c8a...a29019`, A
`cf8b42b1...59d1ea`, B `23e18901...1dc37`, C `b8a423ed...8cd89`, and D
`9f09d14a...2c8af3`. These five assignments were independently checked to be
unique, exhaustive, and Fabric-ID disjoint before training began.

### Nested Data-Scale Comparison

Use deterministic nested prefixes of 25/50/75/100 Train Fabric IDs from the
anchor split. Validation and Test IDs, vocabulary, training schedule, evidence
budget, and checkpoint rule remain fixed. Compare MESA with `simple_matched`
at every size over all three seeds. The 100-ID point reuses the anchor runs.
Report the complete curve even if the MESA-minus-simple gap is not monotonic.

### Semantic-Alignment Decomposition

Use three anchor-split variants over all three seeds:

1. no alignment (`contrastive_weight: 0`);
2. Fabric-ID-only contrastive alignment (`fabric_only` positives);
3. full semantic multi-positive alignment (submitted MESA).

This separates the value of cross-modal alignment itself from the additional
value of semantic positive structure. Compare averaged seed predictions with
paired Fabric-ID bootstrap intervals and report all aggregate/attribute scores.

For reproducibility, retain `best_macro.pt`, `best_legacy.pt`, and `last.pt`
for every MESA/component run. To fit the server disk, strict simple-control
runs retain `best_macro.pt` plus their resolved configuration, full epoch log,
Validation/Test metrics, and per-Fabric predictions.

## Reporting Rules

- Do not report a single favorable seed.
- Do not select a method, threshold, or checkpoint using Test.
- Report Macro Main, MKDS, Legacy Main, and all four attribute metrics together.
- State explicitly when a paired interval contains zero; do not call such a
  difference statistically significant.
- Keep the new-result table compact because each reviewer response has a 5000
  character limit and the KDD rebuttal interface does not allow hyperlinks.
