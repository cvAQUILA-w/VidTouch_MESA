# Frozen Benchmark Splits

`fabric_common_v2.json` is the source of truth for the paper's three-way,
Fabric-ID-disjoint benchmark:

- Train: 100 Fabric IDs
- Validation: 22 Fabric IDs
- Test: 22 Fabric IDs
- Allowed labels: 10 weave, 13 material, 11 usage, 14 features
- Assignment SHA256:
  `c96c6c8af21b52e6a180361d0143f292baddc4892913848ecfa257beb4a29019`
- Split-file SHA256:
  `4a466d92bfaf65812b835eb3f4c28787df4848eb4f0d7ba331a1f9a7267fe45d`

Protocol:

1. Train only on `train_ids`.
2. Tune hyperparameters, calibrate the feature threshold, and select
   checkpoints only on `val_ids`.
3. Evaluate the selected checkpoint once on `test_ids`.
4. Never regenerate a partition from a training seed.
5. Report all four attribute metrics and Fabric-level retrieval.

The 25- and 50-fabric manifests are fixed nested subsets of the 100 training
IDs. Their Validation and Test IDs are identical to the full split. Remaining
training IDs are recorded as `unused_ids` and are never exposed to training or
evaluation for that low-shot run.
