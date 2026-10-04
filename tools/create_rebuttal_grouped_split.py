#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from create_benchmark_split import ATTRS, counts, coverage_candidate, scan_records, split_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an additional 100/22/22 grouped sensitivity split"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=100000)
    return parser.parse_args()


def all_labels_present(partition_counts: dict[str, dict[str, int]], allowed: dict[str, list[str]]) -> bool:
    return all(partition_counts[attr][label] > 0 for attr in ATTRS for label in allowed[attr])


def assignment_sha256(partitions: dict[str, list[str]]) -> str:
    payload = json.dumps(partitions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    source_path = Path(args.source_split)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    label_filter = source["metadata"]["label_filter"]
    allowed = {attr: list(label_filter["allowed_labels"][attr]) for attr in ATTRS}
    records = scan_records(data_root)
    by_id = {record.fabric_id: record for record in records}
    rng = random.Random(args.seed)
    full_counts = counts(records, allowed)
    total_labels = sum(len(labels) for labels in allowed.values())
    best: tuple[float, dict[str, list[str]], dict[str, dict[str, dict[str, int]]]] | None = None

    for trial in range(args.trials):
        test = (
            coverage_candidate(records, allowed, 22, rng)
            if trial % 2 == 0
            else rng.sample(records, 22)
        )
        test_ids = {record.fabric_id for record in test}
        remaining = [record for record in records if record.fabric_id not in test_ids]
        val = coverage_candidate(remaining, allowed, 22, rng)
        val_ids = {record.fabric_id for record in val}
        train = [record for record in remaining if record.fabric_id not in val_ids]
        partition_records = {"train": train, "val": val, "test": test}
        partition_counts = {name: counts(items, allowed) for name, items in partition_records.items()}
        if not all(all_labels_present(partition_counts[name], allowed) for name in partition_records):
            continue
        test_score, test_covered = split_score(full_counts, partition_counts["test"], 144, 22)
        val_score, val_covered = split_score(full_counts, partition_counts["val"], 144, 22)
        if test_covered != total_labels or val_covered != total_labels:
            continue
        partitions = {
            name: sorted(record.fabric_id for record in items)
            for name, items in partition_records.items()
        }
        score = test_score + val_score
        if best is None or score < best[0]:
            best = (score, partitions, partition_counts)

    if best is None:
        raise RuntimeError("Could not find a fully covered 100/22/22 grouped split")
    _, partitions, partition_counts = best
    if {len(partitions[name]) for name in partitions} != {22, 100}:
        raise AssertionError({name: len(ids) for name, ids in partitions.items()})
    if len(set().union(*(set(ids) for ids in partitions.values()))) != len(by_id):
        raise AssertionError("Partitions do not cover all usable Fabric IDs")

    payload = {
        "train_ids": partitions["train"],
        "val_ids": partitions["val"],
        "test_ids": partitions["test"],
        "metadata": {
            "name": f"fabric_common_v2_sensitivity_{args.seed}",
            "version": 1,
            "strategy": "attribute_covered_grouped_sensitivity_split",
            "seed": args.seed,
            "trials": args.trials,
            "partition_sizes": {name: len(ids) for name, ids in partitions.items()},
            "assignment_sha256": assignment_sha256(partitions),
            "source_split_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "label_filter": label_filter,
            "partition_label_counts": {
                name: {
                    attr: dict(sorted(partition_counts[name][attr].items()))
                    for attr in ATTRS
                }
                for name in partitions
            },
            "protocol": {
                "unit": "fabric_id",
                "fabric_disjoint": True,
                "role": "split-sensitivity analysis only; not the benchmark ranking",
                "selection": "select checkpoints and thresholds on this split's Validation only",
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
