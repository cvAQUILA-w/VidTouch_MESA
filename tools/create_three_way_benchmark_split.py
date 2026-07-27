from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from create_benchmark_split import (
    ATTRS,
    FabricRecord,
    counts,
    coverage_candidate,
    scan_records,
    split_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a train/val/test VidTouch benchmark split")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=int, default=22)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--trials", type=int, default=100000)
    return parser.parse_args()


def label_coverage(partition_counts: dict[str, dict[str, int]]) -> int:
    return sum(
        int(count > 0)
        for attr in ATTRS
        for count in partition_counts[attr].values()
    )


def effective_fabric_counts(
    records: list[FabricRecord],
    allowed: dict[str, list[str]],
) -> dict[str, int]:
    allowed_sets = {attr: set(labels) for attr, labels in allowed.items()}
    result = {}
    for attr in ATTRS:
        if attr == "features":
            result[attr] = sum(bool(set(record.features) & allowed_sets[attr]) for record in records)
        else:
            result[attr] = sum(getattr(record, attr) in allowed_sets[attr] for record in records)
    return result


def data_manifest_sha256(data_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update((data_root / "label.txt").read_bytes())
    for directory in ("RGBs", "TACs"):
        for path in sorted((data_root / directory).iterdir(), key=lambda item: item.name):
            if path.is_file():
                digest.update(f"{directory}/{path.name}\t{path.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def assignment_sha256(partitions: dict[str, list[str]]) -> str:
    payload = json.dumps(partitions, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    source_path = Path(args.source_split)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = scan_records(data_root)
    records_by_id = {record.fabric_id: record for record in records}
    known_ids = set(records_by_id)
    source_train_ids = set(source["train_ids"])
    val_ids = set(source["val_ids"])
    if source_train_ids & val_ids or source_train_ids | val_ids != known_ids:
        raise ValueError("Source split is not a complete disjoint partition of the dataset")

    label_filter = source.get("metadata", {}).get("label_filter")
    if not isinstance(label_filter, dict) or not isinstance(label_filter.get("allowed_labels"), dict):
        raise ValueError("Source split does not contain a frozen allowed-label manifest")
    allowed = {attr: list(label_filter["allowed_labels"][attr]) for attr in ATTRS}
    all_counts = counts(records, allowed)
    pool = [records_by_id[fabric_id] for fabric_id in sorted(source_train_ids)]
    rng = random.Random(args.seed)
    total_labels = sum(len(labels) for labels in allowed.values())
    best: tuple[int, float, list[FabricRecord]] | None = None

    for trial in range(args.trials):
        candidate = (
            coverage_candidate(pool, allowed, args.test_size, rng)
            if trial % 2 == 0
            else rng.sample(pool, args.test_size)
        )
        candidate_ids = {record.fabric_id for record in candidate}
        train_records = [record for record in pool if record.fabric_id not in candidate_ids]
        train_counts = counts(train_records, allowed)
        if any(train_counts[attr][label] <= 0 for attr in ATTRS for label in allowed[attr]):
            continue
        score, covered = split_score(
            all_counts,
            counts(candidate, allowed),
            len(records),
            args.test_size,
        )
        rank = (-covered, score)
        if best is None or rank < (-best[0], best[1]):
            best = (covered, score, candidate)

    if best is None or best[0] != total_labels:
        covered = best[0] if best is not None else 0
        raise RuntimeError(f"Could not cover every allowed label in test: {covered}/{total_labels}")

    test_ids = sorted(record.fabric_id for record in best[2])
    test_set = set(test_ids)
    train_ids = sorted(source_train_ids - test_set)
    val_ids_sorted = sorted(val_ids)
    partition_ids = {"train": train_ids, "val": val_ids_sorted, "test": test_ids}
    partition_records = {
        name: [records_by_id[fabric_id] for fabric_id in ids]
        for name, ids in partition_ids.items()
    }
    partition_counts = {name: counts(items, allowed) for name, items in partition_records.items()}
    partition_coverage = {
        name: label_coverage(attr_counts)
        for name, attr_counts in partition_counts.items()
    }
    if set(train_ids) & set(val_ids_sorted) or set(train_ids) & test_set or val_ids & test_set:
        raise AssertionError("Generated partitions overlap")
    if set(train_ids) | set(val_ids_sorted) | test_set != known_ids:
        raise AssertionError("Generated partitions do not cover the dataset")

    payload = {
        "train_ids": train_ids,
        "val_ids": val_ids_sorted,
        "test_ids": test_ids,
        "metadata": {
            "name": "fabric_common_v2",
            "version": 2,
            "strategy": "fixed_val_attribute_stratified_test_search",
            "seed": args.seed,
            "trials": args.trials,
            "partition_sizes": {name: len(ids) for name, ids in partition_ids.items()},
            "partition_ratios": {
                name: len(ids) / len(records) for name, ids in partition_ids.items()
            },
            "covered_labels": partition_coverage,
            "total_labels": total_labels,
            "test_selection_score": best[1],
            "label_filter": label_filter,
            "partition_label_counts": {
                name: {
                    attr: dict(sorted(attr_counts[attr].items()))
                    for attr in ATTRS
                }
                for name, attr_counts in partition_counts.items()
            },
            "effective_fabric_counts": {
                name: effective_fabric_counts(items, allowed)
                for name, items in partition_records.items()
            },
            "source_split": source.get("metadata", {}).get("name", source_path.name),
            "source_split_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "data_manifest_sha256": data_manifest_sha256(data_root),
            "assignment_sha256": assignment_sha256(partition_ids),
            "protocol": {
                "selection": "train on train_ids and select checkpoints only on val_ids",
                "test": "evaluate the selected checkpoint once on test_ids",
                "unit": "fabric_id",
                "fabric_disjoint": True,
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
