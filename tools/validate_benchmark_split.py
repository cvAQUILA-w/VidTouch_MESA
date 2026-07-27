from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from create_benchmark_split import ATTRS, counts, scan_records
from create_three_way_benchmark_split import assignment_sha256, data_manifest_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a frozen VidTouch benchmark split")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_path = Path(args.split)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = scan_records(data_root)
    records_by_id = {record.fabric_id: record for record in records}
    partition_ids = {
        name: list(split[f"{name}_ids"])
        for name in ("train", "val", "test")
    }
    partition_sets = {name: set(ids) for name, ids in partition_ids.items()}
    union = set().union(*partition_sets.values())
    if sum(len(ids) for ids in partition_sets.values()) != len(union):
        raise AssertionError("Split partitions overlap")
    if union != set(records_by_id):
        raise AssertionError("Split IDs do not match the complete dataset")

    metadata = split["metadata"]
    allowed = metadata["label_filter"]["allowed_labels"]
    total_labels = sum(len(labels) for labels in allowed.values())
    coverage = {}
    for name, ids in partition_ids.items():
        attr_counts = counts([records_by_id[fabric_id] for fabric_id in ids], allowed)
        coverage[name] = sum(
            int(attr_counts[attr][label] > 0)
            for attr in ATTRS
            for label in allowed[attr]
        )
    if any(value != total_labels for value in coverage.values()):
        raise AssertionError(f"A partition does not cover the frozen label vocabulary: {coverage}")

    assignment_hash = assignment_sha256(partition_ids)
    if assignment_hash != metadata["assignment_sha256"]:
        raise AssertionError("Assignment SHA256 does not match split metadata")
    manifest_hash = data_manifest_sha256(data_root)
    if manifest_hash != metadata["data_manifest_sha256"]:
        raise AssertionError("Dataset manifest SHA256 does not match split metadata")

    result = {
        "valid": True,
        "partition_sizes": {name: len(ids) for name, ids in partition_ids.items()},
        "covered_labels": coverage,
        "total_labels": total_labels,
        "assignment_sha256": assignment_hash,
        "data_manifest_sha256": manifest_hash,
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
