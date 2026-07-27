from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


TASKS = ("weave", "material", "usage", "features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 50])
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.strip().split()
        if len(fields) < 4 or fields[0].startswith("#"):
            continue
        fabric_id, weave, material, usage, *features = fields
        records[fabric_id] = {
            "weave": weave,
            "material": material,
            "usage": usage,
            "features": features,
        }
    return records


def stable_tie_break(fabric_id: str) -> int:
    return int(hashlib.sha256(f"vidtouch-lowshot-v1:{fabric_id}".encode()).hexdigest(), 16)


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    train_ids = list(split["train_ids"])
    allowed = split["metadata"]["label_filter"]["allowed_labels"]
    dimensions = [f"{task}:{label}" for task in TASKS for label in allowed[task]]
    dim_index = {name: index for index, name in enumerate(dimensions)}

    vectors: dict[str, list[int]] = {}
    for fabric_id in train_ids:
        vector = [0] * len(dimensions)
        record = labels[fabric_id]
        for task in TASKS[:3]:
            value = str(record[task])
            key = f"{task}:{value}"
            if key in dim_index:
                vector[dim_index[key]] = 1
        for value in record["features"]:
            key = f"features:{value}"
            if key in dim_index:
                vector[dim_index[key]] = 1
        vectors[fabric_id] = vector

    totals = [sum(v[index] for v in vectors.values()) for index in range(len(dimensions))]
    selected: list[str] = []
    counts = [0] * len(dimensions)
    remaining = set(train_ids)
    while remaining:
        next_size = len(selected) + 1
        target_fraction = next_size / len(train_ids)

        def score(fabric_id: str) -> tuple[float, int]:
            vector = vectors[fabric_id]
            deficit = 0.0
            coverage = 0.0
            for index, present in enumerate(vector):
                if not present or totals[index] == 0:
                    continue
                normalized_deficit = target_fraction - counts[index] / totals[index]
                deficit += max(0.0, normalized_deficit)
                if counts[index] == 0:
                    coverage += 1.0 / totals[index]
            return deficit + 0.25 * coverage, -stable_tie_break(fabric_id)

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        counts = [count + value for count, value in zip(counts, vectors[chosen], strict=True)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for size in sorted(set(args.sizes)):
        if not 0 < size < len(train_ids):
            raise ValueError(f"Invalid low-shot size: {size}")
        subset = selected[:size]
        payload = copy.deepcopy(split)
        payload["train_ids"] = subset
        # Fixed-split validation requires every known fabric to be assigned. Keep
        # the held-out training fabrics explicit without exposing them to the
        # optimizer or to validation/test evaluation.
        payload["unused_ids"] = [fabric_id for fabric_id in train_ids if fabric_id not in subset]
        subset_counts = {
            dimensions[index]: sum(vectors[fabric_id][index] for fabric_id in subset)
            for index in range(len(dimensions))
        }
        payload["metadata"]["lowshot"] = {
            "protocol": "deterministic_nested_iterative_stratification_v1",
            "source_split": str(args.split),
            "source_train_size": len(train_ids),
            "train_size": size,
            "unused_train_size": len(train_ids) - size,
            "prefix_of_deterministic_order": True,
            "label_counts": subset_counts,
        }
        path = args.output_dir / f"fabric_common_v2_lowshot{size}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        covered = sum(count > 0 for count in subset_counts.values())
        print(f"{path}: {size} fabrics, {covered}/{len(dimensions)} retained labels covered")


if __name__ == "__main__":
    main()
