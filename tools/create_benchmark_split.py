from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ATTRS = ("weave", "material", "usage", "features")


@dataclass(frozen=True)
class FabricRecord:
    fabric_id: str
    weave: str
    material: str
    usage: str
    features: tuple[str, ...]


def scan_records(data_root: str | Path) -> list[FabricRecord]:
    root = Path(data_root)
    image_ids = {path.stem.split()[0] for path in (root / "RGBs").iterdir() if path.is_file()}
    video_ids = {path.stem.split()[0] for path in (root / "TACs").iterdir() if path.is_file()}
    records = []
    with open(root / "label.txt", "r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 4 or parts[0].startswith("#"):
                continue
            fabric_id, weave, material, usage, *features = parts
            if fabric_id not in image_ids or fabric_id not in video_ids:
                continue
            records.append(FabricRecord(fabric_id, weave, material, usage, tuple(features)))
    if not records:
        raise RuntimeError(f"No complete records found under {root}")
    return records


def common_labels(
    records: list[FabricRecord],
    thresholds: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, object]]:
    raw = {
        "weave": Counter(record.weave for record in records),
        "material": Counter(record.material for record in records),
        "usage": Counter(record.usage for record in records),
        "features": Counter(feature for record in records for feature in record.features),
    }
    allowed = {
        attr: sorted(label for label, count in counter.items() if count >= thresholds[attr])
        for attr, counter in raw.items()
    }
    state = {
        "enabled": True,
        "count_source": "all_records",
        "min_counts": thresholds,
        "allowed_labels": allowed,
        "raw_label_counts": {attr: dict(sorted(counter.items())) for attr, counter in raw.items()},
        "kept_label_counts": {attr: len(labels) for attr, labels in allowed.items()},
    }
    return allowed, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed attribute-stratified VidTouch split")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--trials", type=int, default=200000)
    parser.add_argument("--min-weave", type=int, default=3)
    parser.add_argument("--min-material", type=int, default=3)
    parser.add_argument("--min-usage", type=int, default=3)
    parser.add_argument("--min-features", type=int, default=5)
    return parser.parse_args()


def labels_for(record: FabricRecord, attr: str) -> tuple[str, ...]:
    if attr == "features":
        return record.features
    return (str(getattr(record, attr)),)


def counts(records: list[FabricRecord], allowed: dict[str, list[str]]) -> dict[str, Counter[str]]:
    result = {attr: Counter() for attr in ATTRS}
    allowed_sets = {attr: set(values) for attr, values in allowed.items()}
    for record in records:
        for attr in ATTRS:
            for label in labels_for(record, attr):
                if label in allowed_sets[attr]:
                    result[attr][label] += 1
    return result


def split_score(
    all_counts: dict[str, Counter[str]],
    val_counts: dict[str, Counter[str]],
    n_records: int,
    n_val: int,
) -> tuple[float, int]:
    score = 0.0
    covered = 0
    for attr in ATTRS:
        attr_weight = 1.5 if attr != "features" else 1.0
        for label, total in all_counts[attr].items():
            val = val_counts[attr][label]
            train = total - val
            if val > 0:
                covered += 1
            if train <= 0:
                score += 1e6
            if val <= 0:
                score += attr_weight * 8.0
            target = total * n_val / n_records
            score += attr_weight * abs(val - target) / math.sqrt(max(1.0, target))
    return score, covered


def record_label_keys(
    record: FabricRecord,
    allowed_sets: dict[str, set[str]],
) -> set[tuple[str, str]]:
    return {
        (attr, label)
        for attr in ATTRS
        for label in labels_for(record, attr)
        if label in allowed_sets[attr]
    }


def coverage_candidate(
    records: list[FabricRecord],
    allowed: dict[str, list[str]],
    n_val: int,
    rng: random.Random,
) -> list[FabricRecord]:
    allowed_sets = {attr: set(values) for attr, values in allowed.items()}
    uncovered = {(attr, label) for attr, values in allowed.items() for label in values}
    available = list(records)
    selected: list[FabricRecord] = []
    rng.shuffle(available)
    while uncovered and len(selected) < n_val:
        gains = [len(record_label_keys(record, allowed_sets) & uncovered) for record in available]
        max_gain = max(gains)
        choices = [index for index, gain in enumerate(gains) if gain == max_gain]
        chosen_index = rng.choice(choices)
        chosen = available.pop(chosen_index)
        selected.append(chosen)
        uncovered -= record_label_keys(chosen, allowed_sets)
    if len(selected) < n_val:
        selected.extend(rng.sample(available, n_val - len(selected)))
    return selected


def main() -> None:
    args = parse_args()
    records = scan_records(args.data_root)
    thresholds = {
        "weave": args.min_weave,
        "material": args.min_material,
        "usage": args.min_usage,
        "features": args.min_features,
    }
    allowed, filter_state = common_labels(records, thresholds)
    n_val = max(1, round(len(records) * args.val_ratio))
    all_counts = counts(records, allowed)
    rng = random.Random(args.seed)
    best: tuple[float, int, list[FabricRecord]] | None = None
    for trial in range(args.trials):
        candidate = (
            coverage_candidate(records, allowed, n_val, rng)
            if trial % 2 == 0
            else rng.sample(records, n_val)
        )
        score, covered = split_score(all_counts, counts(candidate, allowed), len(records), n_val)
        if score >= 1e6:
            continue
        if best is None or (-covered, score) < (-best[1], best[0]):
            best = (score, covered, candidate)
    if best is None:
        raise RuntimeError("Could not create a split")
    val_ids = sorted(record.fabric_id for record in best[2])
    val_set = set(val_ids)
    train_ids = sorted(record.fabric_id for record in records if record.fabric_id not in val_set)
    val_counts = counts(best[2], allowed)
    payload = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "metadata": {
            "name": "fabric_common_v1",
            "strategy": "random_search_attribute_stratified",
            "seed": args.seed,
            "trials": args.trials,
            "val_ratio": args.val_ratio,
            "score": best[0],
            "covered_labels": best[1],
            "total_labels": sum(len(values) for values in allowed.values()),
            "label_filter": filter_state,
            "val_label_counts": {attr: dict(sorted(counter.items())) for attr, counter in val_counts.items()},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
