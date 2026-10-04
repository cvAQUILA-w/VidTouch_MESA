#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


ATTRS = ("weave", "material", "usage")
METRICS = (
    "macro_main",
    "mkds",
    "legacy_main",
    "weave_balanced_acc",
    "material_balanced_acc",
    "usage_balanced_acc",
    "feature_f1_macro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired Fabric-ID bootstrap for VidTouch prediction exports"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=SEED7.jsonl,SEED42.jsonl,SEED123.jsonl",
    )
    parser.add_argument("--reference", default=None)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        fabric_id = str(row["fabric_id"])
        if fabric_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate Fabric ID {fabric_id}")
        rows[fabric_id] = row
    return rows


def parse_models(values: list[str]) -> dict[str, list[dict[str, dict[str, Any]]]]:
    models: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=path,path,..., got {value}")
        name, raw_paths = value.split("=", 1)
        paths = [Path(item) for item in raw_paths.split(",") if item]
        if not name or not paths:
            raise ValueError(f"Invalid model specification: {value}")
        models[name] = [load_jsonl(path) for path in paths]
    return models


def balanced_accuracy(rows: list[dict[str, Any]], attr: str) -> float:
    valid = [row for row in rows if int(row["target"][attr]) >= 0]
    classes = sorted({int(row["target"][attr]) for row in valid})
    if not classes:
        return 0.0
    recalls = []
    for cls in classes:
        class_rows = [row for row in valid if int(row["target"][attr]) == cls]
        recalls.append(
            sum(int(row["prediction"][attr]) == cls for row in class_rows)
            / len(class_rows)
        )
    return mean(recalls)


def accuracy(rows: list[dict[str, Any]], attr: str) -> float:
    valid = [row for row in rows if int(row["target"][attr]) >= 0]
    if not valid:
        return 0.0
    return mean(
        int(row["prediction"][attr]) == int(row["target"][attr])
        for row in valid
    )


def feature_f1(rows: list[dict[str, Any]]) -> tuple[float, float]:
    valid = [row for row in rows if sum(row["target"]["features"]) > 0]
    if not valid:
        return 0.0, 0.0
    width = len(valid[0]["target"]["features"])
    per_class = []
    total_tp = total_fp = total_fn = 0
    for index in range(width):
        tp = fp = fn = 0
        for row in valid:
            target = int(row["target"]["features"][index])
            pred = int(row["prediction"]["features"][index])
            tp += pred == 1 and target == 1
            fp += pred == 1 and target == 0
            fn += pred == 0 and target == 1
        if tp + fn > 0:
            per_class.append(2 * tp / max(1, 2 * tp + fp + fn))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    macro = mean(per_class) if per_class else 0.0
    micro = 2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)
    return macro, micro


def score(rows: list[dict[str, Any]]) -> dict[str, float]:
    balanced = {attr: balanced_accuracy(rows, attr) for attr in ATTRS}
    ordinary = {attr: accuracy(rows, attr) for attr in ATTRS}
    feature_macro, feature_micro = feature_f1(rows)
    components = [balanced[attr] for attr in ATTRS] + [feature_macro]
    macro_main = mean(components)
    mkds = 0.0
    if all(value > 0 and math.isfinite(value) for value in components):
        mkds = len(components) / sum(1.0 / value for value in components)
    return {
        "macro_main": macro_main,
        "mkds": mkds,
        "legacy_main": mean([ordinary[attr] for attr in ATTRS] + [feature_micro]),
        **{f"{attr}_balanced_acc": balanced[attr] for attr in ATTRS},
        "feature_f1_macro": feature_macro,
    }


def average_seed_scores(
    seed_rows: list[dict[str, dict[str, Any]]], sample_ids: list[str]
) -> dict[str, float]:
    scored = [score([rows[fabric_id] for fabric_id in sample_ids]) for rows in seed_rows]
    return {metric: mean(item[metric] for item in scored) for metric in METRICS}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    reference = args.reference or next(iter(models))
    if reference not in models:
        raise ValueError(f"Unknown reference model: {reference}")

    id_sets = [set(rows) for seeds in models.values() for rows in seeds]
    if not id_sets or any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("Every model/seed file must contain the same Fabric IDs")
    fabric_ids = sorted(id_sets[0])

    reference_targets = {
        fabric_id: models[reference][0][fabric_id]["target"] for fabric_id in fabric_ids
    }
    for model_name, seeds in models.items():
        for seed_index, rows in enumerate(seeds):
            for fabric_id in fabric_ids:
                if rows[fabric_id]["target"] != reference_targets[fabric_id]:
                    raise ValueError(
                        f"Target mismatch for {fabric_id}: {model_name} seed {seed_index}"
                    )

    point = {
        name: average_seed_scores(seeds, fabric_ids) for name, seeds in models.items()
    }
    rng = random.Random(args.seed)
    distributions = {
        name: {metric: [] for metric in METRICS} for name in models
    }
    difference_distributions = {
        name: {metric: [] for metric in METRICS}
        for name in models
        if name != reference
    }
    for _ in range(args.replicates):
        sample_ids = [rng.choice(fabric_ids) for _ in fabric_ids]
        replicate = {
            name: average_seed_scores(seeds, sample_ids)
            for name, seeds in models.items()
        }
        for name in models:
            for metric in METRICS:
                distributions[name][metric].append(replicate[name][metric])
        for name in difference_distributions:
            for metric in METRICS:
                difference_distributions[name][metric].append(
                    replicate[name][metric] - replicate[reference][metric]
                )

    output: dict[str, Any] = {
        "unit": "Fabric ID",
        "replicates": args.replicates,
        "seed": args.seed,
        "reference": reference,
        "models": {},
        "paired_differences": {},
    }
    for name in models:
        output["models"][name] = {}
        for metric in METRICS:
            values = distributions[name][metric]
            output["models"][name][metric] = {
                "estimate": point[name][metric],
                "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
            }
    for name, metric_values in difference_distributions.items():
        output["paired_differences"][name] = {}
        for metric, values in metric_values.items():
            output["paired_differences"][name][metric] = {
                "estimate": point[name][metric] - point[reference][metric],
                "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
                "bootstrap_probability_gt_zero": mean(value > 0 for value in values),
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
