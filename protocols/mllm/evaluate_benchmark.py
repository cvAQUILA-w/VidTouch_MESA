#!/usr/bin/env python3
"""Reference evaluator for the frozen VidTouch fabric_common_v2 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SINGLE_LABEL_TASKS = ("weave", "material", "usage")
ALL_TASKS = (*SINGLE_LABEL_TASKS, "features")
INVALID_LABEL = "__invalid__"
EXPECTED_SPLIT_SHA256 = "4a466d92bfaf65812b835eb3f4c28787df4848eb4f0d7ba331a1f9a7267fe45d"
EXPECTED_ASSIGNMENT_SHA256 = (
    "c96c6c8af21b52e6a180361d0143f292baddc4892913848ecfa257beb4a29019"
)
EXPECTED_LABELS_SHA256 = "bd263e1e73118e4cafff754d5a123b6e78549ae3e1aa4a0081d14ca681b303d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score fabric-level VidTouch predictions with the frozen protocol."
    )
    parser.add_argument("--predictions", required=True, help="Fabric-level JSONL predictions")
    parser.add_argument("--labels", required=True, help="Canonical label.txt")
    parser.add_argument("--split", required=True, help="Frozen fabric_common_v2.json")
    parser.add_argument("--partition", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: expected at least four fields")
        fabric_id, weave, material, usage, *features = fields
        if fabric_id in records:
            raise ValueError(f"{path}:{line_number}: duplicate Fabric ID {fabric_id!r}")
        records[fabric_id] = {
            "weave": weave,
            "material": material,
            "usage": usage,
            "features": features,
        }
    return records


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        fabric_id = record.get("fabric_id")
        if not isinstance(fabric_id, str) or not fabric_id:
            raise ValueError(f"{path}:{line_number}: missing string fabric_id")
        if fabric_id in records:
            raise ValueError(f"{path}:{line_number}: duplicate Fabric ID {fabric_id!r}")
        records[fabric_id] = record
    return records


def vocabulary_maps(allowed: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for task in ALL_TASKS:
        maps[task] = {label.casefold(): label for label in allowed[task]}
        if len(maps[task]) != len(allowed[task]):
            raise ValueError(f"Case-insensitive duplicate in {task} vocabulary")
    return maps


def normalize_single(value: Any, mapping: dict[str, str]) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, False
    canonical = mapping.get(value.strip().casefold())
    return canonical, canonical is not None


def normalize_features(
    value: Any, mapping: dict[str, str]
) -> tuple[set[str], bool]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return set(), False
    canonical: list[str] = []
    for item in value:
        label = mapping.get(item.strip().casefold())
        if label is None:
            return set(), False
        canonical.append(label)
    return set(canonical), True


def prediction_payload(view: dict[str, Any]) -> dict[str, Any]:
    payload = view.get("parsed_prediction")
    if not isinstance(payload, dict):
        raise ValueError("Each view must contain a parsed_prediction object")
    return payload


def aggregate_view_predictions(
    views: list[dict[str, Any]],
    allowed: dict[str, list[str]],
    mappings: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, int]]:
    if not views:
        raise ValueError("view_predictions must not be empty")
    if not all(isinstance(view, dict) for view in views):
        raise ValueError("view_predictions must contain JSON objects")
    image_paths = [view.get("image_path") for view in views]
    if not all(isinstance(path, str) and path for path in image_paths):
        raise ValueError("Every view_prediction must contain a non-empty image_path")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("view_predictions contains duplicate image_path values")
    ordered = sorted(views, key=lambda view: str(view.get("image_path", "")))
    payloads = [prediction_payload(view) for view in ordered]
    result: dict[str, Any] = {}
    invalid_views = {task: 0 for task in ALL_TASKS}

    for task in SINGLE_LABEL_TASKS:
        votes: list[str] = []
        for payload in payloads:
            label, valid = normalize_single(payload.get(task), mappings[task])
            if not valid:
                invalid_views[task] += 1
                votes.append(INVALID_LABEL)
            else:
                votes.append(label or INVALID_LABEL)
        counts = Counter(votes)
        max_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == max_count}
        primary = votes[0]
        if primary in tied:
            winner = primary
        else:
            order = [*allowed[task], INVALID_LABEL]
            winner = next(label for label in order if label in tied)
        result[task] = None if winner == INVALID_LABEL else winner

    feature_sets: list[set[str]] = []
    for payload in payloads:
        labels, valid = normalize_features(payload.get("features"), mappings["features"])
        if not valid:
            invalid_views["features"] += 1
            labels = set()
        feature_sets.append(labels)
    primary_features = feature_sets[0]
    num_views = len(feature_sets)
    aggregated_features = []
    for label in allowed["features"]:
        votes = sum(label in labels for labels in feature_sets)
        if votes > num_views / 2 or (
            votes == num_views / 2 and label in primary_features
        ):
            aggregated_features.append(label)
    result["features"] = aggregated_features
    return result, invalid_views


def resolve_prediction(
    record: dict[str, Any],
    allowed: dict[str, list[str]],
    mappings: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, int]]:
    aggregate = record.get("aggregate_prediction")
    if isinstance(aggregate, dict):
        return aggregate, {task: 0 for task in ALL_TASKS}
    parsed = record.get("parsed_prediction")
    if isinstance(parsed, dict):
        return parsed, {task: 0 for task in ALL_TASKS}
    views = record.get("view_predictions")
    if isinstance(views, list):
        return aggregate_view_predictions(views, allowed, mappings)
    return {}, {task: 0 for task in ALL_TASKS}


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def score_single_label(
    rows: list[dict[str, Any]], task: str, labels: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    total = len(rows)
    correct = sum(row["prediction"] == row["ground_truth"] for row in rows)
    invalid = sum(not row["valid_output"] for row in rows)
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    confusion_labels = [*labels, INVALID_LABEL]
    confusion = [[0 for _ in confusion_labels] for _ in labels]
    gt_index = {label: index for index, label in enumerate(labels)}
    pred_index = {label: index for index, label in enumerate(confusion_labels)}

    for row in rows:
        prediction = row["prediction"] if row["valid_output"] else INVALID_LABEL
        confusion[gt_index[row["ground_truth"]]][pred_index[prediction]] += 1

    for label in labels:
        support = sum(row["ground_truth"] == label for row in rows)
        if support == 0:
            raise ValueError(f"{task} label {label!r} has zero support in this partition")
        tp = sum(
            row["ground_truth"] == label and row["prediction"] == label for row in rows
        )
        fp = sum(
            row["ground_truth"] != label and row["prediction"] == label for row in rows
        )
        fn = support - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        recalls.append(recall)
        f1s.append(f1)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    summary = {
        "count": total,
        "correct": correct,
        "accuracy": safe_div(correct, total),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "invalid_output_count": invalid,
        "invalid_output_rate": safe_div(invalid, total),
    }
    confusion_output = {
        "ground_truth_labels": labels,
        "prediction_labels": confusion_labels,
        "matrix": confusion,
    }
    return summary, per_class, confusion_output


def score_features(
    rows: list[dict[str, Any]], labels: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    totals = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}
    example_f1s: list[float] = []
    invalid = 0
    for row in rows:
        ground_truth = row["ground_truth"]
        prediction = row["prediction"] if row["valid_output"] else set()
        if not row["valid_output"]:
            invalid += 1
        tp_example = len(ground_truth & prediction)
        fp_example = len(prediction - ground_truth)
        fn_example = len(ground_truth - prediction)
        example_f1s.append(
            safe_div(2 * tp_example, 2 * tp_example + fp_example + fn_example)
        )
        for label in labels:
            is_ground_truth = label in ground_truth
            is_prediction = label in prediction
            totals[label]["support"] += int(is_ground_truth)
            totals[label]["tp"] += int(is_ground_truth and is_prediction)
            totals[label]["fp"] += int(not is_ground_truth and is_prediction)
            totals[label]["fn"] += int(is_ground_truth and not is_prediction)

    per_class: dict[str, dict[str, float | int]] = {}
    macro_f1s: list[float] = []
    tp_micro = fp_micro = fn_micro = 0
    for label in labels:
        counts = totals[label]
        if counts["support"] == 0:
            raise ValueError(f"Feature label {label!r} has zero support in this partition")
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_div(
            2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]
        )
        macro_f1s.append(f1)
        tp_micro += counts["tp"]
        fp_micro += counts["fp"]
        fn_micro += counts["fn"]
        per_class[label] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    summary = {
        "count": len(rows),
        "macro_f1": sum(macro_f1s) / len(macro_f1s),
        "micro_f1": safe_div(
            2 * tp_micro, 2 * tp_micro + fp_micro + fn_micro
        ),
        "example_f1": sum(example_f1s) / len(example_f1s),
        "invalid_output_count": invalid,
        "invalid_output_rate": safe_div(invalid, len(rows)),
    }
    return summary, per_class


def evaluate(
    predictions_path: Path,
    labels_path: Path,
    split_path: Path,
    partition: str,
) -> dict[str, Any]:
    split_hash = sha256(split_path)
    labels_hash = sha256(labels_path)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            f"Wrong frozen split SHA256: {split_hash}; expected {EXPECTED_SPLIT_SHA256}"
        )
    if labels_hash != EXPECTED_LABELS_SHA256:
        raise ValueError(
            f"Wrong canonical labels SHA256: {labels_hash}; "
            f"expected {EXPECTED_LABELS_SHA256}"
        )
    split = load_split(split_path)
    if split["metadata"].get("assignment_sha256") != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("Frozen Fabric assignment SHA256 does not match the protocol")
    ground_truth = load_ground_truth(labels_path)
    predictions = load_predictions(predictions_path)
    partition_ids = list(split[f"{partition}_ids"])
    expected_ids = set(partition_ids)
    actual_ids = set(predictions)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"Prediction IDs do not match {partition}: missing={missing}, extra={extra}")

    for fabric_id in partition_ids:
        record_partition = predictions[fabric_id].get("partition")
        if record_partition is not None and record_partition != partition:
            raise ValueError(
                f"Fabric {fabric_id}: record partition {record_partition!r} != {partition!r}"
            )
        if fabric_id not in ground_truth:
            raise ValueError(f"Fabric {fabric_id}: missing from canonical labels")

    allowed = split["metadata"]["label_filter"]["allowed_labels"]
    mappings = vocabulary_maps(allowed)
    expected_counts = split["metadata"]["effective_fabric_counts"][partition]
    single_rows = {task: [] for task in SINGLE_LABEL_TASKS}
    feature_rows: list[dict[str, Any]] = []
    view_invalid_counts = {task: 0 for task in ALL_TASKS}

    for fabric_id in partition_ids:
        gt = ground_truth[fabric_id]
        prediction, view_invalid = resolve_prediction(
            predictions[fabric_id], allowed, mappings
        )
        for task in ALL_TASKS:
            view_invalid_counts[task] += view_invalid[task]

        for task in SINGLE_LABEL_TASKS:
            if gt[task] not in allowed[task]:
                continue
            parsed, valid = normalize_single(prediction.get(task), mappings[task])
            single_rows[task].append(
                {
                    "fabric_id": fabric_id,
                    "ground_truth": gt[task],
                    "prediction": parsed,
                    "valid_output": valid,
                }
            )

        filtered_gt_features = set(gt["features"]) & set(allowed["features"])
        if filtered_gt_features:
            parsed_features, valid_features = normalize_features(
                prediction.get("features"), mappings["features"]
            )
            feature_rows.append(
                {
                    "fabric_id": fabric_id,
                    "ground_truth": filtered_gt_features,
                    "prediction": parsed_features,
                    "valid_output": valid_features,
                }
            )

    actual_counts = {
        **{task: len(single_rows[task]) for task in SINGLE_LABEL_TASKS},
        "features": len(feature_rows),
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Effective counts do not match frozen split: "
            f"actual={actual_counts}, expected={expected_counts}"
        )

    task_metrics: dict[str, Any] = {}
    per_class: dict[str, Any] = {}
    confusion_matrices: dict[str, Any] = {}
    for task in SINGLE_LABEL_TASKS:
        summary, task_per_class, confusion = score_single_label(
            single_rows[task], task, allowed[task]
        )
        task_metrics[task] = summary
        per_class[task] = task_per_class
        confusion_matrices[task] = confusion
    feature_summary, feature_per_class = score_features(
        feature_rows, allowed["features"]
    )
    task_metrics["features"] = feature_summary
    per_class["features"] = feature_per_class

    macro_main = (
        task_metrics["weave"]["balanced_accuracy"]
        + task_metrics["material"]["balanced_accuracy"]
        + task_metrics["usage"]["balanced_accuracy"]
        + task_metrics["features"]["macro_f1"]
    ) / 4
    legacy_main = (
        task_metrics["weave"]["accuracy"]
        + task_metrics["material"]["accuracy"]
        + task_metrics["usage"]["accuracy"]
        + task_metrics["features"]["micro_f1"]
    ) / 4

    return {
        "protocol": split["metadata"]["name"],
        "partition": partition,
        "partition_fabric_count": len(partition_ids),
        "split_file_sha256": split_hash,
        "assignment_sha256": split["metadata"]["assignment_sha256"],
        "data_manifest_sha256": split["metadata"]["data_manifest_sha256"],
        "labels_file_sha256": labels_hash,
        "predictions_file_sha256": sha256(predictions_path),
        "effective_fabric_counts": actual_counts,
        "metrics": {
            "macro_main": macro_main,
            "legacy_main": legacy_main,
            **task_metrics,
        },
        "per_class": per_class,
        "confusion_matrices": confusion_matrices,
        "view_invalid_output_count": view_invalid_counts,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(
        predictions_path=Path(args.predictions),
        labels_path=Path(args.labels),
        split_path=Path(args.split),
        partition=args.partition,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {
        "partition": result["partition"],
        "effective_fabric_counts": result["effective_fabric_counts"],
        "macro_main": result["metrics"]["macro_main"],
        "legacy_main": result["metrics"]["legacy_main"],
        "output": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
