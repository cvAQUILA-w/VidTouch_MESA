#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "val_main_score",
    "val_weave_acc",
    "val_material_acc",
    "val_usage_acc",
    "val_feature_f1_micro",
    "val_feature_f1_micro_calibrated",
    "val_weave_balanced_acc",
    "val_material_balanced_acc",
    "val_usage_balanced_acc",
    "val_feature_f1_macro",
    "val_retrieval_mean_r1",
    "val_retrieval_mean_r5",
)


def read_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(metrics_path: Path) -> dict[str, object]:
    records = read_records(metrics_path)
    best = max(records, key=lambda row: float(row["val_main_score"]))
    output: dict[str, object] = {
        "run": metrics_path.parent.name,
        "epochs": len(records),
        "best_epoch": best["epoch"],
        "last_main": records[-1]["val_main_score"],
    }
    output.update({field: best.get(field) for field in FIELDS})
    calibrated_main = (
        float(best["val_weave_acc"])
        + float(best["val_material_acc"])
        + float(best["val_usage_acc"])
        + float(best["val_feature_f1_micro_calibrated"])
    ) / 4.0
    output["calibrated_main_at_best"] = calibrated_main
    output["feature_threshold_at_best"] = best.get("val_feature_best_global_threshold")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        if root.name == "metrics.jsonl":
            paths = [root]
        else:
            paths = sorted(root.glob("*/metrics.jsonl"))
        rows.extend(summarize(path) for path in paths)
    rows.sort(key=lambda row: float(row["val_main_score"]), reverse=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
