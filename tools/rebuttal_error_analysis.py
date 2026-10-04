#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MESA Test errors across seeds")
    parser.add_argument("--prediction", action="append", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    names = {
        "weave": vocab["weaves"],
        "material": vocab["materials"],
        "usage": vocab["usages"],
        "features": vocab["features"],
    }
    confusion = {attr: Counter() for attr in ("weave", "material", "usage")}
    feature_fp, feature_fn = Counter(), Counter()
    valid_counts = Counter()

    for path in args.prediction:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for attr in ("weave", "material", "usage"):
                target = int(row["target"][attr])
                prediction = int(row["prediction"][attr])
                if target < 0:
                    continue
                valid_counts[attr] += 1
                if target != prediction:
                    confusion[attr][(names[attr][target], names[attr][prediction])] += 1
            target_features = row["target"]["features"]
            predicted_features = row["prediction"]["features"]
            if sum(target_features) <= 0:
                continue
            valid_counts["features"] += 1
            for index, (target, prediction) in enumerate(zip(target_features, predicted_features)):
                if target == 0 and prediction == 1:
                    feature_fp[names["features"][index]] += 1
                elif target == 1 and prediction == 0:
                    feature_fn[names["features"][index]] += 1

    output = {
        "aggregation": "counts across all supplied seed-level Test predictions",
        "valid_prediction_rows": dict(valid_counts),
        "top_confusions": {
            attr: [
                {"target": pair[0], "prediction": pair[1], "count": count}
                for pair, count in confusion[attr].most_common(10)
            ]
            for attr in confusion
        },
        "feature_false_positives": dict(feature_fp.most_common()),
        "feature_false_negatives": dict(feature_fn.most_common()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
