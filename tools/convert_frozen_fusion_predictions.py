#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert frozen-fusion string predictions to MESA bootstrap format"
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--mesa-root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["fabric_id"])] = row
    return rows


def main() -> None:
    args = parse_args()
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    names = {
        "weave": vocab["weaves"],
        "material": vocab["materials"],
        "usage": vocab["usages"],
        "features": vocab["features"],
    }
    indices = {attr: {name: index for index, name in enumerate(values)} for attr, values in names.items()}
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seed in (7, 42, 123):
        benchmark_path = (
            args.benchmark_root
            / f"fusion_resnet18_mc3__linear_balanced__seed{seed}"
            / "test_predictions.jsonl"
        )
        mesa_path = (
            args.mesa_root
            / f"mesa_submitted_seed{seed}"
            / "best_macro_test_predictions.jsonl"
        )
        benchmark = read_jsonl(benchmark_path)
        mesa = read_jsonl(mesa_path)
        if set(benchmark) != set(mesa):
            raise ValueError(f"Fabric-ID mismatch for seed {seed}")
        output_path = args.output_root / f"frozen_resnet18_mc3_seed{seed}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for fabric_id in sorted(mesa):
                raw = benchmark[fabric_id]["aggregate_prediction"]
                prediction = {}
                for attr in ("weave", "material", "usage"):
                    value = raw.get(attr)
                    prediction[attr] = -1 if value is None else indices[attr][value]
                predicted_features = set(raw.get("features") or [])
                prediction["features"] = [int(name in predicted_features) for name in names["features"]]
                row = {
                    "fabric_id": fabric_id,
                    "prediction": prediction,
                    "target": mesa[fabric_id]["target"],
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(output_path)


if __name__ == "__main__":
    main()
