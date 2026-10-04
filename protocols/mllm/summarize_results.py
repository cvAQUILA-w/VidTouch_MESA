#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path


OUTPUT_ROOT = Path(
    os.environ.get(
        "VIDTOUCH_BENCHMARK_OUTPUT",
        "/root/VBTSINT_DATASET/benchmark/protocol_v2/outputs/formal_v1",
    )
)


def percentage(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> None:
    headers = [
        "Model",
        "Macro Main",
        "Legacy Main",
        "W Bal.",
        "M Bal.",
        "U Bal.",
        "F Macro",
        "W Acc.",
        "M Acc.",
        "U Acc.",
        "F Micro",
    ]
    rows: list[list[str]] = []
    for path in sorted(OUTPUT_ROOT.glob("*/test_metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        rows.append(
            [
                path.parent.name,
                percentage(metrics["macro_main"]),
                percentage(metrics["legacy_main"]),
                percentage(metrics["weave"]["balanced_accuracy"]),
                percentage(metrics["material"]["balanced_accuracy"]),
                percentage(metrics["usage"]["balanced_accuracy"]),
                percentage(metrics["features"]["macro_f1"]),
                percentage(metrics["weave"]["accuracy"]),
                percentage(metrics["material"]["accuracy"]),
                percentage(metrics["usage"]["accuracy"]),
                percentage(metrics["features"]["micro_f1"]),
            ]
        )
    if not rows:
        raise RuntimeError(f"No test_metrics.json files found under {OUTPUT_ROOT}")
    lines = [
        "# VidTouch MLLM Benchmark Summary",
        "",
        "All values are percentages scored by the frozen reference evaluator.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    output = OUTPUT_ROOT / "summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
