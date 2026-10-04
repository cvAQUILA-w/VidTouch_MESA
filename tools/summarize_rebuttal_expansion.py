#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (7, 42, 123)
METRICS = {
    "macro_main": "main_score",
    "mkds": "mkds",
    "legacy_main": "legacy_main_score",
    "weave": "weave_balanced_acc",
    "material": "material_balanced_acc",
    "usage": "usage_balanced_acc",
    "features": "feature_f1_macro",
    "retrieval_r1": "retrieval_mean_r1",
    "retrieval_r5": "retrieval_mean_r5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--existing-split-root", type=Path, required=True)
    return parser.parse_args()


def load_rows(pattern: str) -> list[dict[str, float]]:
    rows = []
    for seed in SEEDS:
        path = Path(pattern.format(seed=seed)) / "best_macro_test.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for name, source in METRICS.items():
        values = [float(row[source]) for row in rows]
        result[name] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }
    return result


def paired_gap(
    mesa: dict[str, dict[str, float]], simple: dict[str, dict[str, float]]
) -> dict[str, float]:
    return {
        metric: mesa[metric]["mean"] - simple[metric]["mean"]
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    output = args.output_root
    split_root = output / "split_sensitivity"
    lowshot_root = output / "lowshot"
    semantic_root = output / "semantic_alignment"

    split_patterns = {
        "anchor": {
            "mesa": str(args.matched_root / "mesa_submitted_seed{seed}"),
            "simple": str(args.matched_root / "simple_matched_seed{seed}"),
        },
        "a": {
            method: str(args.existing_split_root / f"split_a_{method}_seed{{seed}}")
            for method in ("mesa", "simple")
        },
        "b": {
            method: str(args.existing_split_root / f"split_b_{method}_seed{{seed}}")
            for method in ("mesa", "simple")
        },
        "c": {
            method: str(split_root / f"split_c_{method}_seed{{seed}}")
            for method in ("mesa", "simple")
        },
        "d": {
            method: str(split_root / f"split_d_{method}_seed{{seed}}")
            for method in ("mesa", "simple")
        },
    }
    split_summary: dict[str, object] = {"splits": {}}
    all_gaps = {metric: [] for metric in METRICS}
    for split_name, methods in split_patterns.items():
        method_summary = {
            method: summarize_rows(load_rows(pattern))
            for method, pattern in methods.items()
        }
        gaps = paired_gap(method_summary["mesa"], method_summary["simple"])
        split_summary["splits"][split_name] = {
            **method_summary,
            "mesa_minus_simple": gaps,
        }
        for metric, value in gaps.items():
            all_gaps[metric].append(value)

    # Five split-level paired differences. The t interval describes variation
    # across split assignments; per-Fabric bootstrap files remain separate.
    t975_df4 = 2.7764451051977987
    split_summary["across_splits"] = {}
    for metric, values in all_gaps.items():
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        half_width = t975_df4 * sd / (len(values) ** 0.5)
        split_summary["across_splits"][metric] = {
            "mean_gap": mean,
            "sample_sd_gap": sd,
            "t95_ci": [mean - half_width, mean + half_width],
            "positive_splits": sum(value > 0 for value in values),
            "total_splits": len(values),
        }

    lowshot_summary: dict[str, object] = {"sizes": {}}
    for size in (25, 50, 75, 100):
        if size == 100:
            patterns = split_patterns["anchor"]
        else:
            patterns = {
                method: str(lowshot_root / f"lowshot_{size}_{method}_seed{{seed}}")
                for method in ("mesa", "simple")
            }
        method_summary = {
            method: summarize_rows(load_rows(pattern))
            for method, pattern in patterns.items()
        }
        lowshot_summary["sizes"][str(size)] = {
            **method_summary,
            "mesa_minus_simple": paired_gap(
                method_summary["mesa"], method_summary["simple"]
            ),
        }

    semantic_patterns = {
        "no_alignment": str(semantic_root / "no_alignment_seed{seed}"),
        "fabric_only": str(semantic_root / "fabric_only_seed{seed}"),
        "semantic_alignment": split_patterns["anchor"]["mesa"],
    }
    semantic_summary = {
        name: summarize_rows(load_rows(pattern))
        for name, pattern in semantic_patterns.items()
    }
    semantic_summary["semantic_minus_no_alignment"] = {
        metric: semantic_summary["semantic_alignment"][metric]["mean"]
        - semantic_summary["no_alignment"][metric]["mean"]
        for metric in METRICS
    }
    semantic_summary["semantic_minus_fabric_only"] = {
        metric: semantic_summary["semantic_alignment"][metric]["mean"]
        - semantic_summary["fabric_only"][metric]["mean"]
        for metric in METRICS
    }

    payload = {
        "protocol": {
            "seeds": list(SEEDS),
            "checkpoint": "Validation Macro Main",
            "test_access": "after Validation checkpoint and feature-threshold selection",
            "split_unit": "Fabric ID",
        },
        "five_split_comparison": split_summary,
        "nested_lowshot_comparison": lowshot_summary,
        "semantic_alignment_ablation": semantic_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
