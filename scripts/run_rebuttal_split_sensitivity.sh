#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT [DEVICE]" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
DEVICE=${3:-cuda}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}

# Keep checkpoint re-evaluation offline when the DINOv2 Torch Hub repository
# is already cached on the worker.
if [[ -z "${VIDTOUCH_DINOV2_REPO:-}" ]]; then
  cached_dinov2="$HOME/.cache/torch/hub/facebookresearch_dinov2_main"
  if [[ -d "$cached_dinov2" ]]; then
    export VIDTOUCH_DINOV2_REPO="$cached_dinov2"
  fi
fi

mkdir -p "$OUTPUT_ROOT"

for split_name in a b; do
  for method in mesa simple; do
    config="$ROOT/configs/rebuttal/split_${split_name}_${method}.yaml"
    for seed in 7 42 123; do
      run_dir="$OUTPUT_ROOT/split_${split_name}_${method}_seed${seed}"
      if [[ ! -f "$run_dir/best_macro.pt" ]]; then
        "$PYTHON" -m vidtouch.train \
          --config "$config" \
          --data-root "$DATA_ROOT" \
          --output-dir "$run_dir" \
          --device "$DEVICE" \
          --seed "$seed"
      fi

      "$PYTHON" -m vidtouch.evaluate \
        --checkpoint "$run_dir/best_macro.pt" \
        --data-root "$DATA_ROOT" \
        --partition val \
        --device "$DEVICE" \
        --output "$run_dir/best_macro_val.json" \
        --predictions-output "$run_dir/best_macro_val_predictions.jsonl"

      threshold=$("$PYTHON" - "$run_dir/best_macro_val.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["feature_best_global_threshold"])
PY
)

      "$PYTHON" -m vidtouch.evaluate \
        --checkpoint "$run_dir/best_macro.pt" \
        --data-root "$DATA_ROOT" \
        --partition test \
        --feature-threshold "$threshold" \
        --device "$DEVICE" \
        --output "$run_dir/best_macro_test.json" \
        --predictions-output "$run_dir/best_macro_test_predictions.jsonl"

      # Sensitivity runs only require the Validation-selected checkpoint and
      # prediction exports. Remove redundant aliases and non-selected roles.
      rm -f "$run_dir/best.pt" "$run_dir/best_legacy.pt" "$run_dir/last.pt"
    done
  done
done

for split_name in a b; do
  "$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
    --model "simple=$OUTPUT_ROOT/split_${split_name}_simple_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/split_${split_name}_simple_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/split_${split_name}_simple_seed123/best_macro_test_predictions.jsonl" \
    --model "mesa=$OUTPUT_ROOT/split_${split_name}_mesa_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/split_${split_name}_mesa_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/split_${split_name}_mesa_seed123/best_macro_test_predictions.jsonl" \
    --reference simple \
    --replicates 10000 \
    --output "$OUTPUT_ROOT/split_${split_name}_paired_bootstrap.json"
done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys = {
    "macro_main": "main_score",
    "mkds": "mkds",
    "legacy_main": "legacy_main_score",
    "weave": "weave_balanced_acc",
    "material": "material_balanced_acc",
    "usage": "usage_balanced_acc",
    "features": "feature_f1_macro",
}
summary = {}
for split_name in ("a", "b"):
    summary[split_name] = {}
    for method in ("mesa", "simple"):
        rows = [
            json.loads(
                (root / f"split_{split_name}_{method}_seed{seed}" / "best_macro_test.json").read_text()
            )
            for seed in (7, 42, 123)
        ]
        summary[split_name][method] = {
            name: {
                "mean": statistics.mean(row[source] for row in rows),
                "sample_sd": statistics.stdev(row[source] for row in rows),
            }
            for name, source in keys.items()
        }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[$(date --iso-8601=seconds)] SPLIT_SENSITIVITY_ALL_DONE"
