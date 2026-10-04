#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT REFERENCE_ROOT [DEVICE]" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
REFERENCE_ROOT=$3
DEVICE=${4:-cuda}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${VIDTOUCH_DINOV2_REPO:-}" ]]; then
  cached_dinov2="$HOME/.cache/torch/hub/facebookresearch_dinov2_main"
  if [[ -d "$cached_dinov2" ]]; then
    export VIDTOUCH_DINOV2_REPO="$cached_dinov2"
  fi
fi

mkdir -p "$OUTPUT_ROOT"

for frames in 6 24; do
  "$PYTHON" -m vidtouch.cache \
    --data-root "$DATA_ROOT" \
    --cache-root "$DATA_ROOT/.cache" \
    --frames "$frames" \
    --size 160 \
    --dtype float16

  config="$ROOT/configs/rebuttal/frames_${frames}.yaml"
  for seed in 7 42 123; do
    run_dir="$OUTPUT_ROOT/frames_${frames}_seed${seed}"
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

    rm -f "$run_dir/best.pt" "$run_dir/best_legacy.pt" "$run_dir/last.pt"
  done
done

for frames in 6 24; do
  "$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
    --model "frames12=$REFERENCE_ROOT/mesa_submitted_seed7/best_macro_test_predictions.jsonl,$REFERENCE_ROOT/mesa_submitted_seed42/best_macro_test_predictions.jsonl,$REFERENCE_ROOT/mesa_submitted_seed123/best_macro_test_predictions.jsonl" \
    --model "frames${frames}=$OUTPUT_ROOT/frames_${frames}_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/frames_${frames}_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/frames_${frames}_seed123/best_macro_test_predictions.jsonl" \
    --reference frames12 \
    --replicates 10000 \
    --output "$OUTPUT_ROOT/frames_${frames}_vs_12_bootstrap.json"
done

"$PYTHON" - "$OUTPUT_ROOT" "$REFERENCE_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
reference_root = Path(sys.argv[2])
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
for frames in (6, 12, 24):
    if frames == 12:
        paths = [
            reference_root / f"mesa_submitted_seed{seed}" / "best_macro_test.json"
            for seed in (7, 42, 123)
        ]
    else:
        paths = [
            output_root / f"frames_{frames}_seed{seed}" / "best_macro_test.json"
            for seed in (7, 42, 123)
        ]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary[str(frames)] = {
        name: {
            "mean": statistics.mean(row[source] for row in rows),
            "sample_sd": statistics.stdev(row[source] for row in rows),
        }
        for name, source in keys.items()
    }

(output_root / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

echo "[$(date --iso-8601=seconds)] FRAME_ABLATION_ALL_DONE"
