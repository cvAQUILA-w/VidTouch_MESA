#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT CACHE_PID_FILE [DEVICE]" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
CACHE_PID_FILE=$3
DEVICE=${4:-cuda}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
SPLIT="$ROOT/splits/fabric_common_v2.json"

mkdir -p "$OUTPUT_ROOT"

video_count=$(find "$DATA_ROOT/TACs" -type f -iname '*.mp4' | wc -l)
while true; do
  cache_count=$(find "$DATA_ROOT/.cache/video_f16_s256" -type f -name '*.pt' | wc -l)
  if [[ "$cache_count" -eq "$video_count" ]]; then
    break
  fi
  if ! pgrep -f "vidtouch.cache.*video_f16_s256" >/dev/null; then
    echo "Cache process stopped before completion: $cache_count/$video_count" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] waiting for 16-frame video cache: $cache_count/$video_count"
  sleep 30
done

if [[ "$cache_count" -ne "$video_count" ]]; then
  echo "Incomplete 16-frame cache: $cache_count/$video_count" >&2
  exit 1
fi

BENCH_ROOT="$OUTPUT_ROOT/frozen_resnet18_mc3"
mkdir -p "$BENCH_ROOT"

if [[ ! -f "$BENCH_ROOT/rgb_cache/rgb_resnet18.npz" ]]; then
  "$PYTHON" "$ROOT/benchmark/supervised_v2/run_frozen_baselines.py" \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --cache-dir "$BENCH_ROOT/rgb_cache" \
    --output-dir "$BENCH_ROOT/rgb_runs" \
    --methods rgb_resnet18 \
    --seeds 7 42 123
fi

if [[ ! -f "$BENCH_ROOT/tac_cache/tac_mc3_18.npz" ]]; then
  "$PYTHON" "$ROOT/benchmark/supervised_v2/run_extended_frozen.py" \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --cache-dir "$BENCH_ROOT/tac_cache" \
    --output-dir "$BENCH_ROOT/tac_runs" \
    --methods tac_mc3_18 \
    --seeds 7 42 123 \
    --device "$DEVICE"
fi

if [[ ! -f "$BENCH_ROOT/fusion_runs/summary.json" ]]; then
  "$PYTHON" "$ROOT/benchmark/supervised_v2/run_cached_fusions.py" \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --rgb-cache-dir "$BENCH_ROOT/rgb_cache" \
    --tac-cache-dir "$BENCH_ROOT/tac_cache" \
    --output-dir "$BENCH_ROOT/fusion_runs" \
    --methods fusion_resnet18_mc3 \
    --seeds 7 42 123
fi

REL_ROOT="$OUTPUT_ROOT/no_relation_corrected"
for seed in 7 42 123; do
  run_dir="$REL_ROOT/seed${seed}"
  if [[ ! -f "$run_dir/best_macro.pt" ]]; then
    "$PYTHON" -m vidtouch.train \
      --config "$ROOT/configs/rebuttal/no_relation_corrected.yaml" \
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
done

echo "[$(date --iso-8601=seconds)] ALL_DONE"
