#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT CHECKPOINT_ROOT [DEVICE]" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
CHECKPOINT_ROOT=$3
DEVICE=${4:-cuda}

for seed in 7 42 123; do
  checkpoint="$CHECKPOINT_ROOT/fabricmal_seed${seed}/best_macro.pt"
  run_dir="$OUTPUT_ROOT/mesa_submitted_seed${seed}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing submitted-MESA checkpoint: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$run_dir"

  python -m vidtouch.evaluate \
    --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT" \
    --partition val \
    --device "$DEVICE" \
    --output "$run_dir/best_macro_val.json" \
    --predictions-output "$run_dir/best_macro_val_predictions.jsonl"

  threshold=$(python - "$run_dir/best_macro_val.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
print(metrics["feature_best_global_threshold"])
PY
)

  python -m vidtouch.evaluate \
    --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT" \
    --partition test \
    --feature-threshold "$threshold" \
    --device "$DEVICE" \
    --output "$run_dir/best_macro_test.json" \
    --predictions-output "$run_dir/best_macro_test_predictions.jsonl"
done

python tools/paired_fabric_bootstrap.py \
  --model "simple=$OUTPUT_ROOT/simple_matched_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/simple_matched_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/simple_matched_seed123/best_macro_test_predictions.jsonl" \
  --model "simple_fpr=$OUTPUT_ROOT/simple_matched_fpr_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/simple_matched_fpr_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/simple_matched_fpr_seed123/best_macro_test_predictions.jsonl" \
  --model "mesa_fpr_unique=$OUTPUT_ROOT/mesa_fpr_unique_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/mesa_fpr_unique_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/mesa_fpr_unique_seed123/best_macro_test_predictions.jsonl" \
  --model "mesa_submitted=$OUTPUT_ROOT/mesa_submitted_seed7/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/mesa_submitted_seed42/best_macro_test_predictions.jsonl,$OUTPUT_ROOT/mesa_submitted_seed123/best_macro_test_predictions.jsonl" \
  --reference simple \
  --replicates 10000 \
  --output "$OUTPUT_ROOT/paired_fabric_bootstrap_all.json"
