#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT MATCHED_ROOT EXISTING_SPLIT_ROOT DEVICE" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
MATCHED_ROOT=$3
EXISTING_SPLIT_ROOT=$4
DEVICE=$5
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
SEEDS=(7 42 123)

if [[ -z "${VIDTOUCH_DINOV2_REPO:-}" ]]; then
  cached_dinov2="$HOME/.cache/torch/hub/facebookresearch_dinov2_main"
  if [[ -d "$cached_dinov2" ]]; then
    export VIDTOUCH_DINOV2_REPO="$cached_dinov2"
  fi
fi

required=(
  "$ROOT/splits/rebuttal_sensitivity_c.json"
  "$ROOT/splits/rebuttal_sensitivity_d.json"
  "$ROOT/splits/fabric_common_v2_lowshot25.json"
  "$ROOT/splits/fabric_common_v2_lowshot50.json"
  "$ROOT/splits/fabric_common_v2_lowshot75.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required manifest: $path" >&2; exit 1; }
done

mkdir -p \
  "$OUTPUT_ROOT/split_sensitivity" \
  "$OUTPUT_ROOT/semantic_alignment" \
  "$OUTPUT_ROOT/lowshot"

run_one() {
  local config=$1
  local run_dir=$2
  local seed=$3
  mkdir -p "$run_dir"

  if [[ ! -f "$run_dir/best_macro_test.json" || ! -f "$run_dir/best_macro_test_predictions.jsonl" ]]; then
    if [[ ! -f "$run_dir/best_macro.pt" ]]; then
      echo "[$(date --iso-8601=seconds)] TRAIN config=$config seed=$seed run=$run_dir"
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

    local threshold
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
  else
    echo "[$(date --iso-8601=seconds)] SKIP completed run=$run_dir"
  fi

  # Keep all three reproducibility checkpoints for MESA/component runs. The
  # simple control is fully reproducible from its resolved config and seed, so
  # retain only its selected Macro checkpoint to stay within server capacity.
  rm -f "$run_dir/best.pt"
  if [[ "$run_dir" == *"_simple_seed"* ]]; then
    rm -f "$run_dir/best_legacy.pt" "$run_dir/last.pt"
  fi
  echo "[$(date --iso-8601=seconds)] DONE run=$run_dir"
}

pred_list() {
  local pattern=$1
  local result=""
  local seed
  for seed in "${SEEDS[@]}"; do
    local path=${pattern//\{seed\}/$seed}/best_macro_test_predictions.jsonl
    if [[ -z "$result" ]]; then result=$path; else result="$result,$path"; fi
  done
  printf '%s' "$result"
}

# 1. Complete a five-split matched comparison: anchor + existing A/B + new C/D.
for split_name in c d; do
  for method in mesa simple; do
    config="$ROOT/configs/rebuttal/split_${split_name}_${method}.yaml"
    for seed in "${SEEDS[@]}"; do
      run_one "$config" \
        "$OUTPUT_ROOT/split_sensitivity/split_${split_name}_${method}_seed${seed}" \
        "$seed"
    done
  done
  "$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
    --model "simple=$(pred_list "$OUTPUT_ROOT/split_sensitivity/split_${split_name}_simple_seed{seed}")" \
    --model "mesa=$(pred_list "$OUTPUT_ROOT/split_sensitivity/split_${split_name}_mesa_seed{seed}")" \
    --reference simple \
    --replicates 10000 \
    --output "$OUTPUT_ROOT/split_sensitivity/split_${split_name}_paired_bootstrap.json"
done

# 2. Separate cross-modal alignment from semantic multi-positive targets.
for variant in no_alignment fabric_only; do
  for seed in "${SEEDS[@]}"; do
    run_one "$ROOT/configs/ablation/${variant}.yaml" \
      "$OUTPUT_ROOT/semantic_alignment/${variant}_seed${seed}" \
      "$seed"
  done
done

semantic_predictions=$(pred_list "$MATCHED_ROOT/mesa_submitted_seed{seed}")
no_alignment_predictions=$(pred_list "$OUTPUT_ROOT/semantic_alignment/no_alignment_seed{seed}")
fabric_only_predictions=$(pred_list "$OUTPUT_ROOT/semantic_alignment/fabric_only_seed{seed}")
"$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
  --model "no_alignment=$no_alignment_predictions" \
  --model "fabric_only=$fabric_only_predictions" \
  --model "semantic_alignment=$semantic_predictions" \
  --reference no_alignment \
  --replicates 10000 \
  --output "$OUTPUT_ROOT/semantic_alignment/paired_vs_no_alignment.json"
"$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
  --model "fabric_only=$fabric_only_predictions" \
  --model "semantic_alignment=$semantic_predictions" \
  --reference fabric_only \
  --replicates 10000 \
  --output "$OUTPUT_ROOT/semantic_alignment/paired_semantic_vs_fabric_only.json"

# 3. Fixed nested Train subsets; Validation and Test remain the anchor split.
for size in 25 50 75; do
  for method in mesa simple; do
    config="$ROOT/configs/rebuttal/lowshot_${size}_${method}.yaml"
    for seed in "${SEEDS[@]}"; do
      run_one "$config" \
        "$OUTPUT_ROOT/lowshot/lowshot_${size}_${method}_seed${seed}" \
        "$seed"
    done
  done
  "$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
    --model "simple=$(pred_list "$OUTPUT_ROOT/lowshot/lowshot_${size}_simple_seed{seed}")" \
    --model "mesa=$(pred_list "$OUTPUT_ROOT/lowshot/lowshot_${size}_mesa_seed{seed}")" \
    --reference simple \
    --replicates 10000 \
    --output "$OUTPUT_ROOT/lowshot/lowshot_${size}_paired_bootstrap.json"
done

"$PYTHON" "$ROOT/tools/summarize_rebuttal_expansion.py" \
  --output-root "$OUTPUT_ROOT" \
  --matched-root "$MATCHED_ROOT" \
  --existing-split-root "$EXISTING_SPLIT_ROOT"

echo "[$(date --iso-8601=seconds)] REBUTTAL_EXPANSION_ALL_DONE"
