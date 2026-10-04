#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 FOLLOWUP_ROOT MATCHED_ROOT DATA_ROOT" >&2
  exit 2
fi

FOLLOWUP_ROOT=$1
MATCHED_ROOT=$2
DATA_ROOT=$3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ANALYSIS_ROOT="$FOLLOWUP_ROOT/analysis"
mkdir -p "$ANALYSIS_ROOT"

"$PYTHON" "$ROOT/tools/convert_frozen_fusion_predictions.py" \
  --benchmark-root "$FOLLOWUP_ROOT/frozen_resnet18_mc3/fusion_runs" \
  --mesa-root "$MATCHED_ROOT" \
  --vocab "$MATCHED_ROOT/mesa_fpr_unique_seed7/vocab.json" \
  --output-root "$ANALYSIS_ROOT/frozen_converted"

"$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
  --model "frozen=$ANALYSIS_ROOT/frozen_converted/frozen_resnet18_mc3_seed7.jsonl,$ANALYSIS_ROOT/frozen_converted/frozen_resnet18_mc3_seed42.jsonl,$ANALYSIS_ROOT/frozen_converted/frozen_resnet18_mc3_seed123.jsonl" \
  --model "mesa=$MATCHED_ROOT/mesa_submitted_seed7/best_macro_test_predictions.jsonl,$MATCHED_ROOT/mesa_submitted_seed42/best_macro_test_predictions.jsonl,$MATCHED_ROOT/mesa_submitted_seed123/best_macro_test_predictions.jsonl" \
  --reference frozen \
  --replicates 10000 \
  --output "$ANALYSIS_ROOT/paired_frozen_vs_mesa.json"

"$PYTHON" "$ROOT/tools/paired_fabric_bootstrap.py" \
  --model "mesa=$MATCHED_ROOT/mesa_submitted_seed7/best_macro_test_predictions.jsonl,$MATCHED_ROOT/mesa_submitted_seed42/best_macro_test_predictions.jsonl,$MATCHED_ROOT/mesa_submitted_seed123/best_macro_test_predictions.jsonl" \
  --model "no_relation=$FOLLOWUP_ROOT/no_relation_corrected/seed7/best_macro_test_predictions.jsonl,$FOLLOWUP_ROOT/no_relation_corrected/seed42/best_macro_test_predictions.jsonl,$FOLLOWUP_ROOT/no_relation_corrected/seed123/best_macro_test_predictions.jsonl" \
  --reference mesa \
  --replicates 10000 \
  --output "$ANALYSIS_ROOT/paired_no_relation_vs_mesa.json"

"$PYTHON" "$ROOT/tools/rebuttal_error_analysis.py" \
  --prediction "$MATCHED_ROOT/mesa_submitted_seed7/best_macro_test_predictions.jsonl" \
  --prediction "$MATCHED_ROOT/mesa_submitted_seed42/best_macro_test_predictions.jsonl" \
  --prediction "$MATCHED_ROOT/mesa_submitted_seed123/best_macro_test_predictions.jsonl" \
  --vocab "$MATCHED_ROOT/mesa_fpr_unique_seed7/vocab.json" \
  --output "$ANALYSIS_ROOT/mesa_error_analysis.json"

"$PYTHON" "$ROOT/tools/profile_rebuttal_efficiency.py" \
  --checkpoint "mesa=$MATCHED_ROOT/mesa_submitted_checkpoints/fabricmal_seed7/best_macro.pt" \
  --checkpoint "matched_simple=$MATCHED_ROOT/simple_matched_seed7/best_macro.pt" \
  --data-root "$DATA_ROOT" \
  --output "$ANALYSIS_ROOT/efficiency.json"

echo "[$(date --iso-8601=seconds)] POSTPROCESS_ALL_DONE"
