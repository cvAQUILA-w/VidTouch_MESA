#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CHECKPOINT DATA_ROOT OUTPUT_DIR" >&2
  exit 2
fi

checkpoint="$1"
data_root="$2"
output_dir="$3"
mkdir -p "$output_dir"

python -m vidtouch.evaluate \
  --checkpoint "$checkpoint" \
  --data-root "$data_root" \
  --partition val \
  --output "$output_dir/val.json"

threshold="$(
  python - "$output_dir/val.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metrics = json.load(handle)
print(metrics["feature_best_global_threshold"])
PY
)"

python -m vidtouch.evaluate \
  --checkpoint "$checkpoint" \
  --data-root "$data_root" \
  --partition test \
  --feature-threshold "$threshold" \
  --output "$output_dir/test.json"

echo "Validation threshold: $threshold"
echo "Test metrics: $output_dir/test.json"
