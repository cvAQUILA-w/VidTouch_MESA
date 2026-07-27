#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 DATA_ROOT RUN_ROOT [CONFIG]" >&2
  exit 2
fi

data_root="$1"
run_root="$2"
config="${3:-configs/mesa.yaml}"

for seed in 42 7 123; do
  python -m vidtouch.train \
    --config "$config" \
    --seed "$seed" \
    --data-root "$data_root" \
    --output-dir "$run_root/seed${seed}"
done
