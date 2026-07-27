#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_ROOT RUN_ROOT" >&2
  exit 2
fi

data_root="$1"
run_root="$2"
variants=(
  no_fpr
  fabric_only
  no_texture
  no_balance
  no_alignment
  image_only
  tactile_only
)

for variant in "${variants[@]}"; do
  for seed in 42 7 123; do
    python -m vidtouch.train \
      --config "configs/ablation/${variant}.yaml" \
      --seed "$seed" \
      --data-root "$data_root" \
      --output-dir "$run_root/${variant}_seed${seed}"
  done
done
