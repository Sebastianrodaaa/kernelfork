#!/usr/bin/env bash
# Re-run all training arms with fixed seeds (MOCK fixtures on CPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${ROOT}/results/MOCK"
mkdir -p "$OUT"
SEEDS=(0 1 2)
ARMS=(kernelfork diffu_grpo dispo_branch random_span)
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    python -m kernelfork.train_grpo \
      --arm "$arm" \
      --seed "$seed" \
      --output "${OUT}/step_${arm}_seed${seed}.json"
  done
done
echo "MOCK repro complete: ${OUT}"
