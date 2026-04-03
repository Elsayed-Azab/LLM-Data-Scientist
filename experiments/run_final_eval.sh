#!/bin/bash
# Run the full definitive evaluation with gpt-4o-mini
# Usage: bash experiments/run_final_eval.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "Starting full evaluation with gpt-4o-mini..."
echo "============================================="

for dataset in arab_barometer wvs gss; do
  for agent in single rag multi; do
    echo ""
    echo ">>> Running $agent on $dataset..."
    .venv/bin/python experiments/run_evaluation.py \
      --model gpt-4o-mini \
      --agents $agent \
      --datasets $dataset \
      --report "experiments/results/final_${agent}_${dataset}.md"
    echo ">>> Done: $agent on $dataset"
  done
done

echo ""
echo "============================================="
echo "ALL DONE! Results in experiments/results/final_*.md"
