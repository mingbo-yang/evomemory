#!/usr/bin/env bash
# Phase 5 core ablations, chained with no idle gap.
#
# Arms run in order of scientific value so that partial progress is still
# informative: no_experience answers "does experience matter at all"; then
# outcome_hidden (the field-level ablation), then the When knob, then the
# retrieval and pool manipulations.
#
# full_static is NOT repeated here -- Phase 4 provides it and the plan says the
# 2x2's Full cell reuses it.
#
# Models: the plan names glm4-9b + qwen3-8b.  An earlier substitution to
# qwen3-4b was WITHDRAWN on 2026-09-12: glm4-9b loads fine from the
# sha256-verified local copy (reports/progress.md 6.5-6.7), so the plan's
# original pair stands and no substitution is applied.  The default below is
# therefore glm4-9b,qwen3-8b, and scheduling is owned by dispatcher.py.
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-2}"
MODELS="${MODELS:-glm4-9b,qwen3-8b}"
ARMS="${ARMS:-no_experience outcome_hidden fixed_rounds random_retrieve positive_only sr_j_fixed sr_j_stop}"

for arm in $ARMS; do
  echo "########## ablation arm=$arm models=$MODELS $(date) ##########"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" supervisor.py \
      --gpu "$GPU" --models "$MODELS" --arm "$arm" \
      --out-dir runs/ablation --stuck-min 12 --max-attempts 3 \
      --seed 42 --alpha 0.5 || echo "!!! arm $arm failed"
done
echo "########## ablation chain complete $(date) ##########"
