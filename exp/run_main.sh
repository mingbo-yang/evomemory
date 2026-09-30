#!/usr/bin/env bash
# Phase 4 main experiment: Full-Static over every model whose experience library
# passed the positive-pool precheck.  alpha is frozen at 0.5 and
# draft_source=stored (user-approved) so every arm starts from an identical y0.
#
# Each attempt runs under watchdog_run.sh, which kills a run whose own process tree
# stops making I/O or CPU progress (card utilisation is useless here: the box is shared)
# for 10 minutes -- the observed failure mode is an engine-core load hang that
# never exits, so a plain retry loop would wait forever.
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-2}"
MODELS="${MODELS:-glm4-9b}"
SEED="${SEED:-42}"
TASKS="${TASKS:-all}"
LOG="logs/main_${MODELS//,/_}_s${SEED}.log"

for attempt in 1 2 3 4; do
  echo "########## main/full_static attempt $attempt models=$MODELS seed=$SEED $(date) ##########"
  CUDA_VISIBLE_DEVICES="$GPU" ./watchdog_run.sh 12 "$LOG" \
    env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" run_experiment.py \
      --arm full_static --gpu "$GPU" --models "$MODELS" --tasks "$TASKS" \
      --seed "$SEED" --alpha 0.5 --max-rounds 3 \
      --draft-source stored --out-dir runs/main
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== main OK on attempt $attempt ==="; exit 0; fi
  if [ $rc -eq 124 ]; then
    echo "!!! attempt $attempt killed by watchdog (load hang); retrying immediately"
  else
    echo "!!! attempt $attempt failed (rc=$rc); backing off 60s"
    sleep 60
  fi
done
echo "=== main FAILED after 4 attempts ==="
exit 1
