#!/usr/bin/env bash
# llama3.1-8b / gigaword at K=4: K=3 yielded 7 positives against a required 8.
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-3}"
for attempt in 1 2 3; do
  echo "########## llama gigaword K=4 attempt $attempt $(date) ##########"
  CUDA_VISIBLE_DEVICES="$GPU" ./watchdog_run.sh "$GPU" 10 "logs/rebuild_llama_giga.log" \
    env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" build_experience.py \
      --gpu "$GPU" --models llama3.1-8b --tasks gigaword \
      --interventions 4 --min-positive 8
  rc=$?
  [ $rc -eq 0 ] && { echo "=== gigaword OK on attempt $attempt ==="; exit 0; }
  echo "!!! attempt $attempt rc=$rc; backoff"
  sleep 45
done
exit 1
