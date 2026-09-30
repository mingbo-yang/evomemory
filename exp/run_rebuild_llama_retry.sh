#!/usr/bin/env bash
# Retry the llama3.1-8b library rebuild.
#
# The first attempt died inside vLLM's memory profiler:
#   "Initial free memory 55.17 GiB, current free memory 63.48 GiB. This happens
#    when other processes sharing the same container release GPU memory while
#    vLLM is profiling during initialization."
# That is a race with the other users on this shared box, not a config error, so
# it is retried with backoff.  Later attempts run after the qwen3-8b rebuild has
# released the device, which also removes the race.
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-3}"
KMAP="glm4-9b:wmt19_en_zh=2,qwen3-4b:wmt19_en_zh=2,qwen3-8b:wmt19_en_zh=2,llama3.1-8b:wmt19_en_zh=5,llama3.1-8b:wmt19_zh_en=4,llama3.1-8b:gigaword=4,qwen3-8b:wmt19_zh_en=3,coedit_gec=8"

# wait for any running rebuild/experiment on this GPU to finish
while pgrep -f "build_experience.py --gpu $GPU" >/dev/null 2>&1; do sleep 30; done
echo "=== device free, starting llama rebuild at $(date) ==="

for attempt in 1 2 3; do
  echo "########## llama3.1-8b attempt $attempt ##########"
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" build_experience.py \
      --gpu "$GPU" --models llama3.1-8b \
      --tasks wmt19_en_zh,wmt19_zh_en,gigaword \
      --interventions 2 --k-map "$KMAP" --min-positive 8
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== llama rebuild OK on attempt $attempt ==="; exit 0; fi
  echo "!!! attempt $attempt failed (rc=$rc); backing off"
  sleep 60
done
echo "=== llama rebuild FAILED after 3 attempts ==="
exit 1
