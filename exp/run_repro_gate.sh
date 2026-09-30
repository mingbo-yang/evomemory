#!/usr/bin/env bash
# Phase 0b driver: run the Direct-Zero reproduction gate with ONE process per
# model.  vLLM does not reliably release GPU memory when an LLM object is
# dropped, so a single process that loops over five checkpoints accumulates
# allocations and OOMs on the 32B model.  Each model therefore gets a fresh
# process, and the per-model JSON reports are merged at the end.
set -u

cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-2}"
SAMPLES="${SAMPLES:-5}"
MODELS="${MODELS:-glm4-9b llama3.1-8b qwen3-4b qwen3-8b qwen3-32b}"
TASKS="${TASKS:-wmt19_en_zh wmt19_zh_en coedit_gec gigaword}"

echo "=== repro gate start: GPU=$GPU samples=$SAMPLES ==="
overall=0
for m in $MODELS; do
  echo "--- model: $m ---"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" repro_gate.py \
      --gpu "$GPU" --samples "$SAMPLES" --models "$m" \
      --tasks "$(echo $TASKS | tr ' ' ',')"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! model $m reported divergence (rc=$rc)"
    overall=$rc
  fi
  # give the driver a moment to reclaim the device
  sleep 5
done

echo "=== repro gate finished (overall rc=$overall) ==="
exit $overall
