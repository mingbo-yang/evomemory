#!/usr/bin/env bash
# Build the shared initial experience library for every non-32B model.
# Per-task K is applied in ONE invocation so each checkpoint loads exactly once
# (loading from the NFS model mount costs several minutes per shard).
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-3}"
MODELS="${MODELS:-llama3.1-8b,qwen3-4b,qwen3-8b}"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" build_experience.py \
    --gpu "$GPU" --models "$MODELS" --tasks all \
    --interventions 2 --k-map "coedit_gec=8" --min-positive 8
