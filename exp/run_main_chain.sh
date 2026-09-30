#!/usr/bin/env bash
# Phase 4 main run: all four non-32B models, one process, serialised on GPU 3.
#
# GPU 2 is shared with another tenant (isaac, ~17 GB).  Every glm4-9b load
# attempted there hung after its weights were resident, while identical loads on
# GPU 3 (exclusively ours) always completed.  GPU 2 is therefore left to the
# other tenant for the main run.
#
# A single invocation with --models a,b,c,d loads each checkpoint exactly once
# (loading is minutes per shard on the NFS model mount) and runs all four tasks
# per model.
set -u
cd "$(dirname "$0")"
GPU="${GPU:-3}"
MODELS="${MODELS:-glm4-9b,qwen3-4b,qwen3-8b,llama3.1-8b}"
for attempt in 1 2 3; do
  echo "########## main chain attempt $attempt models=$MODELS $(date) ##########"
  CUDA_VISIBLE_DEVICES="$GPU" env GPU="$GPU" MODELS="$MODELS" SEED=42 TASKS=all ./run_main.sh \
    && { echo "=== chain OK on attempt $attempt ==="; exit 0; }
  echo "!!! chain attempt $attempt failed; backoff 60s"
  sleep 60
done
exit 1
