#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/tmp
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec "$project_root/evoscope/.local/glm-vllm-env/bin/python" -m vllm.entrypoints.openai.api_server \
  --model /mnt/huawei/ymb/model/GLM-4.7-Flash \
  --served-model-name glm-4.7-flash-local --host 127.0.0.1 --port 8123 \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs 1 --gpu-memory-utilization "${EVOSCOPE_GPU_MEMORY_UTILIZATION:-0.90}" --enforce-eager \
  --cpu-offload-gb "${EVOSCOPE_CPU_OFFLOAD_GB:-0}" \
  --kv-cache-memory-bytes "${EVOSCOPE_KV_CACHE_BYTES:-2147483648}" \
  --generation-config vllm --reasoning-parser glm45 \
  --no-enable-prefix-caching --no-async-scheduling
