#!/usr/bin/env bash
# Persistently fetch the ONE missing GLM-4-9B shard.
#
# Why a retry loop: the NFS copy of this shard is unreadable server-side
# (see reports/glm4_nfs_root_cause.md), and HuggingFace throughput on this host
# swings between ~14 MB/s and ~1 KB/s depending on contention (load average is
# 230+).  hf_hub_download resumes from the .incomplete file, so we simply keep
# retrying with backoff until the file is complete and sha256-verified.
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
EXPECT=029329f34bb73bd71b7e5f6391c62383d6bf888b88d4168fb7c416ea283da3f7
INC=$(find /home/ymb/glm_local/hf_shard2 -name '*.incomplete' 2>/dev/null | head -1)
TARGET=/home/ymb/glm_local/model/model-00002-of-00004.safetensors

for attempt in $(seq 1 40); do
  echo "=== attempt $attempt  $(date) ==="
  before=$(stat -c %s "$INC" 2>/dev/null || echo 0)
  HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" "$PY" - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(repo_id="zai-org/glm-4-9b-chat-hf",
                        filename="model-00002-of-00004.safetensors",
                        local_dir="/home/ymb/glm_local/hf_shard2",
                        revision="8599336fc6c125203efb2360bfaf4c80eef1d1bf")
    print("FETCHED", p, flush=True)
except Exception as e:
    print("fetch error:", type(e).__name__, str(e)[:120], flush=True)
PYEOF
  after=$(stat -c %s "$INC" 2>/dev/null || echo 0)
  echo "  bytes: $before -> $after"
  # if the completed file exists, verify it
  F=/home/ymb/glm_local/hf_shard2/model-00002-of-00004.safetensors
  if [ -f "$F" ] && [ "$(stat -c %s "$F")" = "4895075168" ]; then
    got=$(sha256sum "$F" | cut -d' ' -f1)
    echo "  sha256: $got"
    if [ "$got" = "$EXPECT" ]; then
      cp "$F" "$TARGET"
      echo "=== SHARD2 COMPLETE AND VERIFIED -> $TARGET  $(date) ==="
      exit 0
    fi
    echo "  sha mismatch, retrying"
  fi
  # backoff that grows but caps at 10 min; slow = network throttled
  if [ "$after" -gt "$before" ]; then sleep 10; else sleep 120; fi
done
echo "=== gave up after 40 attempts  $(date) ==="
exit 1
