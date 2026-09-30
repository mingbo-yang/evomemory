#!/usr/bin/env bash
# Safely free ONE gpu by killing only OUR processes that hold memory ON THAT GPU.
#
# Why this exists: a previous ad-hoc cleanup loop checked only
#   user == ymb  &&  gpu_memory > 10 GiB
# WITHOUT restricting the card, so it killed our production engines on GPU 0 and
# GPU 2 while trying to clear GPU 3.  Always filter by GPU uuid -> index.
#
# Usage: ./free_gpu.sh <gpu_index> [min_mib]
set -u
GPU="${1:?usage: free_gpu.sh <gpu_index> [min_mib]}"
MIN="${2:-8000}"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' ')
[ -z "$UUID" ] && { echo "no such gpu: $GPU"; exit 2; }
killed=0
while IFS=, read -r uu pid mem; do
  uu=$(echo "$uu" | tr -d ' '); pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' MiB')
  [ "$uu" != "$UUID" ] && continue
  [ "$(ps -o user= -p "$pid" 2>/dev/null)" != "ymb" ] && continue
  [ "${mem:-0}" -lt "$MIN" ] && continue
  kill -9 "$pid" 2>/dev/null && echo "killed pid=$pid (gpu$GPU, ${mem}MiB)" && killed=$((killed+1))
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)
echo "freed $killed process(es) on gpu$GPU"
