#!/usr/bin/env bash
# Pre-read a checkpoint so vLLM's load is not bottlenecked on cold NFS reads.
# glm4-9b shards read at ~25 MB/s cold vs ~4-5 GB/s cached; at 4.9 GB/shard that
# is ~3.3 min per shard, which made a 10-minute load look like a hang and got it
# killed by the watchdog.
D="$1"
for f in "$D"/*.safetensors; do
  ( cat "$f" > /dev/null 2>&1 ) &
done
wait
echo "cache warmed for $D at $(date)"
