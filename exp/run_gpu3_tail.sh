#!/usr/bin/env bash
# GPU 3 tail chain: wait for the in-flight qwen3-4b job, then give glm4-9b its
# best remaining chance on the card that has successfully loaded two models.
# GPU 2 was released rather than left occupied by a card that cannot complete
# this model's load.
set -u
cd "$(dirname "$0")"
echo "=== waiting for in-flight GPU3 job ($(date)) ==="
# only wait for the GPU-3 job, not for GPU 0's independent chain
while pgrep -f "run_experiment.py --arm full_static --gpu 3" >/dev/null 2>&1; do sleep 60; done
echo "=== in-flight done ($(date)); starting glm4-9b on GPU 3 ==="
exec /home/ymb/miniconda3/envs/qwen35/bin/python supervisor.py \
    --gpu 3 --models glm4-9b --stuck-min 15 --max-attempts 4
