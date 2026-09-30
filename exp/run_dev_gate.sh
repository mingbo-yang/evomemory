#!/usr/bin/env bash
# Phase 1g: alpha selection + BM25-vs-random dev gate.
#
# Three fixes learned the hard way:
#  1. dev items have no stored draft (the stored lookup is positional over the
#     test prefix), so dev runs must not use --draft-source stored;
#  2. every sweep point MUST get its own --tag, else the runs overwrite each
#     other and only the last value survives;
#  3. the initial draft MUST be frozen across sweep points.  vLLM is not
#     bit-deterministic at T=0.1, so regenerating per point changed the starting
#     condition (corpus BLEU on the same 16 dev items: 32.51 vs 38.21).
#     Drafts are therefore generated once into a cache and replayed.
#
# Alpha governs which experiences are retrieved, so it is measured on the
# forced-refinement arm where retrieval affects every round.
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-2}"
MODELS="${MODELS:-glm4-9b}"
TASKS="${TASKS:-all}"
LIMIT="${LIMIT:-16}"
CACHE="data/draft_cache/dev_${MODELS//,/_}.jsonl"
mkdir -p data/draft_cache

# warm the draft cache once with a no-retrieval pass over the same dev items
if [ ! -s "$CACHE" ]; then
  echo "########## warming draft cache: $CACHE ##########"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_experiment.py \
      --arm no_experience --gpu "$GPU" --models "$MODELS" --tasks "$TASKS" \
      --split dev --limit "$LIMIT" --max-rounds 0 \
      --draft-source cached --draft-cache "$CACHE" --tag warmup \
      --out-dir runs/gate_warmup 2>&1 | grep -aE "^\[LOAD\]|^\[RUN\]|final=|Error|Traceback"
fi

for a in 0.0 0.5 1.0; do
  echo "########## fixed_rounds alpha=$a ##########"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_experiment.py \
      --arm fixed_rounds --gpu "$GPU" --models "$MODELS" --tasks "$TASKS" \
      --split dev --limit "$LIMIT" --alpha "$a" --max-rounds 3 \
      --draft-source cached --draft-cache "$CACHE" --tag "a$a" \
      --out-dir runs/gate_alpha 2>&1 | grep -aE "^\[LOAD\]|^\[RUN\]|final=|Error|Traceback"
done
echo "########## random-retrieval control ##########"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_experiment.py \
    --arm random_retrieve --gpu "$GPU" --models "$MODELS" --tasks "$TASKS" \
    --split dev --limit "$LIMIT" --alpha 0.5 --max-rounds 3 \
    --draft-source cached --draft-cache "$CACHE" --tag rand \
    --out-dir runs/gate_alpha 2>&1 | grep -aE "^\[LOAD\]|^\[RUN\]|final=|Error|Traceback"
echo "########## dev gate done ##########"
