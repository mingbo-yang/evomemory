#!/usr/bin/env bash
# Rebuild ONLY the four model-tasks whose positive pool fell short, at a K
# sized from their measured positive rate (plan section 4.4).
set -u
cd "$(dirname "$0")"
PY=/home/ymb/miniconda3/envs/qwen35/bin/python
GPU="${GPU:-3}"
KMAP="coedit_gec=8,glm4-9b:wmt19_en_zh=2,qwen3-4b:wmt19_en_zh=2,qwen3-8b:wmt19_en_zh=2,llama3.1-8b:wmt19_en_zh=5,llama3.1-8b:wmt19_zh_en=4,llama3.1-8b:gigaword=3,qwen3-8b:wmt19_zh_en=3"

echo "########## llama3.1-8b (en_zh K=5, zh_en K=4, gigaword K=3) ##########"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" build_experience.py --gpu "$GPU" \
    --models llama3.1-8b --tasks wmt19_en_zh,wmt19_zh_en,gigaword \
    --interventions 2 --k-map "$KMAP" --min-positive 8

echo "########## qwen3-8b (zh_en K=3) ##########"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" build_experience.py --gpu "$GPU" \
    --models qwen3-8b --tasks wmt19_zh_en \
    --interventions 2 --k-map "$KMAP" --min-positive 8
