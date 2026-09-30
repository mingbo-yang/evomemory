# AAAI2027 TTS Baselines

Unified runner for verifier-poor open-ended generation TTS baselines.

## Methods

Implemented:

- `Direct-Zero`
- `BoN-U-4`
- `SelfRefine-Fixed`
- `SelfRefine-U`
- `SR-Fixed`
- `SR-U`
- `PDR-2-1`
- `ChecklistRefine`
- `ModeX-4`
- `TEaR-Native` and `TEaR-U` for translation only
- `AdaCompute-SR-U` as a training-free adapted budget allocator

Classic ICL baselines are intentionally not reimplemented because they already exist under `/mnt/huawei/ymb/icml/baseline`.

## Tasks And Default Test Sizes

- `wmt19_en_zh`: 3000 examples
- `wmt19_zh_en`: 3000 examples
- `coedit_gec`: 3000 examples
- `gigaword`: 3000 examples

Use `--max-samples` only for debugging. Omit it for paper runs.

## Example Commands

Dry-run expansion:

```bash
python3 /mnt/huawei/ymb/aaai2027/baseline/run_baseline.py \
  --dry-run \
  --task all \
  --model qwen3-4b \
  --method all
```

Run one small smoke test with a real backend:

```bash
python /mnt/huawei/ymb/aaai2027/baseline/run_baseline.py \
  --task wmt19_en_zh \
  --model qwen3-4b \
  --method Direct-Zero \
  --backend vllm \
  --gpu 0 \
  --max-samples 2
```

Run a full default-size job:

```bash
python /mnt/huawei/ymb/aaai2027/baseline/run_baseline.py \
  --task wmt19_en_zh \
  --model qwen3-4b \
  --method SR-U \
  --backend vllm \
  --gpu 0
```

## Outputs

Results are written under `results/<task>/<model>/`.

Each run writes:

- `<method>.csv`: compact per-example table
- `<method>.jsonl`: full trace with candidates, generations, utility scores, prompts, token counts, and per-call latency
- `<method>.summary.json`: aggregate metric, latency, token, and call-count summary

All LLM calls count toward total cost, including discarded parallel candidates, feedback, checklist evaluation, MQM estimate, distillation, and refinement.

