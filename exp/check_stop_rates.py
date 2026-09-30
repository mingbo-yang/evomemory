#!/usr/bin/env python
"""Sanity check: does experience actually change the controller's decisions?

After neutralising the controller guidance we must confirm the arms are no
longer degenerate.  If ``full`` and ``none`` stop at the same rate, the
experience has no room to act and the How/When comparison would be vacuous.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
from core import MAX_MODEL_LEN, gpu_mem_util
from core.manifest import read_manifest, manifest_path
from core.experience import load_experiences, experience_dir
from core.bm25_fields import ExperienceRetriever
from core.pipeline import Pipeline, RunConfig
from baseline_core.config import MODEL_CONFIGS
from baseline_core.llm import LLMClient

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
TASK = sys.argv[2] if len(sys.argv) > 2 else "wmt19_en_zh"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40

refs = read_manifest(manifest_path(TASK, os.environ.get('SPLIT','dev')))[:N]
lib = load_experiences(experience_dir(MODEL, TASK) / "initial.jsonl")
llm = LLMClient(MODEL_CONFIGS[MODEL], backend="vllm", gpu="3",
                gpu_memory_utilization=gpu_mem_util(MODEL), max_model_len=MAX_MODEL_LEN,  # was hardcoded 8192; aligned to the frozen 4096
                enforce_eager=True)
out = {}
for mode in ["none", "full", "outcome_hidden"]:
    cfg = RunConfig(task=TASK, model=MODEL, experience_mode=mode, stop_mode="adaptive",
                    alpha=0.5, seed=42, snapshot_id="initial")
    L = lib if mode != "none" else []
    pipe = Pipeline(cfg, ExperienceRetriever(L) if L else None, L, llm)
    stop = refine = acc = 0
    dq = 0.0
    for i, r in enumerate(refs):
        t = pipe.run_sample(r, i)
        dq += (t.final_metric_offline or 0) - (t.initial_metric_offline or 0)
        for x in t.rounds:
            if x.controller_action == "STOP":
                stop += 1
            else:
                refine += 1
                if x.accepted:
                    acc += 1
    out[mode] = {"stop": stop, "refine": refine, "accept": acc,
                 "stop_rate": stop / max(1, stop + refine), "mean_dQ": dq / len(refs)}
    print(f"{mode:16s} STOP={stop:4d} REFINE={refine:4d} accept={acc:3d} "
          f"stop_rate={out[mode]['stop_rate']:.1%} mean_dQ={out[mode]['mean_dQ']:+.2f}", flush=True)
json.dump(out, open(f"reports/stop_rate_check_{MODEL}_{TASK}.json", "w"), indent=2)
