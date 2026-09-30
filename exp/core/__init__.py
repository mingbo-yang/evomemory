"""Experience-driven refinement experiment package (AAAI2027).

All modules here are additive: they import from ``aaai2027/baseline`` but never
modify it.  See ``reports/plan_v5.md`` for the design.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# --- Paths -----------------------------------------------------------------
EXP_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = EXP_ROOT.parent / "baseline"

# The baseline package is also called ``core``, which collides with this one.
# Register it under the alias ``baseline_core`` so submodule imports resolve
# (baseline/core/*.py only use relative imports, so this is safe) while
# ``aaai2027/baseline`` itself stays untouched and read-only.
BASELINE_PKG = "baseline_core"
if BASELINE_PKG not in sys.modules:
    _pkg_dir = BASELINE_ROOT / "core"
    _spec = importlib.util.spec_from_file_location(
        BASELINE_PKG,
        _pkg_dir / "__init__.py",
        submodule_search_locations=[str(_pkg_dir)],
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load baseline core package from {_pkg_dir}")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[BASELINE_PKG] = _mod
    _spec.loader.exec_module(_mod)

# --- Hard constraints from the plan ----------------------------------------
ALLOWED_GPUS = ("0", "1", "2", "3")  # all four granted (2026-09-12); GPU 1 is shared
PINNED_VERSIONS = {
    "vllm": "0.17.0",
    "transformers": "4.57.6",
    "torch": "2.10.0+cu128",
}

# Exactly the values the baseline scheduler used
# (run_full_baseline_scheduler.py, --gpu-memory-utilization defaults).  They are
# per model because Qwen3-32B needs ~0.86 of an 80 GB card just for its weights;
# reusing the baseline values keeps the KV-cache budget identical and is
# required for the 32B model to load at all.
# The baseline scheduler always passed --vllm-max-model-len 4096
# (verified in results_full_main_qwen35_vllm/logs/*__attempt0.log).  My first
# version of run_experiment.py/build_experience.py used 8192, which is a silent
# deviation from the baseline configuration and doubles the KV-cache profiling
# cost.  All new code must take this value from here.
MAX_MODEL_LEN = 4096

GPU_MEMORY_UTILIZATION = {
    "glm4-9b": 0.45,
    "llama3.1-8b": 0.40,
    "qwen3-4b": 0.30,
    "qwen3-8b": 0.40,
    "qwen3-32b": 0.86,
}


# Model path overrides (exp side only -- aaai2027/baseline stays read-only).
#
# glm4-9b's NFS copy is unreadable: model-00002-of-00004.safetensors cannot be
# served by 172.25.76.194 past byte 2,769,682,432 (see
# reports/glm4_nfs_root_cause.md).  A local copy was assembled and verified
# against the shipped manifest_sha256.json -- all four shard sha256 match --
# so the experiment is pointed at that instead.  The MODEL, dtype, tokenizer and
# engine version are unchanged; only the directory differs.
MODEL_PATH_OVERRIDES = {
    "glm4-9b": "/home/ymb/glm_local/model",
}


def resolve_model_config(model: str):
    """Return the *baseline* ModelConfig, with any verified path override applied."""
    from baseline_core.config import MODEL_CONFIGS
    from baseline_core.types import ModelConfig

    cfg = MODEL_CONFIGS[model]
    new_path = MODEL_PATH_OVERRIDES.get(model)
    if not new_path or new_path == cfg.path:
        return cfg
    import os

    if not os.path.isdir(new_path):
        raise FileNotFoundError(
            f"override path for {model} does not exist: {new_path}; "
            "refusing to silently fall back to the unreadable NFS copy"
        )
    return ModelConfig(
        key=cfg.key, path=new_path, param_count=cfg.param_count,
        tensor_parallel_size=cfg.tensor_parallel_size, max_model_len=cfg.max_model_len,
    )


def gpu_mem_util(model: str, override: float | None = None) -> float:
    if override is not None:
        return override
    if model not in GPU_MEMORY_UTILIZATION:
        raise KeyError(f"no GPU memory utilization registered for model {model!r}")
    return GPU_MEMORY_UTILIZATION[model]

TASKS = ("wmt19_en_zh", "wmt19_zh_en", "coedit_gec", "gigaword")
MODELS = ("glm4-9b", "llama3.1-8b", "qwen3-4b", "qwen3-8b", "qwen3-32b")

# Test sizes must match the existing baseline exactly (first-N prefix).
TEST_SAMPLES = {
    "wmt19_en_zh": 1000,
    "wmt19_zh_en": 1000,
    "coedit_gec": 1000,
    "gigaword": 100,
}

# Auxiliary split sizes (per model-task).
AUX_SIZES = {"initial": 32, "dev": 32, "accumulation": 128}

# Gigaword's only clean source is gigaword_tiny validation+test (200 rows), of
# which 2 are duplicates, leaving 198 unique.  The guard in build_aux_splits
# refuses to silently shrink, so Gigaword uses the same 32/32/128 profile as
# every other task and keeps 6 rows of headroom.
AUX_SIZES_GIGAWORD = {"initial": 32, "dev": 32, "accumulation": 128}


def aux_sizes(task: str) -> dict:
    return AUX_SIZES_GIGAWORD if task == "gigaword" else AUX_SIZES


def baseline_results_dir() -> Path:
    return BASELINE_ROOT / "results_full_main_qwen35_vllm"


MAX_GPUS = 4  # updated 2026-09-12: user granted GPUs 1 and 2 as well


def ensure_gpu_whitelist() -> str:
    """Return the CUDA_VISIBLE_DEVICES value, enforcing the granted GPU budget.

    Policy (updated 2026-09-12): all four of GPUs 0-3 are authorised for this run.
    The operator widened the budget step by step (2,3 -> any two -> +2 -> +1);
    see configs/frozen.json gpu_policy for the history and the exact wording.
    The card *ids* are unrestricted; the *count* is capped at ``MAX_GPUS``.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return ",".join(ALLOWED_GPUS[:MAX_GPUS])
    requested = [x.strip() for x in cvd.split(",") if x.strip()]
    illegal = [g for g in requested if g not in ALLOWED_GPUS]
    if illegal:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cvd!r} contains GPU(s) {illegal} outside {ALLOWED_GPUS}."
        )
    if len(requested) > MAX_GPUS:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cvd!r} requests {len(requested)} GPUs; the granted "
            f"budget is at most {MAX_GPUS} cards at any time."
        )
    return cvd
