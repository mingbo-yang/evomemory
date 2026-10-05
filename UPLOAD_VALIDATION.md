# Source synchronization — 2026-10-05

Synchronized the current `aaai2027/exp` implementation and its sibling baseline
source dependency. This update includes formal single-candidate acceptance,
checkpoint retention and continuation, fixed-current five-generator data
collection, sealed-Test enforcement, source-clustered evaluation, concurrent
collection/cleaning and completion monitoring. Runtime artifacts and model weights
remain on the experiment host.

Validation was run in this independent publication checkout:

- 93 selected offline regression tests passed in 10.72 seconds.
- All 480 source-manifest entries match both the local source and publication copy
  by SHA-256; 291 Python files passed syntax parsing.
- The full staged whitespace check reports an existing extra final blank line in
  `exp/ablations/laya_candidate_count.py` and `exp/core/candidate_validation.py`.
  Both are preserved byte-for-byte from the experiment source; this upload does
  not reformat frozen runtime code. Publication metadata/documents pass their
  separate whitespace check.
- No matches were found for the checked common credential-token/private-key
  patterns in the source manifest.
- No real GPU inference, checkpoint training or Test quality evaluation was run
  for this upload. These checks validate software behavior, not model efficacy.

The selected regression command, from `exp/`, was:

```bash
PYTHONPATH=/mnt/huawei/ymb/.tmp/laya_multigen_test_deps:$PWD \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/ymb/miniconda3/envs/qwen35/bin/python -m pytest -q \
  -p no:cacheprovider --basetemp=/tmp/aaai-github-20261005-tests --tb=short \
  tests/test_multigen_data.py tests/test_multigen_async.py \
  tests/test_multigen_pipeline.py tests/test_multigen_completion_monitor.py \
  tests/test_multigen_evaluation.py tests/test_laya_acceptance.py \
  tests/test_laya_relaxed_flow.py tests/test_candidate_count_ablation.py \
  tests/test_prepare_acceptance_training.py tests/test_laya_checkpoint_retention.py \
  tests/test_laya_continuation.py tests/test_laya_epoch_comparison.py \
  tests/test_acceptance_data.py tests/test_optimized_feedback.py \
  tests/test_comet_feedback.py
```

The temporary dependency directory supplies pytest for the existing environment.
A fresh checkout still needs the documented Python dependencies and local external
resources for data-dependent audits. The historical validation records below refer
to the earlier snapshots, not the current protocol.

---

# Binary acceptance update — 2026-09-30

Current Laya task: Accept/Reject with raw two-option classification. Exact no-op candidates bypass Laya and are excluded from primary training/evaluation; changed candidates with equal COMET remain Reject. Both offline tolerances are zero. Prior three-class experiments are isolated under `exp/legacy_laya_v1/`.

Validation: 60 tests and 14 subtests passed in the local experiment checkout, including reference isolation, no-op bypass, delayed memory admission and preservation of COMET virtual-environment/checkpoint symlinks. Static checks and patch whitespace checks passed. Source checksums match the synchronized local files; no weight, training-data or credential artifacts are included.

Real GPU 3 labeling, four training epochs, frozen-test evaluation and base-model comparison are complete; see `exp/reports/LAYA_BINARY_TRAINING_V1.md`. Test accuracy increased from 44.77% (base Laya) to 55.01%, but stable quality gain is not demonstrated. No new-model full_online rollout was run. Runtime output and model weights stay on the experiment server.

---

# 源码发布验证（2026-09-30）

本次仅整理并上传 `aaai2027/exp` 及其代码依赖；未修改实验算法，未训练或调用模型，未启动 GPU 服务。所有检查在独立发布副本中进行。

## 离线回归

共选择 60 项 pytest/unittest 测试：30 项基础、数据隔离、BERT/Laya 策略与经验构造检查，以及 30 项人工评测导出检查。

- 不提供实验数据清单时：52 项通过，8 项人工评测导出测试因缺少 `exp/data/manifests/wmt19_en_zh__test.jsonl` 失败。
- 临时接入已有本地 `exp/data` 后，人工评测导出的全部 30 项通过；临时链接随后移除，数据未加入发布副本。
- 因而所选 60 项测试均已在所需资源存在时通过；这不表示纯源码克隆后无需数据即可通过全部检查。

不依赖上述数据清单的 30 项检查：

```bash
cd exp
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/test_acceptance_data.py \
  tests/test_bert_feedback_study.py \
  tests/test_experience_transitions.py \
  tests/test_laya_acceptance.py \
  tests/test_laya_relaxed_flow.py \
  tests/test_laya_safe_acceptance.py \
  tests/test_onpolicy_bert_study.py \
  tests/test_optimized_feedback.py \
  tests/test_prepare_acceptance_training.py \
  tests/test_accept_gate.py::test_draft_source_rule_is_not_duplicated
```

人工评测导出检查另运行 `python -m pytest -q tests/test_human_eval_export.py`，需要已准备的数据清单。

一项常量数组的 Spearman 相关性检查产生 `ConstantInputWarning`，不影响测试通过。历史 `main()` 审计脚本、数据相关全量检查、模型效果评测和第三方模型推理测试不在此次验证范围内。

## 编译与第三方静态检查

- `python -m compileall -q exp baseline`：通过；字节码写到发布目录以外。
- 在 `exp/vendor/laya` 执行 `ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120 --no-cache`：通过。
- 复制文件与 `SOURCE_MANIFEST.json` 中的 SHA-256 对照；本地原实验源码保持不变。
- 发布文件扫描未发现常见 API token、私钥或带密码的 URL。

测试验证的是现有代码的相应行为。特别是旧阈值策略的测试通过，不代表它符合新的“分类 Better 才采纳”方案；这一未修复状态已在 README 中标注。

## 保留的源文件格式

完整快照的 `git diff --check` 报告了 baseline 与第三方 Laya 等原文件中的行尾空白/文件末空行。本次保留原始内容与摘要，没有为上传批量重排或格式化代码。新增发布文档的差异检查单独通过。
