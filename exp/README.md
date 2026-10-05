# EvoMemory：检索经验与答案修改实验

正式流程每轮只生成 **1 个候选**，最多修改 3 轮；Reject 保留当前答案并停止该任务修改。多候选只在独立消融入口运行，正式入口拒绝候选数大于 1。

当前 Laya 验收任务为 **Accept / Reject 二分类**。模型输入只有 Source、Current Answer、Candidate Answer，直接按分类结果决定是否替换当前答案。完全相同的候选由程序直接 Reject；主训练和分类评估仅包含文本不同的候选，其中 COMET 持平或降低仍标为 Reject。no-op 保留在原始记录与独立审计统计中。当前协议与使用步骤见 [reports/LAYA_BINARY_PROTOCOL.md](reports/LAYA_BINARY_PROTOCOL.md)。

| 入口 | 用途 |
| --- | --- |
| `configs/laya_binary_v1.json` | 新协议；离线标签 ε=0、入库 ε_memory=0 |
| `prepare_acceptance_training.py` | 将 no-op 分离到审计文件，对实际改写重算 COMET 二分类标签 |
| `laya_acceptance_common.py` | 两选项编码、输入隔离、分类、检查点 schema |
| `train_laya_acceptance.py` | 从 Laya multilingual 基础权重训练二分类模型；逐轮保存权重，并按开发集选择 best |
| `compare_laya_epoch_checkpoints.py` | 对所有已冻结轮次进行并行测试诊断，与原模型比较；不改变正式开发集选模 |
| `evaluate_laya_acceptance.py` | 直接分类评估；概率温度仅影响诊断指标 |
| `laya_relaxed_flow.py` | 当前正式二分类流程，包含 `Verifier.predict`、`full_static` 与 `full_online` |
| `laya_relaxed_policy.py` | 唯一候选被分类为 Accept 才替换；没有概率接受阈值或多候选排名 |
| `run_candidate_count_ablation.py` / `ablations/laya_candidate_count.py` | 独立的 1/4 候选消融；正式入口不导入此模块 |
| `core/candidate_validation.py` / `core/refinement_instructions.py` | 无采集任务或旧 BERT 流程副作用的公共规则与修改指令 |
| `verify_single_candidate_replay.py` | 用已有真实轨迹检验正式单候选流程的行为一致性，不是新模型效果评估 |
| `core/comet_feedback.py` | 独立的离线/任务结束后 COMET 反馈接口 |
| `legacy_laya_v1/` | 旧三分类实现，仅供历史实验与报告使用 |

`laya_relaxed_flow.py` 保留原文件名以维持入口位置，但运行产物已改为 `runs/laya_binary_flow_v1`。新模型保存到 `/mnt/huawei/ymb/model/laya-multilingual-accept-reject-v1`。旧三分类检查点不可作为新模型直接加载。新数据按 COMET 重标注后从基础 multilingual 权重训练，测试集仅在检查点冻结后开启。

正式运行必须显式传入 `--checkpoint`，不再自动加载早期 `best/`；只开放 `full_static`、`full_online`。`--candidates-per-round` 仅接受 `1`，省略时也是 `1`。检索最多 4 条经验仍保留，经验条数与生成候选数是两个不同参数。

现有 BERT、语义标签比较、旧阈值分析脚本属于历史实验，不是当前二分类主流程。当前不要求进行概率阈值搜索，也没有“必须达到 80% 才运行”的条件。

代码依赖同级 `../baseline/`，以及 `vendor/laya/` 中的第三方实现。2026-09-30 已完成 GPU 3 的 COMET 重标注、4 轮二分类训练与冻结后测试，并同步本仓库代码。[实验结果](reports/LAYA_BINARY_TRAINING_V1.md)：测试准确率 55.01%，优于基础 Laya 的 44.77%，但尚未证实稳定的质量收益；该次初始训练报告未包含 full_online。2026-10-04 已用 epoch20 完成 128 条任务的候选数端到端消融，随后将正式流程固定为单候选；结果见 [消融记录](runs/laya_candidate_ablation_v1/summary_zh.txt)。这不代表完整主实验已完成。


## Fixed-current multigenerator Laya data (2026-10-04)

The new collector is `multigen_data`, separate from historical two-state collection.
It uses the five existing HTTP vLLM services and four independent task datasets.
Each source/model generates one retrieval-free initial answer, then four independently
seeded revisions of that same answer using one frozen retrieval context. Revisions
never become new currents. New Laya examples serialize only Source, Current, Candidate
with `input_template_version=source-current-candidate-only-v2`; the historical
checkpoint serializer is retained for historical checkpoints only.

Use the existing qwen35 Python environment from the project directory:

```bash
python -m multigen_data.cli prepare
python -m multigen_data.cli pilot
python -m multigen_data.cli run
python -m multigen_data.cli status
```

`run` resumes completed requests and pilots, completes the first 5,000 training
sources, and expands in 500-source shards until each task has 100,000 unique valid
changed pairs or exhausts its genuinely isolated source pool. Five generators always
share the same source manifests. Task datasets and their labels stay separate.

Artifacts live in `runs/laya_multigen_data_v1/<dataset>/`: frozen manifests and
references, shared initial memory, immutable request and trajectory records, the
pair/provenance SQLite index, Train/Dev exports, statistics, and timing events.
Cross-generator duplicate pairs have one training row with every occurrence in
`metadata.provenance`; the generator, temperature, slot, seed, request hash, and
original record location are retained. Scores and provenance never become model
features. Translation feedback uses the frozen local COMET checkpoint on GPU 0;
GEC uses NLTK sentence GLEU and Gigaword uses stemmed ROUGE-1 F1. Epsilon is zero.

Test is SEALED. The current campaign only creates its trajectories and audits file
integrity, request success, and output syntax. There are no Test quality scores,
labels, no-op/changed/cleaned statistics, or Laya predictions. Cleaning, scoring,
export and evaluation loaders fail before reading Test data unless an immutable
`evaluation_lock.json` freezes actual model weights, Train-derived loss settings,
Dev-only checkpoint selection, and a source-clustered evaluation protocol. The
historical Laya JSONL loader also refuses to read new sealed Test artifacts.
No evaluation lock is created by `prepare`, `pilot`, or `run`.

Timing is logged per dataset/split/generator/shard. Parallel durations use interval
unions; completed requests are deduplicated when resuming. Test timing reports only
permitted generation counts. Class weighting is intentionally deferred until the
new Train distribution is available; existing trainer weighting must not be silently
inherited. Formal inference remains one candidate per round.

The read-only contract audit can be reproduced with
`python audit_multigen_collection.py --source-isolation`. It reconstructs actual
pilot prompts from Source and fixed Initial Current, checks all four revision seeds,
resolves every request occurrence back to its immutable raw file, and exercises
completed-shard recovery and sealed-Test rejection. Source isolation checks only
manifest identities, never Test generations or quality labels.

`multigen_evaluation.py` is the later evaluation entry point for frozen,
ID-aligned prediction files. It reports independent Source counts alongside pair
counts, 2,000 source-cluster bootstrap draws, 95% percentile intervals, and paired
model differences using identical draws. A Test evaluation additionally requires
the comparison names, metrics and evaluation implementation hash to be frozen in
the evaluation lock. Data collection does not invoke this entry point.

### Concurrent generation and cleaning

The running, already-prepared campaign uses `python run_multigen_pipeline.py`.
Its frozen scheduling amendment and implementation hashes are in
`runs/laya_multigen_data_v1/scheduling_amendments/cross-task-incremental-v2.json`.
Startup requires that campaign-specific amendment, completed pilots and frozen
protocol artifacts; these runtime files are not included in the source repository.
The `prepare`, `pilot`, `run` CLI commands above remain the original collection
entry points. `run_multigen_async.py` supplies the staged collector and source-limit
logic reused by the current scheduler.

Each model advances independently through already-authorized source shards and
can move to another task while CPU cleaning handles a completed five-model prefix.
`multigen_streaming.py` incrementally processes new completed raw records in
transactions and performs a full raw-record integrity audit before final export.
Future trajectories are staged outside `raw/` and published together before
cleaning. Dispatch uses the hard upper bound of 20 unique pairs per source, so
every ahead-of-time shard is provably required to reach the target; models finish
on the same source prefix. The first 5,000 training sources remain mandatory.

Cleaning, Train scoring and export can overlap generation on other tasks.
Dev/Test generation remains reference-blind; Test still has no scoring, cleaning
or Laya inference during this phase. A model waits if it has exhausted every
currently authorized shard; the scheduler does not overcollect sources merely
to keep a GPU busy. Request caches and timing events survive restarts. Scheduling
changes can change vLLM batching; frozen requests and seeds do not imply
bit-identical uncached generations across schedules.

Current cross-task model activity is in `pipeline_workers/<model>.json` under the
campaign root, with per-task activity in `<dataset>/async_progress/`.
`<dataset>/async_dispatch.json` records the cleaned common prefix and authorized
generation prefix. The active process and log are listed in `process.json`.

`monitor_multigen_completion.py` checks campaign completion and service health at
the configured interval. Automatic GPU release requires all task exports and
sealed-Test integrity checks to finish and the coordinator/scorer to exit. It
signals only the model-service processes whose identities were recorded when the
monitor was armed; unrelated GPU processes are preserved.
