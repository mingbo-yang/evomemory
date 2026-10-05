# EvoMemory：AAAI2027 检索经验与答案迭代实验

本仓库当前内容来自 `aaai2027/exp/`，包括经验检索、答案修改、判别模型训练与评估、延迟反馈入库，以及 TTS baseline。`baseline/` 是实验代码实际导入的同级依赖，必须与 `exp/` 一起保留。

此前上传的 EvoScope agent 记忆条件自进化代码属于另一套项目，已从当前文件树移出，仍可在 Git 历史中查看。

## 代码与判别器入口

| 路径 | 用途 |
| --- | --- |
| [exp/laya_acceptance_common.py](exp/laya_acceptance_common.py) | Laya multilingual 权重加载、输入构造、Accept/Reject 标签、输入隔离与两选项推理 |
| [exp/laya_relaxed_flow.py](exp/laya_relaxed_flow.py) | `Verifier` 本地推理接口、候选生成、答案采纳与独立的延迟反馈入库 |
| [exp/laya_relaxed_policy.py](exp/laya_relaxed_policy.py) | 上述流程实际调用的候选选择策略 |
| [exp/train_laya_acceptance.py](exp/train_laya_acceptance.py) | 从基础 Laya multilingual 训练二分类模型 |
| [exp/train_label_comparison.py](exp/train_label_comparison.py) | 语义标签与 BLEU 标签的训练对比 |
| [exp/evaluate_label_comparison.py](exp/evaluate_label_comparison.py) | 两个标签版本的评估 |
| [exp/calibrate_laya_safe_acceptance.py](exp/calibrate_laya_safe_acceptance.py) | 历史三分类与阈值校准脚本，不用于当前主流程 |
| [exp/acceptance_data.py](exp/acceptance_data.py) / [exp/prepare_acceptance_training.py](exp/prepare_acceptance_training.py) | 候选采集与训练数据准备 |
| [exp/core/](exp/core/) | 经验结构、BM25/混合检索、控制器、反馈和实验流水线 |
| [exp/run_experiment.py](exp/run_experiment.py) | 原有实验及消融入口；不等同于 Laya 专用流程 |
| [baseline/](baseline/) | 生成器、任务适配、指标与 TTS baseline 源码 |
| [exp/vendor/laya/](exp/vendor/laya/) | 固定版本的第三方 Laya 实现及其许可证 |
| [exp/configs/](exp/configs/) / [exp/tests/](exp/tests/) / [exp/reports/](exp/reports/) | 配置、回归测试与历史方法/诊断文档 |

项目实际微调和调用的判别模型名为 `laya-multilingual`。`Verifier` 在 Python 进程内加载权重并推理，并未部署一个独立的 JEV HTTP 接口。第三方 Laya 目录中出现的 JEV 名称可能属于上游对比研究，不代表本项目使用了该模型。

## 当前采集与正式执行（2026-10-05）

正式推理每轮只生成一个候选，最多修改三轮；Reject 保留 Current 并停止修改。四候选生成只用于独立的数据采集和候选数消融，详见 [exp/README.md](exp/README.md)。

新的训练数据按四个任务独立准备，由五个已部署的 vLLM 模型共用隔离后的 Source pool。每个 Source 先生成一次不带检索的 Initial Current，再用一次冻结检索生成四个不同 seed 的候选；候选不会成为新的 Current。Laya 输入严格只有 Source、Current、Candidate。翻译使用 COMET，GEC 使用 sentence GLEU，Gigaword 使用带 stemming 的 ROUGE-1 F1；正向差值标 Accept，其他有效修改标 Reject。

| 路径 | 用途 |
| --- | --- |
| [exp/multigen_data/](exp/multigen_data/) | Source 隔离、冻结协议、五模型采集、精确去重、完整 provenance、标注和 Test 封存 |
| [exp/run_multigen_pipeline.py](exp/run_multigen_pipeline.py) / [exp/multigen_streaming.py](exp/multigen_streaming.py) | 跨任务生成队列、CPU 增量清洗、断点恢复与导出前完整性检查 |
| [exp/monitor_multigen_completion.py](exp/monitor_multigen_completion.py) | 定时检查；满足全部完成条件后，仅释放身份核实的实验模型服务 |
| [exp/multigen_evaluation.py](exp/multigen_evaluation.py) | 评估锁校验与按 Source 聚类的配对 bootstrap；采集阶段不调用 |
| [exp/deployment/](exp/deployment/) | 五模型 vLLM 服务配置、管理与调用示例 |
| [exp/compare_laya_epoch_checkpoints.py](exp/compare_laya_epoch_checkpoints.py) | 历史逐轮 checkpoint 对比；训练器支持逐轮保存及续训状态记录 |

Test 在模型、损失/类别权重、Dev 选模规则和评估协议冻结前保持 sealed。本次同步的是当前实现，不代表四任务新数据已全部准备完成或新一轮 Laya 训练、主实验已完成。运行配置仍使用本机路径；已准备采集任务的调度 amendment 与运行产物不随源码发布。

## 当前二分类方案

当前协议见 [exp/reports/LAYA_BINARY_PROTOCOL.md](exp/reports/LAYA_BINARY_PROTOCOL.md)。Laya 只输入 Source、Current、Candidate，直接输出 Accept/Reject；不使用 `p_better`、`p_worse_max` 或其他运行时接受阈值。

- Current 与 Candidate 精确相同：程序直接 Reject，不调用 Laya；样本保留审计，但排除出主训练与分类评估。
- 文本不同：用离线 COMET 差值构造标签，Δ > 0 为 Accept，Δ ≤ 0 为 Reject。发生改写但分数相同的候选仍保留为 Reject。
- 正式执行按原始两选项分类采用或拒绝候选；概率只用于诊断。参考答案、反馈分数和检索经验不进入 Laya 输入。
- 经验入库独立于 Acceptance。任务结束后，所有有效实际修改均可获得反馈；ΔCOMET > 0 的候选才有入库资格，包括被 Laya 拒绝的候选。
- 旧三分类实现隔离在 `exp/legacy_laya_v1/`；旧权重不能直接作为二分类检查点加载。

训练入口为 `exp/train_laya_acceptance.py`，分类评估入口为 `exp/evaluate_laya_acceptance.py`，标签导出入口为 `exp/prepare_acceptance_training.py`。`run_laya_acceptance_job.py` 顺序执行训练和开发集评估。新模型保存在本机 `model/laya-multilingual-accept-reject-v1/`，不上传模型权重或训练数据。

2026-09-30 已完成 GPU 3 的真实 COMET 重标注、4 轮训练与冻结后测试，按开发集 macro F1 选中第 3 轮检查点。449 条实际修改上的测试准确率为 55.01%（基础 Laya 44.77%），但尚未证实稳定的质量收益。详见 [训练与测试报告](exp/reports/LAYA_BINARY_TRAINING_V1.md)。该历史四轮训练报告没有包含 full_online 端到端实验；后续入口与当前正式协议以 exp/README.md 为准。

## 环境与外部资源

保留以下目录关系：

```text
repository/
├── exp/
│   ├── core/
│   ├── tests/
│   ├── configs/
│   ├── reports/
│   └── vendor/laya/
└── baseline/
    ├── core/
    └── methods/
```

`exp/core/__init__.py` 将 `baseline/core` 加载为 `baseline_core`，仅复制 `exp/` 会缺少依赖。一般从 `exp/` 目录运行对应脚本。

[ENVIRONMENT_SNAPSHOT.md](ENVIRONMENT_SNAPSHOT.md) 记录本次检查使用的现有环境版本。源码仍保留服务器上的模型、数据、结果目录和 GPU 选择设置；这是研究代码快照，尚未完成通用路径配置。迁移时检查 `baseline/core/config.py`、`exp/core/manifest.py`、`exp/laya_acceptance_common.py`、`exp/semantic_label_common.py` 及所用入口脚本。

权重、数据集、经验库、训练样本、运行结果、日志、生成的图表、虚拟环境及凭据不在本仓库。COMET 复评使用独立环境；绘图脚本另需 matplotlib。原有文档可能引用未发布的本机产物，链接不表示这些资源已随源码上传。

## 验证与来源

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 记录复制自本地的文件摘要及 Laya 固定版本；发布说明与验证记录单独维护。上游 Laya 使用 [Apache-2.0](exp/vendor/laya/LICENSE)，该许可不自动覆盖本仓库其他代码。

发布时的离线验证范围见 [UPLOAD_VALIDATION.md](UPLOAD_VALIDATION.md)。部分 `tests/` 文件是需要本机数据和历史运行产物的独立审计脚本，不应将未运行的审计计为测试通过。
