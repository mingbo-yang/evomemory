# EvoMemory：AAAI2027 检索经验与答案迭代实验

本仓库当前内容来自 `aaai2027/exp/`，包括经验检索、答案修改、判别模型训练与评估、延迟反馈入库，以及 TTS baseline。`baseline/` 是实验代码实际导入的同级依赖，必须与 `exp/` 一起保留。

此前上传的 EvoScope agent 记忆条件自进化代码属于另一套项目，已从当前文件树移出，仍可在 Git 历史中查看。

## 代码与判别器入口

| 路径 | 用途 |
| --- | --- |
| [exp/laya_acceptance_common.py](exp/laya_acceptance_common.py) | Laya multilingual 权重加载、输入构造、Better/Tie/Worse 标签与推理工具 |
| [exp/laya_relaxed_flow.py](exp/laya_relaxed_flow.py) | `Verifier` 本地推理接口、候选生成、答案采纳与独立的延迟反馈入库 |
| [exp/laya_relaxed_policy.py](exp/laya_relaxed_policy.py) | 上述流程实际调用的候选选择策略 |
| [exp/train_laya_acceptance.py](exp/train_laya_acceptance.py) | Laya acceptance 训练入口 |
| [exp/train_label_comparison.py](exp/train_label_comparison.py) | 语义标签与 BLEU 标签的训练对比 |
| [exp/evaluate_label_comparison.py](exp/evaluate_label_comparison.py) | 两个标签版本的评估 |
| [exp/calibrate_laya_safe_acceptance.py](exp/calibrate_laya_safe_acceptance.py) | 独立的 Better 分类与分差校准检查 |
| [exp/acceptance_data.py](exp/acceptance_data.py) / [exp/prepare_acceptance_training.py](exp/prepare_acceptance_training.py) | 候选采集与训练数据准备 |
| [exp/core/](exp/core/) | 经验结构、BM25/混合检索、控制器、反馈和实验流水线 |
| [exp/run_experiment.py](exp/run_experiment.py) | 原有实验及消融入口；不等同于 Laya 专用流程 |
| [baseline/](baseline/) | 生成器、任务适配、指标与 TTS baseline 源码 |
| [exp/vendor/laya/](exp/vendor/laya/) | 固定版本的第三方 Laya 实现及其许可证 |
| [exp/configs/](exp/configs/) / [exp/tests/](exp/tests/) / [exp/reports/](exp/reports/) | 配置、回归测试与历史方法/诊断文档 |

项目实际微调和调用的判别模型名为 `laya-multilingual`。`Verifier` 在 Python 进程内加载权重并推理，并未部署一个独立的 JEV HTTP 接口。第三方 Laya 目录中出现的 JEV 名称可能属于上游对比研究，不代表本项目使用了该模型。

## 此次快照的准确状态

本次发布保留本地已有实验逻辑，没有借上传修改算法或重新运行模型实验。

- `laya_relaxed_flow.py` 仍调用旧的概率阈值策略；之前运行采用 `P(Better) >= 0.275` 且 `P(Worse) <= 0.4`。它没有强制最高概率类别为 Better，可能接受 Tie/Worse 候选，这是已定位的问题。
- `calibrate_laya_safe_acceptance.py` 已包含 Better 必须为最高概率类别的约束及可选分差，但尚未接入上述完整流程。
- “直接分类为 Better 才替换当前答案”的完整流程修复及重新验证，仍待单独实施。测试通过不代表这项修复已完成，也不代表方法效果已获证明。
- 答案采纳与经验入库是不同决策。`positive_memories` 对实际修改前后的答案计算事后反馈，不要求候选先被 verifier 接受；能够评价所有候选的实验环境是这一做法的前提。
- 文档与脚本涉及不同历史版本，应分别解释结果，不能把旧 BERT、旧阈值和新分类规则的结果合并。

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
