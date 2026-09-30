# Laya 二分类训练与冻结后测试：2026-09-30

已完成真实 COMET 重标注、GPU 3 上的 4 轮训练，以及模型冻结后的候选分类测试。相对基础 Laya，分类准确率和 macro-F1 上升；但当前结果尚未证明稳定的质量收益。

## 固定设置

- 初始化：本地基础 `laya-multilingual`，encoder 为 `jhu-clsp/mmBERT-base`，未沿用旧三分类微调权重。
- 模型输入仅为 Source、Current、Candidate。ε=0，ΔCOMET>0 为 Accept，否则 Reject。
- 精确相同文本排除出主训练/评估并保留审计；文本不同但 COMET 持平仍保留为 Reject。
- RLCD + CE，4 epochs，micro-batch=16，effective-batch=64，seed=42；正式决策为两选项原始分类，无概率阈值。
- 根据开发集 macro-F1、随后 accuracy 选模；选中第 3 轮，全部训练结束后冻结权重，再生成测试标签。没有依据测试结果重新训练或调整规则。
- 训练候选来自冻结 Qwen3-8B 采集，温度混合 0.1/0.7；开发和测试沿用 0.1 候选分布。

## 数据

| 划分 | 实际修改对 | 含实际修改的源句 | Accept | Reject | 排除的有效去重 no-op |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 14018 | 5330 | 6205 | 7813 | 5984 |
| development | 489 | 292 | 231 | 258 | 345 |
| temperature_calibration | 286 | 185 | 105 | 181 | 221 |
| decision_diagnostics | 285 | 190 | 129 | 156 | 221 |
| test | 449 | 295 | 186 | 263 | 344 |

测试清单共 400 个源句；其中 295 个源句产生了保留用于主评估的实际修改。主分类指标只覆盖这 449 对，不把 no-op 的确定性正确拒绝计入模型准确率。源句划分和初始经验库隔离检查通过。

训练集中有 1 对文本不同但 COMET 完全相同，已保留为 Reject；10 对正向差值不超过 0.00001，按 ε=0 保留为 Accept，未在观察测试结果后更改容差。

## 冻结测试对照

| 方法 | 准确率 | macro-F1 | Accept 数 | Accept 精度 | 每候选对净 ΔCOMET |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基础 Laya multilingual | 44.77% | 0.4206 | 360 | 41.39% | -0.00008083 |
| 本次二分类微调 | 55.01% | 0.5083 | 132 | 43.94% | +0.00005160 |
| 始终 Reject | 58.57% | 0.3694 | 0 | — | +0.00000000 |
| 始终 Accept | 41.43% | 0.2929 | 449 | 41.43% | -0.00035578 |

每候选对净 ΔCOMET = Σ[实际分类为 Accept 的候选质量差] / 全部实际修改对数。Reject 贡献为 0。COMET 使用原始分值；该指标不等同于完整多轮在线任务的最终质量变化。

新模型接受 132 对，其中 58 对正向、74 对负向。相对基础模型的准确率差为 +10.24 个百分点，按源句聚类的 2,000 次 bootstrap 95% 区间为 [+2.91, +18.02] 个百分点。
新模型每候选对净 ΔCOMET 的 95% 区间为 [-0.001193, +0.001496]，跨过 0；不能据此声称具有稳定的质量增益。始终 Reject 的准确率也高于新模型的点估计，说明不能单凭超过 50% 的准确率宣称方法有效。

## 产物与验证范围

- 模型：`/mnt/huawei/ymb/model/laya-multilingual-accept-reject-v1/best/`。
- 冻结记录：模型父目录的 `evaluation_lock.json`。
- 四轮开发集结果：`history.json`；冻结测试：`test_evaluation.json`。
- 基础模型与固定动作对照及区间：`frozen_test_comparison.json`；逐对审计：`frozen_test_predictions.jsonl`。
- 标注数据：`exp/runs/acceptance_binary_v1/`；冻结后测试数据：`exp/runs/acceptance_binary_test_v1/`。
- 本地顺序执行记录、GPU 配置、数据审计、对照脚本及源码摘要：`exp/runs/laya_binary_train_job_v1/`。
- 60 项回归测试与 14 项子测试通过，静态检查通过；真实运行修复了 COMET 虚拟环境与模型快照软链接解析问题。
- 本次为判别器训练与冻结后的候选分类测试，未运行新模型的 `full_static/full_online` 端到端实验，未据此认定检索库进化有效。
- GitHub 仅同步源码、配置、测试与结果摘要；模型权重、训练数据和逐样本记录保留本地。

冻结检查点 SHA-256：`ef59df9b1fa18232770a7a1b85f59ec1863007464258036f56dfe3c987e38278`。
