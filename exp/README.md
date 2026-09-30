# EvoMemory：检索经验与答案修改实验

当前 Laya 验收任务为 **Accept / Reject 二分类**。模型输入只有 Source、Current Answer、Candidate Answer，直接按分类结果决定是否替换当前答案。完全相同的候选由程序直接 Reject；主训练和分类评估仅包含文本不同的候选，其中 COMET 持平或降低仍标为 Reject。no-op 保留在原始记录与独立审计统计中。当前协议与使用步骤见 [reports/LAYA_BINARY_PROTOCOL.md](reports/LAYA_BINARY_PROTOCOL.md)。

| 入口 | 用途 |
| --- | --- |
| `configs/laya_binary_v1.json` | 新协议；离线标签 ε=0、入库 ε_memory=0 |
| `prepare_acceptance_training.py` | 将 no-op 分离到审计文件，对实际改写重算 COMET 二分类标签 |
| `laya_acceptance_common.py` | 两选项编码、输入隔离、分类、检查点 schema |
| `train_laya_acceptance.py` | 从 Laya multilingual 基础权重训练二分类模型 |
| `evaluate_laya_acceptance.py` | 直接分类评估；概率温度仅影响诊断指标 |
| `laya_relaxed_flow.py` | 当前正式二分类流程，包含 `Verifier.predict`、`full_static` 与 `full_online` |
| `laya_relaxed_policy.py` | 按生成顺序选第一个 Accept；没有概率接受阈值 |
| `core/comet_feedback.py` | 独立的离线/任务结束后 COMET 反馈接口 |
| `legacy_laya_v1/` | 旧三分类实现，仅供历史实验与报告使用 |

`laya_relaxed_flow.py` 保留原文件名以维持入口位置，但运行产物已改为 `runs/laya_binary_flow_v1`。新模型保存到 `/mnt/huawei/ymb/model/laya-multilingual-accept-reject-v1`。旧三分类检查点不可作为新模型直接加载。新数据按 COMET 重标注后从基础 multilingual 权重训练，测试集仅在检查点冻结后开启。

现有 BERT、语义标签比较、旧阈值分析脚本属于历史实验，不是当前二分类主流程。当前不要求进行概率阈值搜索，也没有“必须达到 80% 才运行”的条件。

代码依赖同级 `../baseline/`，以及 `vendor/laya/` 中的第三方实现。2026-09-30 已启动 GPU 3 的 COMET 重标注、二分类训练与冻结后测试任务，并同步本仓库代码。实验未完成前不将回归测试解释为模型效果证据；本地任务状态记录于 `runs/laya_binary_train_job_v1/status.json`。
