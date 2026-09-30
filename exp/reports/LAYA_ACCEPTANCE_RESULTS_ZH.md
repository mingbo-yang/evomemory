# Laya multilingual 训练与验证结果

4 轮训练和独立验证已完成。使用 convaiinnovations/laya-multilingual，开发集选择第 4 轮权重。
验证结论：未通过 verifier 门槛，暂不能用于主实验。

训练集为 20,002 个候选比较样本；开发、概率校准、阈值校准和最终测试按原文隔离。
最终测试覆盖 400 个原文、793 个比较，其中 449 个确实发生修改。
选择权重和阈值后才解封测试；模型输入仅含原文、当前答案和候选答案。

| 模型 | 实际修改样本 Better AUC | 三分类 Macro-F1 | 校准后接受数 | 覆盖率 | 接受精度 |
|---|---:|---:|---:|---:|---:|
| 原始 Laya multilingual | 0.4793 | 0.2404 | 0 | 0.00% | 无定义 |
| 微调 Laya multilingual | 0.5518 | 0.5301 | 0 | 0.00% | 无定义 |

三分类指标包含完全相同的答案对，不能代替实际修改样本上的判别能力。
冻结的阈值要求：校准集接受精度至少 80%、覆盖率至少 10%、平均反馈增量为正；测试还要求按原文聚类 bootstrap 的增量置信区间下界大于 0。
没有合格阈值时关闭接受；零接受属于失败，精度无定义，不能报告为高准确率。

微调模型实际修改 AUC 的 95% 原文聚类 bootstrap 区间为 [0.4904, 0.6124]。
在独立阈值校准集中，满足至少 10% 覆盖率的网格阈值最高精度为 0.3605，仍未满足 80% 标准。
仅作诊断：校准后 pBetter ≥ 0.5 时，测试接受 40 对，精度 0.2250，平均代理反馈增量 -0.0306（0–1 尺度）。这不是一个通过验证的工作阈值。

补充对照：旧 BERT 在相同测试对上 Better AUC=0.5070；原接受策略接受 141 对，精度 0.3475，平均反馈增量 -0.0109。
该 BERT 对照沿用原有增益阈值与长度限制，不重训、不调参；不是完整在线流程的对比。

## 结果应如何解释

本次标签来自原有参考答案指标的分差，ε=0.01；它们是弱反馈标签，不是独立人工语义判断。已有训练集抽查发现同义改写造成较大分差，因此本次实验不能单独证明翻译质量判断能力，也不能把失败直接归因为某个模型架构。
已完成的是分类器训练、校准、测试与保存后的官方接口一致性检查。full_static/full_online 主实验尚未在本模型上验证。

## 保存位置

- 最终模型：`/mnt/huawei/ymb/model/laya-multilingual-acceptance-enzh-v1/best`
- 原始预训练模型：`/mnt/huawei/ymb/model/laya-multilingual`
- `training_protocol.json`：训练及评估规则。
- `evaluation_lock.json`：测试前冻结的模型、温度和阈值。
- `evaluation_report.json`：主验证指标和置信区间。
- `diagnostics.json`：修改样本、混淆矩阵及旧 BERT 补充对照。
- `best/acceptance_policy.json`：接受阈值、验证状态及使用限制。
- `history.json`、`training.log`：逐轮指标及训练日志。

![开发集学习曲线](/mnt/huawei/ymb/model/laya-multilingual-acceptance-enzh-v1/learning_curves.png)
