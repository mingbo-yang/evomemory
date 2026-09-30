# 新完整方法 full_online：GPU1 流程验证

结论：真实执行的核心闭环完整，逐轮审计通过；当前默认 BERT 分差验收规则尚未证明质量收益，不建议直接扩大为正式主实验。

## 验证设置

- GPU：仅物理卡 1；Qwen3-8B，vLLM；BERT 与 MiniLM 在 CPU。
- 四任务各 32 条 dev，共 128 条，seed42，batch_size=4，最多修改 3 轮。不是正式测试集结论，也没有为获得正向结果而调阈值。
- 配置：configs/optimized_feedback_v1.json；初始库 experience/contrastive；独立输出 runs/optimized_flow_validation。
- 初稿由真实生成模型产生并缓存到 data/draft_cache/optimized_flow_dev_qwen3-8b.jsonl。
- 反馈筛选、BERT 验收与检索策略保持运行前配置不变。

## 真实结果

|任务|样本|初稿分数|最终分数|变化|接受修改|新增经验|在线经验命中|
|---|---:|---:|---:|---:|---:|---:|---:|
|coedit_gec (gleu)|32|82.302|82.302|+0.000|0|0|0|
|gigaword (rouge1)|32|30.451|30.451|+0.000|0|0|0|
|wmt19_en_zh (bleu)|32|38.843|36.718|-2.126|10|1|17|
|wmt19_zh_en (bleu)|32|22.310|22.265|-0.044|3|0|0|

英译中新增经验被 6 个后续样本检索，共 17 次。其 10 次接受中，参考指标 1 次提高、9 次下降；中译英 3 次接受中 2 次下降、1 次持平。纠错 93 次修改尝试全部被拒绝（84 次为相同或空候选），另有 1 次停止；摘要 32 条均在初稿评估后停止。

这些现象说明：旧 BERT 质量预测器能正常工作，不等于直接用其细小预测分差即可可靠验收修改；新接受门槛和停止阈值仍需要 dev 校准。不能由本次小样本把旧 BERT 本身判定为无效，也不能把闭环运行通过写成质量收益已成立。

## 完整性审计

- 四任务全部覆盖预定的 32 条 manifest 前缀，sample_id 无重复，source_hash 一致。
- BERT 停止与接受决策、长度闸门与配置可逐条重放。
- 每轮真实反馈以实际接受的当前答案为起点；最终输出等于最后接受状态。
- 入库集合恰好是“接受且正向反馈”的修改；state_before/state_after 正确，无多余条目。
- 初始经验未被修改；自身输入被排除；所有在线命中来自更早批次，没有同批或未来反馈。
- 对临时复制结果分别注入错误起点、同批提前检索，审计均正确报错（阳性故障对照）；未修改真实结果。

## 验证中修复的成本遗漏

首次 128 条 pilot 发现批处理继承代码未计初稿生成成本，同时把缓存初稿标记成非缓存。已仅对优化路径修复，增加 cost_accounting=initial_and_refinement_v2 参与配置身份，避免复用旧口径结果。
旧 pilot 的质量/反馈轨迹仍有效，但其 cost 字段仅计修改阶段，不能用于完整成本比较；本报告明确保留这一限制，没有伪造或回填历史 token。
修复后在同一 GPU1 用真实模型补跑 4 条摘要：全部直接 STOP，每条仍正确记录初稿生成 1 次、平均 119 token；成本审计通过。补跑结果位于 runs/optimized_flow_cost_check。
新增初稿生成/缓存读取测试通过；优化集成共 5 项测试通过，旧 resume-equivalence 测试通过。

## 产物

- reports/optimized_flow_validation.json：128 条逐轮审计与质量汇总。
- reports/optimized_flow_cost_check.json：4 条实际成本补跑审计。
- audit_optimized_flow.py：可复用的只读审计器。
- logs/optimized_flow_validation_gpu1.log、logs/optimized_flow_cost_check_gpu1.log：实际运行日志。

本次未扩大主实验、未更改当前接受阈值来追求正向结果。下一步应首先校准“是否接受修改”的独立规则，保持库更新消融的其他配置一致。
