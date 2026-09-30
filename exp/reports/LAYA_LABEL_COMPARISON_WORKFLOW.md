# Laya multilingual 标签来源对照：复现说明

输出目录：`/mnt/huawei/ymb/model/laya-multilingual-label-comparison-v1`。

本试验比较原 BLEU 差值标签与不看参考答案的语义教师标签。两组模型都从原始 `laya-multilingual` 初始化，保留相同输入，使用相同更新步数、优化器、增强和开发集选择标准。它是小规模分类试验，不是 `full_static` / `full_online` 主实验。

实际执行顺序：

1. `semantic_label_pilot.py prepare`：冻结训练 512、开发 48、概率校准 48、阈值校准 48、测试 96 对；按原文隔离，每个原文一对有改动候选。
2. `semantic_label_pilot.py label_evaluation`：通过既有本地 Qwen 27B 服务标注开发、校准和测试；测试标签保持封存。
3. `relabel_semantic_training_api.py`：同一 Qwen BF16 服务、精简 JSON 输出、交换 A/B 两种顺序；先做简单预检与训练样本审核，再标训练集。
4. `run_label_comparison_pipeline.py`：等待标注进程完成及 GPU 0 显存满足条件，依次运行配对训练和评估。
5. `train_label_comparison.py`：两臂共同移除语义 Uncertain 样本，训练 8 轮，按同一语义开发集的 Macro F1 选择。
6. `evaluate_label_comparison.py`：分别校准概率与接受阈值，保存 `comparison_evaluation_lock.json` 后才读取测试标签。保存官方 Agent 可加载的模型，并检查其输出与离线评估一致。

环境：`/home/ymb/miniconda3/envs/qwen35/bin/python`。教师使用既有 `127.0.0.1:8124` 服务，不重启该服务。训练使用 GPU 0。

可靠性检查：

- `python -m unittest -v test_label_comparison_protocol` 验证 A/B 映射、输入白名单、不确定接受计数和阈值选择。
- `protocol_checks.json` 保存原文隔离和字段检查；各输入、标签、协议、模型有哈希记录。
- `labels/` 与 `inputs/` 分离。参考答案、BLEU 分数、教师理由不会进入 Laya 的特征。
- 无可行阈值表示停止接受；零接受不算高精度成功。
- 测试指标同时报告解析明确的标签数、完整样本接受覆盖率、不确定接受的精度上下界，以及按原文重采样的配对区间。

保留的失败尝试：

- `rejected_glm9_labels/` 保存最初 GLM-4-9B 标签及旧协议；它通过简单预检，但真实案例中偏好原文没有的实体信息，故在训练前弃用。
- `relabel_semantic_training.py` 是失败的离线 CPU 卸载尝试，不是当前运行入口。当前 vLLM 的混合状态缓存不支持该组合。相关失败日志完整保留。
- 未加严格 JSON schema 的精简 API 请求曾返回格式错误；没有把失败响应转换成训练标签。实际入口已加格式约束，并重新通过预检。

解释限制：训练和评估共用教师模型，且输出格式不同；不能当成人工语义金标。新教师与 32 个 AI 盲审训练样本仅 24 个一致，仍有噪声；该抽样含定向补充，不能估计总体错误率。旧的 20,002 对模型见过本试验测试原文，因此不参与本次主比较。两组新模型与本试验测试原文隔离。

最终结果由评估脚本生成到输出目录的 `COMPARISON_ZH.md` / `comparison_report.json`，同时写入 `reports/LAYA_LABEL_COMPARISON.md`。未生成这些文件前，不能声称训练比较已完成。
