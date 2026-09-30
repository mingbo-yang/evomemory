# BERT、混合检索与有反馈在线更新：实现版本 v1

本次用户授权优化 BERT 复用、检索与长度闸门。新协议通过显式参数启用，不改变旧实验默认行为，不覆盖旧配置或结果。已完成代码与 CPU 功能验证，未运行新的生成模型正式实验，尚不能宣称质量提升。

## 实现

- **BERT 判断**：复用旧 DeepBertClassifier 完整训练权重，严格加载 encoder 和回归头。仍使用 source/candidate 成对输入、512 token、原始回归分数；不加 sigmoid/clamp，不允许缺权重时随机初始化继续运行。adaptive 根据 BERT 阈值停止，fixed 保留固定轮数。生成模型仅负责根据任务指令和经验改写；不再自任 controller/judge。
- **混合检索**：双字段 BM25 与双字段语义排名通过 RRF 融合；本地 all-MiniLM-L6-v2 编码原文和修改前答案。排除自身输入；无词面支持且语义相似度低时允许空检索。库更新后重建索引，缓存已有文本向量，只编码新增文本；不复用可能过期的检索排名缓存。这里尚未增加单独的错误类型分类器，MiniLM 的中文效果需 dev 验证。
- **长度闸门**：基本接受条件为 BERT 增益 > 0.01。常规上限 `1.02 * 旧长度 + 8` 字符；超过常规上限需增益 > 0.05；绝对硬上限 `1.5 * 旧长度 + 8` 字符。相同/空候选不接受。
- **在线反馈**：整批完成后，修改被接受且反馈分数提高才入库。实验使用参考答案指标模拟任务完成后反馈，决策与接受本身不使用当前参考答案。保留已修复的 accepted-state 重放，不把拒绝候选当作下一轮修改起点。

## 权重和默认参数

基础配置/tokenizer：`/mnt/huawei/wwq/wwq/huggingface/model/bert-base-multilingual-cased`。

| 任务 | 已训练权重 | 停止阈值 |
|---|---|---:|
| 双向翻译 | `/mnt/huawei/wwq/model/aaa_experiment/control_n/model/layers_3_best.pth` | 0.6 |
| 纠错 | `/mnt/huawei/ymb/icml/bert/gec/model/model/layers_3_best.pth` | 0.9 |
| 摘要 | `/mnt/huawei/ymb/icml/bert/gigaword_tiny/model/model/layers_3_best.pth` | 0.3 |

摘要选择任务专用权重；旧生成脚本曾指向通用翻译权重。停止阈值沿用旧脚本；新增接受/长度/检索参数是待 dev 验证的工程默认值，不是调优结果。BERT 回归分数不是成功概率。

Hybrid 默认语义权重 0.5、RRF k=60、相似度门槛 0.2。配置见 `configs/optimized_feedback_v1.json`。

## 运行

从 `/mnt/huawei/ymb/aaai2027/exp` 执行：

```bash
# CPU 预检：不加载生成模型、不使用 GPU
CUDA_VISIBLE_DEVICES='' /home/ymb/miniconda3/envs/qwen35/bin/python check_optimized.py \
  --out reports/optimized_feedback_preflight.json

# 后续实验命令，本次没有启动。
# full_online 改为 full_static，其他参数相同，即为进化消融。
/home/ymb/miniconda3/envs/qwen35/bin/python run_experiment.py \
  --arm full_online --models qwen3-8b --tasks all --gpu 2 \
  --seed 42 --batch-size 32 --draft-source stored \
  --experience-root experience/contrastive --retrieval-excludes-own-source \
  --optimization-config configs/optimized_feedback_v1.json \
  --out-dir runs/optimized_feedback_v1
```

`retrieval.method` 可设为 `bm25`，用于相同 BERT/长度策略下的检索消融。不要同时传旧 `--accept-max-len-ratio`；新长度策略完全由 profile 管理。此协议不支持 BoN 或 full_both。

配置身份包含权重内容 SHA256、初始库指纹、batch 大小和策略版本。结果自动增加 `optimized-<profile hash>` 子目录，同一 profile 静态/在线臂共享 tag。在线库保存为本次运行目录中的 `online_experience.json`，不覆盖旧经验库。

入口是 `run_experiment.py`。现有 supervisor/dispatcher 的结构化队列尚不转发新增参数，不能沿用旧队列启动新协议；如需调度，可用已有显式 cmd 作业机制并匹配新 tag。

## 验证及限制

- 四任务实际 BERT 权重均已在 CPU 加载，每任务两个输入与旧评分函数最大绝对误差均为 0。
- 本地 MiniLM 实际加载和混合检索 smoke 通过。
- 假生成模型驱动完整静态/在线 runner，验证质量停止、长度接受、反馈延迟入库及输出隔离。
- 记录 quality_pairs、quality_latency_s、retrieval_latency_s；summary 另记 component_setup_s。LLM token 数不涵盖 BERT/encoder，比较总成本需同时报告额外耗时和总运行耗时。
- BERT 和 encoder 默认 CPU；完整规模速度尚未验证。
- 真实预检结果见 `reports/optimized_feedback_preflight.json`。

下一步应在 dev 验证新增阈值和检索相关性，再冻结新参数。旧初始库仍与评测集同源并排除自身；本次未改变该数据协议，不能描述成完全独立辅助库。


## 后续 GPU1 验证

已完成四任务 128 条 dev 的真实流程验证，并另跑 4 条验证初稿成本修复。流程审计通过，但未验证出质量收益；详见 `reports/OPTIMIZED_FLOW_VALIDATION.md`。成本版本 `initial_and_refinement_v2` 已加入新运行身份。
