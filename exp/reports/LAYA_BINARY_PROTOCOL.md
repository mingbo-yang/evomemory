# 当前方案：Laya Accept / Reject

## 冻结的决策定义

模型可见输入严格限定为 `source`、`current`、`candidate`。参考答案、COMET/BLEU、经验内容与事后反馈不进入 Laya 的输入。目标为两个离散动作：

- Accept：以 Candidate 替换 Current。
- Reject：保留 Current；本轮没有可采纳候选时停止修改。

程序先比较 Current 与 Candidate 的文本：精确相同时直接 Reject，跳过 Laya，日志记为 `status=no_op`、`decision_origin=deterministic_rule`，不伪造模型概率。该规则在正式流程和 `Verifier.predict` 入口均执行。相等判断不做大小写、空白或 Unicode 归一化。

对于基本有效且文本发生修改的候选，模型使用两项选项的原始 logits 分类；完全相等时固定选择 Reject。没有 `p_better`、`p_worse_max`、概率分差等可调采纳阈值。概率与温度仅用于诊断。多个有效且有实际改动的候选被判为 Accept 时，采用生成顺序中的第一个，不按概率重排。

生成器、检索器及已训练的 Laya 在正式推理时冻结。每轮仍生成 4 个候选，最多 3 轮，初稿与预算保持可比较。输出完整性、空文本、硬长度上限属于基本有效性条件，不是模型质量置信度阈值。

## 离线训练标签

先按文本过滤，再计算 COMET 标签：

1. `Current == Candidate`：不用于主模型训练、开发集选模、诊断或测试分类评估；原始采集记录保留，去重后的有效 no-op 写入 `audit/<fold>_no_ops.jsonl`，单独记录数量和确定性 Reject，不调用 COMET。
2. `Current != Candidate`：保留，并计算 `ΔQ = COMET(Source, Candidate, Reference) − COMET(Source, Current, Reference)`。即使 COMET 分数相同，也仍是有效训练/评估样本，标为 Reject。

主模型学习和报告的对象仅为“已经发生文本修改的候选是否提升质量”。数据元信息记录 `model_population`、`no_op_rule`、有效去重总量与 no-op 数量；训练和评估入口会拒绝混入 no-op 的数据，避免简单字符串相等样本抬高分类指标。源级隔离仍检查所有原始源句，包括最终只留下 no-op 的源句。

按用户确认，`ε = 0`：严格正差为 Accept，零差与负差为 Reject，微小正差暂不剔除。旧 Better 对应 Accept，Tie/Worse 对应 Reject 只表示语义映射；新训练必须重算 COMET，不复用旧 BLEU 标签或把 Uncertain 强行改为 Reject。

源文本级划分、初始经验库与最终测试隔离继续执行。模型数据文件仅含 id、三个输入字段和目标 label；数值反馈单独保存在 audit。原来的 `threshold_calibration` 候选可作为 `decision_diagnostics` 折观察分类行为，不用于搜索运行时阈值。

训练保留 RLCD + CE，使用两个选项目标并随机排列选项位置。禁用原来的 Current/Candidate 随机交换及标签反转：Reject 合并了持平和变差，交换后不能简单变为 Accept。若单独做这种增强，必须用反向反馈 `−ΔQ` 重算标签。

开发集以 macro F1、accuracy 选择检查点，不使用最终测试选择参数。正式模型记录二分类 schema、标签顺序、输入字段、COMET 权重摘要、离线 ε 和已见源文本摘要；旧三分类检查点会被拒绝加载。Laya 上游的三个 temperature 缓冲值对应 question type，不代表此任务仍有三个类别。

## 延迟反馈与记忆

`ε_memory = 0`，仅严格正向 COMET 修改具有入库资格。它与训练标签的 ε 分别配置，均不作为 Laya 推理参数。

同一批任务全部结束后，对所有基本有效且实际改变答案的候选计算反馈，包括被 Reject 的候选和没有被选中的其他 Accept 候选。反馈使用每轮实际的 `before`，不是一律与初稿比较。去重后，只有 `full_online` 写入新经验；`full_static` 记录反馈但冻结检索库。

Candidate 被 Accept 不保证入库；被 Reject 也不阻止事后正反馈入库。参考答案文本只传给独立反馈模块，不写入生成器提示、Laya 输入或经验正文。该实验设定依赖能够对未执行候选获得反馈；若实际环境不具备这种能力，需要另外声明可观察反馈范围。

正式比较显式运行指定的 `full_static` / `full_online`，不再由旧 BLEU 的一次点估计自动决定是否运行 online。

## 本地入口

以下入口按顺序执行标注、训练与验证。2026-09-30 已启动 GPU 3 实验，运行状态以本地 `runs/laya_binary_train_job_v1/status.json` 为准：

```bash
cd /mnt/huawei/ymb/aaai2027/exp

# 先剔除精确相同文本并保留审计；再对实际改写计算 COMET 标签，ε=0。
python prepare_acceptance_training.py --output runs/acceptance_binary_v1

# 新检查点位于 model 文件夹；默认从基础 Laya multilingual 初始化。
python train_laya_acceptance.py

# 开发集分类评估。温度参数只能改变诊断，不改变采纳。
python evaluate_laya_acceptance.py \
  --output /mnt/huawei/ymb/model/laya-multilingual-accept-reject-v1/development_evaluation.json

# 用从未参与模型开发且与初始经验库隔离的 source 清单做正式比较。
python laya_relaxed_flow.py --manifest /path/to/heldout_manifest.jsonl \
  --arms full_static full_online --output runs/laya_binary_flow_v1
```

COMET 使用本地 `wmt22-comet-da` 和独立 Python 环境，默认在 CPU 上运行，不下载或静默退回 BLEU。正式生成与 Laya 推理默认使用指定 GPU；这些运行都需二分类模型训练完成后才能进行。

测试标签只允许在提供 `--frozen-checkpoint` 后构造，并验证反馈定义、权重摘要和训练/开发源文本隔离；不会在训练阶段自动打开测试反馈。支持从已有 `raw/test` 未标注轨迹读取候选。准备测试折时使用 `--folds test --output <新目录>`，随后对该目录运行 evaluator 的 `--fold test`。

## 初次代码迁移的验证记录

回归测试使用人工夹具验证两选项输入与梯度、标签与参考隔离、无阈值动作、真实 before、Reject 正例入库、Accept 负例不入库，以及冻结库/在线库差异；不代表真实模型效果。另检查本地真实 tokenizer 的两选项编码，不加载生成器或判别器权重。

初次代码迁移时尚未执行真实 COMET 重新标注、二分类模型训练和独立测试；当前 GPU 实验结果需单独报告。原训练权重、数据和历史结果不改写。

二分类初次迁移检查保存在 [实施检查记录](LAYA_BINARY_IMPLEMENTATION_CHECKS.json)。本次 no-op 规则更新后，59 项测试与 14 项子测试通过，静态检查通过，详见 [no-op 检查记录](LAYA_NO_OP_IMPLEMENTATION_CHECKS.json)。这些测试明确覆盖“文本改变但 COMET 相同仍保留为 Reject”，以及全部数据折的 no-op 排除与运行时模型调用跳过。
