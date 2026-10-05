# 当前方案：Laya Accept / Reject

## 冻结的决策定义

模型可见输入严格限定为 `source`、`current`、`candidate`。参考答案、COMET/BLEU、经验内容与事后反馈不进入 Laya 的输入。目标为两个离散动作：

- Accept：以 Candidate 替换 Current。
- Reject：保留 Current，并立即停止该任务的修改；不额外生成其他候选尝试通过验收。

程序先比较 Current 与 Candidate 的文本：精确相同时直接 Reject，跳过 Laya，日志记为 `status=no_op`、`decision_origin=deterministic_rule`，不伪造模型概率。该规则在正式流程和 `Verifier.predict` 入口均执行。相等判断不做大小写、空白或 Unicode 归一化。

对于基本有效且文本发生修改的候选，模型使用两项选项的原始 logits 分类；完全相等时固定选择 Reject。没有 `p_better`、`p_worse_max`、概率分差等可调采纳阈值。概率与温度仅用于诊断。每轮只有一个候选，直接执行该候选的二分类结果，没有多候选选择、排名或绕过 Laya 的正式分支。

生成器、检索器及已训练的 Laya 在正式推理时冻结。**每轮固定生成 1 个候选**，最多 3 轮；只有 Accept 后才可能继续下一轮，初稿与采样设置固定。每轮最多检索 4 条经验，这与候选数无关。输出完整性、空文本、硬长度上限属于基本有效性条件，不是模型质量置信度阈值。

## 离线训练标签

先按文本过滤，再计算 COMET 标签：

1. `Current == Candidate`：不用于主模型训练、开发集选模、诊断或测试分类评估；原始采集记录保留，去重后的有效 no-op 写入 `audit/<fold>_no_ops.jsonl`，单独记录数量和确定性 Reject，不调用 COMET。
2. `Current != Candidate`：保留，并计算 `ΔQ = COMET(Source, Candidate, Reference) − COMET(Source, Current, Reference)`。即使 COMET 分数相同，也仍是有效训练/评估样本，标为 Reject。

主模型学习和报告的对象仅为“已经发生文本修改的候选是否提升质量”。数据元信息记录 `model_population`、`no_op_rule`、有效去重总量与 no-op 数量；训练和评估入口会拒绝混入 no-op 的数据，避免简单字符串相等样本抬高分类指标。源级隔离仍检查所有原始源句，包括最终只留下 no-op 的源句。

按用户确认，`ε = 0`：严格正差为 Accept，零差与负差为 Reject，微小正差暂不剔除。旧 Better 对应 Accept，Tie/Worse 对应 Reject 只表示语义映射；新训练必须重算 COMET，不复用旧 BLEU 标签或把 Uncertain 强行改为 Reject。

源文本级划分、初始经验库与最终测试隔离继续执行。模型数据文件仅含 id、三个输入字段和目标 label；数值反馈单独保存在 audit。原来的 `threshold_calibration` 候选可作为 `decision_diagnostics` 折观察分类行为，不用于搜索运行时阈值。

训练保留 RLCD + CE，使用两个选项目标并随机排列选项位置。禁用原来的 Current/Candidate 随机交换及标签反转：Reject 合并了持平和变差，交换后不能简单变为 Accept。若单独做这种增强，必须用反向反馈 `−ΔQ` 重算标签。

开发集以 macro F1、accuracy 选择检查点，不使用最终测试选择参数。训练现在将每一轮推理权重独立保留到 `checkpoints/epoch_NN/`，记录各轮 SHA-256，另保留开发集选择的 `best/`；此前已完成的 1–10 轮检查点不包含优化器状态；自第 11 轮续训起，每轮额外保存 `training_state.pt`，其中包含 AdamW、scheduler、Python/NumPy/Torch/CUDA 随机数状态及模型权重摘要。全部轮次训练结束后生成 `all_checkpoints_lock.json`。正式模型记录二分类 schema、标签顺序、输入字段、COMET 权重摘要、离线 ε 和已见源文本摘要；旧三分类检查点会被拒绝加载。Laya 上游的三个 temperature 缓冲值对应 question type，不代表此任务仍有三个类别。

## 延迟反馈与记忆

`ε_memory = 0`，仅严格正向 COMET 修改具有入库资格。它与训练标签的 ε 分别配置，均不作为 Laya 推理参数。

同一批（默认 16 条）任务全部结束后，对所有基本有效且实际改变答案的候选计算反馈，包括被 Reject 的候选。正式流程每轮只有一个候选，没有多余的未选中 Accept 候选。反馈使用每轮实际的 `before`，不是一律与初稿比较。去重后，只有 `full_online` 写入新经验；`full_static` 记录反馈但冻结检索库。

Candidate 被 Accept 不保证入库；被 Reject 也不阻止事后正反馈入库。参考答案文本只传给独立反馈模块，不写入生成器提示、Laya 输入或经验正文。该实验设定依赖能够对未执行候选获得反馈；若实际环境不具备这种能力，需要另外声明可观察反馈范围。

正式比较显式运行指定的 `full_static` / `full_online`，不再由旧 BLEU 的一次点估计自动决定是否运行 online。

## 本地入口

以下入口按顺序执行标注、训练与验证。2026-09-30 已完成 GPU 3 的标注、训练和冻结后分类测试，结果见 [训练测试报告](LAYA_BINARY_TRAINING_V1.md)；本地运行状态记录于 `runs/laya_binary_train_job_v1/status.json`：

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
  --checkpoint /mnt/huawei/ymb/model/laya-multilingual-accept-reject-20ep-continued-v1/checkpoints/epoch_20 \
  --candidates-per-round 1 --arms full_static full_online --output runs/laya_binary_flow_v1
```

正式入口要求显式指定 `--checkpoint`，以上 epoch20 路径是明确指定的例子，不再隐式回退到早期 `best/`。`--candidates-per-round` 只接受 `1`；旧四候选配置和 `unfiltered_static` 不能从正式入口执行，程序调用层也校验单候选约束。日志保留长度为 1 的 `candidates` 列表和 `slot=0` 以兼容轨迹审计，不表示仍存在多个候选。

COMET 使用本地 `wmt22-comet-da` 和独立 Python 环境，默认在 CPU 上运行，不下载或静默退回 BLEU。正式生成与 Laya 推理默认使用指定 GPU；这些运行都需二分类模型训练完成后才能进行。

测试标签只允许在提供 `--frozen-checkpoint` 后构造，并验证反馈定义、权重摘要和训练/开发源文本隔离；不会在训练阶段自动打开测试反馈。支持从已有 `raw/test` 未标注轨迹读取候选。准备测试折时使用 `--folds test --output <新目录>`，随后对该目录运行 evaluator 的 `--fold test`。

## 初次代码迁移的验证记录

回归测试使用人工夹具验证两选项输入与梯度、标签与参考隔离、无阈值动作、真实 before、Reject 正例入库、Accept 负例不入库，以及冻结库/在线库差异；不代表真实模型效果。另检查本地真实 tokenizer 的两选项编码，不加载生成器或判别器权重。

初次代码迁移时尚未执行真实 COMET 重新标注、二分类模型训练和独立测试；真实 GPU 训练与分类测试结果现已单独记录在 [训练测试报告](LAYA_BINARY_TRAINING_V1.md)。原训练权重、数据和历史结果不改写。

二分类初次迁移检查保存在 [实施检查记录](LAYA_BINARY_IMPLEMENTATION_CHECKS.json)。本次 no-op 规则更新后，59 项测试与 14 项子测试通过，静态检查通过，详见 [no-op 检查记录](LAYA_NO_OP_IMPLEMENTATION_CHECKS.json)。这些测试明确覆盖“文本改变但 COMET 相同仍保留为 Reject”，以及全部数据折的 no-op 排除与运行时模型调用跳过。

## 逐轮测试诊断

`compare_laya_epoch_checkpoints.py` 支持按用户要求对已经使用过的 449 条候选比较全部轮次；默认 10 个进程在同一张指定 GPU 上评估，batch size 为 32，与原比较一致。它检查全部权重摘要、源隔离和反馈协议，复用原始 COMET 标签并保持原测试集元信息不变。这是事后诊断；测试集上的最高分轮次不会自动替代按开发集选择的 `best/`，也不作为新的盲测证据。

```bash
CUDA_VISIBLE_DEVICES=3 python compare_laya_epoch_checkpoints.py \
  --model /mnt/huawei/ymb/model/laya-multilingual-accept-reject-10ep-allcheckpoints-v1 \
  --output /mnt/huawei/ymb/model/laya-multilingual-accept-reject-10ep-allcheckpoints-v1/test_all_epochs \
  --workers 10 --batch-size 32
```

该诊断输出各轮指标、逐候选 logits、配对源级 bootstrap 区间及补跑复现检查。原先只保留的第 8 轮模型仍作为单独对照；补跑所得其余轮次明确标为补跑结果。

## 从第 10 轮权重继续到第 20 轮

`--warm-start` 只恢复已保存的模型权重；因为原第 10 轮没有优化器和 RNG 状态，不能称为完整恢复连续训练。`--initial-epoch 10 --epochs 10` 表示继续生成第 11–20 轮，样本顺序和选项排列使用全局轮次编号。原 1–10 轮权重保持原目录，统一索引指向原文件，新权重与训练状态在新目录保存。

本轮保持原终点学习率 `1e-6`（encoder/head 相同）及 RLCD 噪声 `0.1`，重建 AdamW，seed 仍为 42。它检验该续训设置下多训练 10 轮的影响，不等价于从头训练 20 轮，或重新启动高学习率日程。

```bash
CUDA_VISIBLE_DEVICES=3 python train_laya_acceptance.py \
  --warm-start /mnt/huawei/ymb/model/laya-multilingual-accept-reject-10ep-allcheckpoints-v1/checkpoints/epoch_10 \
  --output /mnt/huawei/ymb/model/laya-multilingual-accept-reject-20ep-continued-v1 \
  --initial-epoch 10 --epochs 10 --encoder-lr 1e-6 --head-lr 1e-6 --min-lr 1e-6 \
  --sigma-start 0.1 --sigma-end 0.1 --batch-size 16 --effective-batch 64 --seed 42
```

父模型第 10 轮由已用测试集上的表现选出，因此整个续训分支具有事后选择性质，相关 lock 会显式标记。续训过程中仍只按开发集选 `best/`；完成后对 1–20 轮统一复测相同 449 条候选，并报告相对第 10 轮的配对差异及置信区间。

## 正式单候选与采集/消融隔离（2026-10-04）

128条同源测试的1/4候选消融未显示四候选的额外收益，修改阶段逻辑token约为单候选4.14倍。差异区间包含0，不能据此声称四候选必然有害。按用户确认，正式流程现固定为单候选，`flow_schema=laya-single-candidate-v1`。历史结果保留在 `runs/laya_candidate_ablation_v1/`。

多候选的生成循环和首个Accept选择逻辑移到 `ablations/laya_candidate_count.py`，只由 `run_candidate_count_ablation.py` 显式使用。正式入口不导入消融、训练采集或标签准备模块；修改指令来自无模型依赖的 `core/refinement_instructions.py`。离线训练仍可采集多候选及混合温度，这些机制不进入正式流程。

正式流程为：初稿 → 检索经验 → 生成一个候选 → 基本有效性/no-op检查 → Laya分类 → Accept更新并进入下一轮，Reject停止 → 整批任务结束后对有效改写计算COMET → 正反馈且未重复的经验进入在线库。无效输出和no-op同样不继续尝试其他候选；no-op不调用Laya、不入库。被Reject的有效正向修改仍可入库，长度闸门和epsilon=0保持原定义。

`verify_single_candidate_replay.py` 使用历史真实生成缓存、Laya判定和COMET分数，重新计算检索与提示并检查全部轨迹/反馈/最终经验库一致性；这是CPU行为回归，不是重新生成候选或独立效果验证。
