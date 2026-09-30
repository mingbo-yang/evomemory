# AAAI2027 经验驱动 Refinement —— 已批准方案 v5（重建件 / reconstruction）

> **本文件是重建件，不是原始文件。** 仓库中从未存在过方案文档（`core/__init__.py:4` 引用的
> `reports/plan_v5.md` 不存在）。本文件把 DSH 会话记录里**已被批准**的方案 v5 原文重新钉进仓库，
> 使"我们按方案执行"这句话可以被机械核对。

## 0.1 重建来源（精确路径与定位）

| 项 | 值 |
|---|---|
| 源文件 | `/home/ymb/.dsh/sessions/--mnt-huawei-ymb--/session-0a0e27ff-7513-4d3e-8a81-26903b5023a5/session.v3.jsonl.zstd` |
| 解压方式 | `zstd -dc <源文件> > session.jsonl`（3,824,262 B → 13,787,610 B，3,563 条记录） |
| 方案 v5 所在记录 | 解压后 JSONL **第 667 行**（`type=assistant/message`，`seq=665`，`data.turn=10`，`data.step=1`），方案正文在同一 step 的 `exit_plan_mode` 工具调用参数 `plan` 字段内（另见第 668 行 `tool/call`，`callId=call_00_bNU9tvxkB9tz5t3M1Xkm7766`） |
| 记录时间 | 2026-09-11 08:00:07（America/New_York） |
| 批准证据 | 第 669 行 `tool/result`：`"Plan approved — plan mode exited; carry out the plan starting with your next step."`；第 671 行 `plan/mode {"active": false}` |
| 批准后即开工的证据 | 第 676 行 `goal/change`（create）目标原文："在 /mnt/huawei/ymb/aaai2027/exp 下实现并执行「经验驱动 Refinement」实验方案 v5…" |
| 上级方案（用户的原始方案文本） | 同文件**第 341 行** `user/message`，标题《开放式生成 TTS：Experience-Driven Refinement 实验方案》，12,126 字符，2026-09-11 06:32:12 |
| 重建日期 | **2026-09-12**（EDT 05:12 起；重建时仓库为活动状态，见 §5 的时点说明） |
| 重建方式 | 方案 v5 正文为**逐字机械拼接**（从上述记录提取，未改写、未摘要），并用 `diff` 与会话原文逐字节核对通过；本文件其余小节为重建者撰写，凡引用均标注出处 |

### 候选版本清单（共 7 份完整方案稿，全部在 `exit_plan_mode` 工具调用中）

| # | transcript 行 | 时间 | 标题（原文） | 作者自标版本 | 结局 |
|---:|---:|---|---|---|---|
| 1 | 463 | 09-11 06:38 | 《AAAI2027 经验驱动 Refinement：可行性评估与复用优先实施方案》 | — | 被后续版本取代 |
| 2 | 522 | 09-11 07:39 | 《…复用优先实施方案（GPU 已就绪版）》 | — | 被后续版本取代 |
| 3 | 549 | 09-11 07:42 | 《…复用优先实施方案（零编码模型 · 2 卡版）》 | — | 被后续版本取代 |
| 4 | 616 | 09-11 07:50 | 《…最终实施方案（零编码模型 · 2 卡 · 全数据源已验证）》 | （未标注；即 v3 之前的"v2"） | 被后续版本取代 |
| 5 | 629 | 09-11 07:55 | 《…最终实施方案（**v3** · 单变量消融已加固）》 | v3 | 被 v4 取代 |
| 6 | 654 | 09-11 07:57 | 《…最终实施方案（**v4** · 消融单变量加固 + 检索分数归一化 + 预算重算）》 | v4 | 被 v5 取代 |
| 7 | **667** | **09-11 08:00** | 《…最终实施方案（**v5** · 不变量按臂对重写 + 预算自洽）》 | **v5** | **用户批准 → 本文件重建的对象** |

批准链条：第 588 行用户「好的，请你合并并再次检查方案」→ 第 628/641/666 行用户三轮逐条评审
（`NoOutcome` 必须复用 Full 的检索 ID、RandomRetrieve 控制 outcome 分布、BM25 双字段必须归一化、
"192 aux"口径错误、accumulation 配置不一致、§4.5 不变量写错 等）→ 第 667 行提交 v5 → 第 669 行批准。
v3/v4/v5 三稿即对这三轮评审的响应。**会话中不存在 v6 或任何 v5 之后的修订稿**（全文件检索 `方案 v6` / `plan_v6`
均为 0 命中）。

### 版本间差异（可见者，逐条列出）

**v2（行 616）→ v3（行 629）**
- v3 新增 §4 整节「单变量消融加固」：检索记忆化缓存 + 确定性 A/B 顺序 + 确定性采样种子，把"单变量"从约定变成可测试不变量。
- v3 §2.4 **删掉了 v2 的 `seed = args.seed + i*9973`**，改为按 `(sample_id, round, call_type)` 确定性导出种子（§4.4）。
- v3 把 Gigaword 重新分配为 initial 32 / dev 40 / accumulation 128（v2 同为 32/40/128，但 v2 的"192 aux"口径在 v3 被纠正为 32/model-task）。
- 预算从 ≈490 GPU·h / 10 天（v2/v3 相同）在 v4 被重算。

**v3 → v4（行 654）**
- ① RandomRetrieve 从"保留分层配额（上限）"改为**钉死每类条数**；② 新增 R3：BM25 双字段必须 **per-query min-max 归一化**（或 RRF）后才能加权，否则 α 无意义；③ 修正 initial experience 口径与预算（"上轮报的 490 是错的"）；④ 恢复 5-snapshot 完整设计并重算 Phase 8（360/180/90）。

**v4 → v5（行 667）**
- ① 不变量改为**按臂对**约束（原"任意两臂相同 state 返回相同 exp_id"是逻辑错误，会让 RandomRetrieve 无法实现）；② Full-Online 降为两模型 3-seed 补充实验，Phase 4 预算改为 127 GPU·h；③ Phase 8 评价次数重算（毛 180/360，净 152/304）；④ 新增 positive pool 预检（§4.4，≥8）；⑤ 改名 `NoOutcome`→`OutcomeHidden`、`NoNegativeExperience`→`PositiveOnly`；⑥ 删除 α"单调"验收要求。

## 0.2 未能恢复 / v5 本身未写明而必须标注的部分

方案 v5 的**正文已 100% 逐字恢复**（240 行、16,216 字节，diff 通过）。以下不是"丢失的文本"，而是
**v5 本身没有写明的细节**——凡本文件其他小节需要用到它们时，都已标注来源，绝不静默补写：

1. **Phase 5 的"8 配置"没有逐条列举**。v5 §8 只写「核心消融：8 配置 × 2 模型（2×2 跑 3 seed，其余 1 seed）」。
   逐条名单在 v1 的方案里才有（D7 列 7 个 + D8 单列 SR-J-Stop），与实现代码 `run_experiment.py` 的 arm 注册表
   （9 arm，去掉 `full_online` 后正好 8 个）一致。见 §3.1。
2. **"两个代表模型"在 v5 中未点名**（§8 Phase 1c「2 消融模型」、§6 D9「两个代表模型」、Phase 8「2 模型」）。
   其定义来自用户原始方案 §2.1「消融及补充实验只使用两个代表模型：GLM4-9B、Qwen3-8B」与 v3 的 Phase 10 行。
3. **推理配置数值 v5 未写**：`max_rounds=3`、`max_tokens`（WMT 1024 / GEC 1024 / Gigaword 128）、
   统一 context（用户原始方案 §12 写 8192，执行期已改为 4096，见 §4.3）。v5 只写了 controller/judge 各 max 256 tok、temperature=0。
4. **v5 §3 的 R2 是一句残句**——原文只有「**R2｜分词器按字段选择。**」（v4 的对应条文才有展开：
   「zh_en 的 input 是中文而 state 是英文，en_zh 反之」）。
5. **Phase 7 的"≤100 个 STOP 状态"未说明粒度**。v5 原文「Stopping Diagnostics：≤100 个 STOP 状态强制补一次 refinement」；
   用户原始方案 §8.2 写的是「最多每 task 抽取 100 个 STOP 状态」——两者粒度不同，本文件按原文并列，不作取舍。
6. **Phase 5 的样本量与任务范围 v5 未写**（用户原始方案 §5：GLM4-9B + Qwen3-8B、四个任务、seeds 42/43/44）。
7. **Phase 6 的 K/N 扫描所用样本范围与评价集 v5 未写**（只写 K∈{1,2,3 复用}+K=5、BoN N∈{1,2,4 复用}+N=8，110 GPU·h）。
8. **Phase 9 的"绘图"没有图表清单**（v5 原文只有「统一重算评分、bootstrap、绘图」，3 h）。
9. **一处 v5 内部数字冲突：Gigaword 的 dev 切分 40 还是 32。** 见 §5.4 冲突 1，含逐字引文。
10. **v5 未写 controller prompt 的具体措辞**（只写输出协议）。执行期实际使用的判据见 §4.2 与 §2 表 F4。

---

## 1. 已批准方案 v5 正文（逐字重建，未改写、未摘要）

> 以下 240 行是 transcript 第 667 行 `exit_plan_mode.plan` 字段的**逐字内容**（正文自 `# AAAI2027 经验驱动 Refinement：最终实施方案（v5 · 不变量按臂对重写 + 预算自洽）` 起，
> 至「（本轮仅做只读检查；早前一次瞬时可写性探测已立即清理，未新建或修改任何文件。）」止）。
> 机械核对：`diff <(本段) <(提取自 transcript 的原文)` 无差异。

<!-- BEGIN VERBATIM plan_00667.md -->
# AAAI2027 经验驱动 Refinement：最终实施方案（v5 · 不变量按臂对重写 + 预算自洽）

## 0. 摘要

- 四个任务全部有干净辅助数据；零编码模型、零外部下载；GPU 限 2、3 号卡。
- **v5 修正**：① 不变量改为**按臂对**约束（RandomRetrieve 本就应该返回不同 ID）；② Full-Online 降为两模型 3-seed 补充实验，Phase 4 预算自洽；③ Phase 8 评价次数重算（180/360，净 152/304）；④ 新增 positive pool 预检；⑤ `NoOutcome`→**`OutcomeHidden`**、`NoNegativeExperience`→**`PositiveOnly`**；⑥ 删除 α"单调"验收要求。
- **推荐档 ≈700 GPU·h → 2 卡约 14.5 天。**

---

## 1. 数据源（逐项实测）

| 任务 | 测试集 | 辅助数据源 | 干净量 | 验证 |
|---|---|---|---:|---|
| WMT19 en→zh | `validation.parquet[:1000]` | `train.parquet` | 1,998,814 | ⚠️ 与 validation 有 **1 en / 2 zh 碰撞 → 必须过滤** |
| WMT19 zh→en | 同上（zh 侧为输入） | 同上 | 同上 | 同上 |
| CoEdit GEC | `coedit gec[:1000]` | `coedit gec[3000:19823]` | **16,823** | ✓ |
| Gigaword | `gigaword_tiny` **train**（100） | `gigaword_tiny` **validation(100)+test(100)** | **200** | ✓ 三 split 互不相交 |

`train.parquet` 与 validation 有碰撞 → manifest 必须是**过滤操作**，不能只做零交集断言。
Gigaword 分配：initial **32** / dev 40 / accumulation 128；任何一行被剔除即报错中止。

---

## 2. 模型一致性硬约束

**`aaai2027/baseline/` 只读复用，绝不修改；新逻辑全部在 `aaai2027/exp/`。**

1. **环境锁定**：只用 `qwen35`（vllm 0.17.0 / transformers 4.57.6 / torch 2.10.0+cu128）；不得为 nltk 切到 `llama4`；版本写入 metadata 并断言。
2. **模型路径唯一来源**：`from core.config import MODEL_CONFIGS`；禁止硬编码与未注册 checkpoint。
3. **GLM 特例**：只用 baseline 本地 `/mnt/huawei/ymb/model/glm-4-9b/model`，不得用 icml 的 `THUDM/glm-4-9b-chat-hf`。
4. **采样参数逐字一致**：`temperature=0.1`、`top_p=1.0`、`max_tokens` 取自 `TASK_CONFIGS`、chat template 走同一 `LLMClient._chat_text`。
5. **不引入第二个模型**：controller 与 judge 复用同一 task model 的 LLMClient 实例。
6. **Phase 0 复现门禁**：每 model-task 重跑 5 条 Direct-Zero 与已存 `final_output` 精确比对；不一致则放弃 `y₀` 复用。

**GPU 白名单**：`--gpus 2,3`，必须在**未设置 `CUDA_VISIBLE_DEVICES`** 的 shell 中启动调度器。

---

## 3. 检索方案（零编码模型）

**动机数据**（已有 6 万步轨迹，5 模型 × 4 任务 × 前 1000）：

| 任务 | 方法 | improve | degrade | **tie** | 净 ΔQ/步 |
|---|---|---:|---:|---:|---:|
| en_zh | SR-Fixed | 15.1% | 17.1% | **67.8%** | −0.0015 |
| en_zh | SelfRefine-Fixed | 26.0% | 32.9% | 41.1% | −0.0080 |
| zh_en | SR-Fixed | 16.2% | 19.3% | **64.5%** | −0.0038 |
| zh_en | SelfRefine-Fixed | 26.4% | 36.0% | 37.6% | −0.0131 |
| gec | SR-Fixed | 4.9% | 5.2% | **89.9%** | −0.0002 |
| gec | SelfRefine-Fixed | 14.8% | 28.3% | 56.9% | −0.0234 |
| gigaword | SR-Fixed | 29.5% | 35.2% | 35.3% | −0.0025 |
| gigaword | SelfRefine-Fixed | 35.1% | 38.0% | 26.9% | +0.0005 |

**R1｜自建 BM25，不用编码模型。** ⚠️ 仓库现有 `compute_bm25`（`icml/baseline/generation/en-zh/glm4-9b/en_zh_bm25_10_data_related.py:142`）返回**原始无界分数**且用 `doc.split()` 纯英文分词——英文侧可借鉴，中文侧必须改字符 1–2 gram。
**R2｜分词器按字段选择。**
**R3｜双字段分数必须先归一化再加权。** 原始 BM25 分数跨语言/跨字段尺度不可比，直接加权会让 α 失去意义。做法：**per-query min-max 归一化**（首选，池子 ~192 条，确定性、与缓存兼容）或 **RRF 倒数秩融合**。
**R4｜outcome 分层配额**：每个 verdict 类最多 ⌈k/2⌉ 条。GEC 上 89.9% 是 tie，不做配额则 top-4 极可能全为"什么都没发生"。
**R5｜tie 也要显式呈现。**
**R6｜条数上限 4 封顶，token 上限仅作安全阀。** 实测经验单元 ≈ **224 token**（Qwen3-8B tokenizer）；4 条 ≈ 900 token，2048 budget 可容 9–13 条 → 条数上限先于 token 上限生效，保证跨任务可比。
**R7｜索引落盘 + 哈希**，命中 exp_id 写入 trace。

**dev 门禁**：BM25 vs 随机检索的 controller 决策质量对比。

---

## 4. 单变量消融加固（方案核心）

### 4.1 对照表（每行只变一个东西）

| 臂 | 经验池 | 排序 | **每类条数** | outcome 可见 |
|---|---|---|---|---|
| Full | 全量 | BM25（归一化后） | Full 的每类条数 | 是 |
| **OutcomeHidden** | 全量 | 同 Full | **同 Full** | **否（仅渲染层遮蔽）** |
| **RandomRetrieve** | 全量 | **随机** | **同 Full（钉死配比，非仅上限）** | 是 |
| **PositiveOnly** | **仅正面** | BM25（归一化后） | 退化 | 是 |
| NoExperience | 无 | — | — | — |

**命名修正**：`NoOutcome`→**`OutcomeHidden`**（它仍用 outcome 做分层检索，只从 prompt 隐藏，叫 NoOutcome 属过度声称）；`NoNegativeExperience`→**`PositiveOnly`**。配置项 `--experience {full,random,none,outcome_hidden,positive_only}`。

**配比钉死**：仅保留"每类 ≤⌈k/2⌉"的上限不够——Full 可能是 2 improve+2 tie，随机版可能是 1+1+2。必须把**每类条数钉死为 Full 的每类条数**，只在类内换成随机排序。

### 4.2 检索记忆化缓存

按 **(model, task, snapshot, round, hash(input), hash(state_before))** 记忆化，保证臂内确定性；跨臂一致性见 §4.5。

### 4.3 两个全局隐藏变量

1. **Judge A/B 顺序**：由 `hash(model, task, sample_id, round)` **确定性导出**，所有臂同一排列。
2. **采样种子**：controller/refine/judge 的 seed 按 `(sample_id, round, call_type)` 确定性导出，而非全局递增。

### 4.4 PositiveOnly 的固有限度与预检

经验池本身是被操纵对象，池子变化必然改变 exp_id，无法做到同等严格。此外 **positive 条数可能不足**（GEC 上 SR-Fixed improve 仅 4.9%）。因此：

- **Phase 1 预检**：测量每个 model-task 的 positive transition 数量，**要求 ≥8**（为相似度筛选留余量）。
- 不足时**全局**提高每个 aux input 的 intervention 数 K（而非更换辅助输入集合），重建经验库。**K 只允许在 Phase 1 全局确定一次**，禁止按臂调整，否则破坏单变量性。
- 若提高 K 后仍不足，该 model-task 允许 k<4 并在论文中记为偏离。
- 同时报告两臂的**检索相似度分布**，供读者判断是否顺带降低了匹配质量。

### 4.5 不变量（v5 重写：按臂对约束）

原先写成"任意两臂相同 state 返回相同 exp_id"是**错误**的——RandomRetrieve 的定义就是返回不同 ID，该断言会让它无法实现。正确表述：

| 关系 | 约束 |
|---|---|
| **Full vs OutcomeHidden** | 相同 `(input, state_before)` ⇒ exp_id 列表**逐字相同**（第 0 轮必然成立；后续按同一过程重检索，一致率写入 trace） |
| **RandomRetrieve vs Full** | **条数相同 + 每类 outcome 条数相同**；ID **按设计不同**；臂内由 state hash 播种保证确定性 |
| **所有臂** | A/B 顺序与采样种子逐字相同 |

**测试**：分别断言上述三条；`delta_offline`/`reference_text` 不得出现在任何 prompt 快照中。

---

## 5. 复用清单与一处撤回

| 资产 | 复用方式 | 节省 |
|---|---|---|
| 220 次 baseline | 统一 scorer 重算，截断至前 1000（gigaword 100） | 12 方法 × 19 model-task 生成成本 |
| Direct-Zero 输出 | **作为 `y₀`** | 每样本省 1 次调用 + 初稿条件一致 |
| `baseline/core/*.py` | 直接 import，不修改 | ≈1,000 行基础设施 |
| `run_baseline.py` CLI + 调度器/监控 | 沿用相同 flag 与输出三元组 | 调度与监控 |
| 已有 6 万步轨迹 | Phase 0 动机分析 | 免去动机实验 |

**撤回"icml 轨迹冷启动初始经验"**：(a) icml 的 intervention 来自其自身 critique prompt，指令分布不同；(b) 实测 `round_0_reasoning` 列 **200/200 行为空**。代价仅 ~1.5 GPU·h，不值得引入混淆。

---

## 6. 设计决策

**D1** 环境见 §2.1。**D2** seed 42 的 `y₀` 用已存 Direct-Zero（经门禁），43/44 重新生成。
**D3** baseline **只重算不重跑**（因此不需要 U 预测器/BERT）；重算数字与现有 `*.summary.json` 不同，所有表格整体重生成。
**D4** Gigaword 用 validation+test 200 条；测试 N=100 须标注。
**D5** 检索：词法 + **归一化** + 分层配额。
**D6** 消融正交化：`--experience {full,random,none,outcome_hidden,positive_only}` × `--stop {adaptive,judge,fixed}`。**配置差异只允许体现在 §4.1 的那一列上。**
**D7** SR-J-Stop（无 controller，仅 judge 接受为停止信号）≠ NoExperience（有 controller、无经验）。
**D8** seed 分层：仅 2×2 跑 3 seed，其余 1 seed。
**D9（v5 修正）** **Full-Static 为五模型主表（1 seed）；Full-Online 降为两个代表模型的 3-seed 补充实验**（打乱测试顺序）。P4 结论由独立 128 条 stream 承担。
**D10** 成本口径**必须含 controller 与 judge**。
**D11** Judge 自偏好缓解：主 judge 复用 task model；200 条子集用不同模型交叉 judge，并与人工盲评对齐。

---

## 7. 模块

```
aaai2027/exp/
├── core/{manifest,experience,retrieval_cache,bm25_fields,controller,judge,pipeline,scoring,stats}.py
├── configs/  data/{manifests,auxiliary_splits}/
├── experience/{initial,snapshots,indexes}/
├── runs/{main,ablation,scaling,stopping,accumulation}/
├── scores/  plots/  human_eval/  reports/
```

**Experience schema**
```json
{"exp_id": "wmt19_en_zh/qwen3-8b/aux-0007/r1",
 "task": "wmt19_en_zh", "model": "qwen3-8b",
 "source_input": "...", "state_before": "...", "state_after": "...",
 "intervention": {"instruction": "...", "rationale": "..."},
 "outcome": {"verdict": "better|worse|tie|uncertain",
             "reason_a": "...", "reason_b": "...", "order_consistent": true},
 "delta_offline": 0.031,
 "provenance": {"run_id": "...", "round": 1, "source": "wmt19_train|coedit_gec_3000+|gigaword_tiny_valtest"}}
```
`delta_offline` 仅供离线评价，渲染器走字段白名单，**必须屏蔽**。

**Controller**：任务输入 + 当前答案 + 4 条经验 → `{"action":"REFINE"|"STOP","instruction":"...","reason":"..."}`，`temperature=0`，max 256 tok，struct 失败重试 1 次，仍失败则保留当前答案并标记 `structured_failure`。
**Judge**：匿名 A/B + 交换顺序各一次 → `{"verdict":"A"|"B"|"tie"|"uncertain","reason":"..."}`，`temperature=0`，max 256 tok；**两次均明确判 candidate 更优才接受**；全程 reference-free。
**写入时机**：样本完全结束后才提交，按 `exp_id` 去重。

**统一评价**：en→zh SacreBLEU `zh`；zh→en SacreBLEU `13a`；GEC NLTK `sentence_gleu`；Gigaword ROUGE-1/2/L（`use_stemmer=True`）。全部 ×100；所有来源**共用同一 scorer 实例**。依赖仅 `pip install sacrebleu nltk rouge_score`。

---

## 8. 分阶段执行与预算（仅 2、3 号卡）

锚点：Full ≈ 13 calls/样本 ≈ 13.4 s → 每 model-config 全 4 任务 ≈ 11.5 GPU·h。

| Phase | 内容 | GPU | 预估 | 复用 |
|---|---|---|---|---|
| **0** | 环境/GPU 断言 + 复现门禁 + 6 万步动机表与 oracle-stop 上界 | 少量 | 3 h | **全部复用** |
| **1** | manifest（**过滤式**）+ 隔离断言 + 辅助切分 + dev 门禁 + **positive pool 预检（§4.4）** + 不变量测试 | 无 | 10 h | `core/data.py` |
| **1b** | `pip install sacrebleu nltk rouge_score` | 无 | 1 h | — |
| **1c** | dev 切分轨迹（2 消融模型 × 4 任务 × 32） | 有 | 0.5 GPU·h | — |
| **2** | 每 model-task 3 条真实 smoke | 有 | 1 GPU·h | `y₀` |
| **3** | **Initial Experience：32/model-task** × 5 模型 × 4 任务 | 有 | 1.5 GPU·h | — |
| **4** | **Full-Static × 5 模型（主表，1 seed）** + **Full-Online × 2 模型 × 3 seed（补充）** | 有 | **127 GPU·h** | **12 baseline 重算 + `y₀`** |
| **5** | 核心消融：8 配置 × 2 模型（2×2 跑 3 seed，其余 1 seed） | 有 | 368（全 1 seed 则 184） | 2×2 的 Full 格复用 Phase 4 |
| **6** | Scaling：K∈{1,2,3 复用}+K=5；BoN N∈{1,2,4 复用}+N=8 | 有 | 110 GPU·h | **K≤3 / N≤4 复用** |
| **7** | Stopping Diagnostics：≤100 个 STOP 状态强制补一次 refinement | 有 | 3 GPU·h | 主轨迹复用 |
| **8** | Accumulation：见下表 | 有 | 47 / 88 / 176 | snapshot-0 复用 Phase 4 |
| **9** | 统一重算评分、bootstrap、绘图 | 无 | 3 h | **220 次运行全复用** |
| **10** | 人工盲评材料导出（≤200 对） | 无 | 2 h | — |

**Phase 8 推导（v5 重算）**

| 选项 | 配置 | 毛次数 | 净新增 | GPU·h |
|---|---|---:|---:|---:|
| 8A 完整 | 5 snapshot × **4 任务** × 3 order × 3 method × 2 模型 | 5×4×3×3×2 = **360** | **304** | ~176 |
| **8B 推荐** | 5 snapshot × **2 任务** × 3 order × 3 method × 2 模型 | 5×2×3×3×2 = **180** | **152** | ~88 |
| 8C 精简 | 3 snapshot × 2 任务 × …（**须在论文标注为 3 点曲线**） | 90 | 80 | ~47 |

净新增的推导：snapshot-0 的 initial experience 按设计**跨 order 共享**，其 3 个 order 折叠为 1（4 任务减 48、2 任务减 24），得 312/156；再扣除 Phase 4 已覆盖的 snapshot-0 Full 格（8/4），得 304/152。若保守按你给的 144/288 计，预算 +4/+12 GPU·h。

建议 **8B**：保留 5 个 snapshot（累积曲线形状是"过去算力帮助未来算力"的核心证据），任务数由 4 降为 2，并**在论文中显式标注为预算受限的折中**。

**总预算（2 卡）**

| 档 | 构成 | GPU·h | 墙钟 |
|---|---|---:|---:|
| A 完整 | 2×2 三 seed + 8A | ~790 | ~16.5 天 |
| **B 推荐** | 2×2 三 seed + 8B | **~700** | **~14.5 天** |
| C 精简 | 消融全 1 seed + 8C | ~475 | ~10 天 |

Phase 0/1/1b/9/10 为纯 CPU，可与 GPU 阶段并行。±20% 量级估算。

---

## 9. 验收标准

1. **无编码模型**：依赖仅 `sacrebleu/nltk/rouge_score`；无 HF 模型下载。
2. **模型一致性**：metadata 版本等于锁定值；模型路径全部来自 `MODEL_CONFIGS`；`baseline/` git 无改动。
3. **复现门禁**通过，或已按规则放弃 `y₀` 复用。
4. **GPU 白名单**：所有运行 metadata `CUDA_VISIBLE_DEVICES ∈ {2,3}`；0/1 号卡无新进程。
5. **隔离断言**：test ∩ aux 哈希零交集；翻译方向排除 icml KB 0–999 行；Gigaword 辅助仅取 validation+test。
6. **不变量（§4.5，按臂对）**：Full vs OutcomeHidden 的 exp_id 逐字相同；RandomRetrieve 与 Full 条数及每类条数相同、ID 不同；所有臂 A/B 顺序与种子逐字相同。
7. **归一化验证**：两字段相似度归一化后落在 [0,1]。
8. **α 选择**：由**预注册的 dev 指标**选定，**tie-break 规则预先固定**，随后冻结并写入 config hash。**不要求 α 效应单调。**
9. **positive pool 预检**：每个 model-task positive ≥8；K 全局一次性确定；不足情形已在论文记录。
10. **渲染器测试**：`delta_offline`、`reference_text` 不出现在任何 prompt 快照中。
11. **smoke**：4 任务各 3 条端到端完整轨迹。
12. 主表 2000 次 paired bootstrap CI，每个数字可追溯到 `runs/` 原始 jsonl。
13. 成本表含 controller 与 judge 的 token/调用分解。
14. 2×2 四格在同一 model-task-seed 下 `y₀` 完全一致（哈希校验）。
15. 所有新结论报告 3 seed 方差，或明确说明为何只跑 1 seed（Full-Online 按其 2 模型 3 seed 补充报告）。
16. `PositiveOnly` 须附两臂检索相似度分布对比（§4.4）。
17. **所有相对原设计的缩水（Phase 8 任务数/snapshot 数、消融样本量、Full-Online 模型数）在论文中显式标注为预算折中。**

（本轮仅做只读检查；早前一次瞬时可写性探测已立即清理，未新建或修改任何文件。）
<!-- END VERBATIM plan_00667.md -->

---

## 2. 冻结决策（pre-registered，批准时即已固定）

本节汇总"方法组件在看见测试集之前就已冻结"的那些决定。**来源**：方案 v5 正文（§1 之上）、
用户原始方案（transcript 行 341）、以及执行期把它们落盘的 `configs/frozen.json`。

| # | 冻结项 | 冻结值 | 依据 |
|---|---|---|---|
| F1 | **检索权重 α** | **α = 0.5**；预注册集合 α ∈ {0, 0.5, 1}；规则：按 **dev** 指标取优，**完全并列时取更大的 α**；选定后冻结并写入 config hash；**不要求 α 效应单调** | v5 验收标准 #8（逐字：「由**预注册的 dev 指标**选定，**tie-break 规则预先固定**，随后冻结并写入 config hash。**不要求 α 效应单调**」——v5 未写具体取值集合）；α 集合见 v4 §3 R3「α ∈ {0, 0.5, 1} 在 dev 上选型；**归一化必须与 α 一起在 dev 上冻结后再用于测试**」；"完全并列取更大 α" 与考察范围记录于 `configs/frozen.json.alpha_selection`（`rule` / `scope`，`chosen: 0.5`） |
| F2 | **dev 切分** | 每个 model-task 的 `dev` manifest；**用于** dev 门禁（BM25 vs 随机）、α 选型、controller prompt 调参；**测试集不得参与任何方法组件选择** | v5 §8 Phase 1「dev 门禁」+ 验收 #8；落盘 `frozen.json.controller_prompt.tuned_on` 与 `alpha_selection.scope`。注意 dev 条数存在 40/32 冲突，见 §5.4 冲突 1 |
| F3 | **α 选型的臂与样本** | **forced-refinement（`fixed_rounds`）臂**、glm4-9b × 4 任务 × **16 条 dev**；理由：α 只影响检索到哪些经验，自适应停止下 32 条里只修订 0–6 条，比较是空的 | `frozen.json.alpha_selection.scope` / `.why_forced`；实测轨迹 `runs/gate_alpha/fixed_rounds/seed42/*/glm4-9b/dev/a{0.0,0.5,1.0}/`（每文件 16 行） |
| F4 | **controller prompt 判据** | **基于缺陷（defect-based）**：先说出一条**具体缺陷**，说不出就 STOP；说得出时，只有"修它值得再改一版"才 REFINE，否则 STOP。**不得**在提示词里断言论文结论 | `frozen.json.controller_prompt.criterion`；实现 `core/controller.py:51-54`；变更经过见 §4.2 |
| F5 | **y₀ 来源** | `draft_source = "stored"`：**复用已存 Direct-Zero 文本作为 y₀**，不重新生成；dev 扫描另加一次生成、多次回放的 `data/draft_cache/dev_<models>.jsonl` | v5 §6 D2（原文：seed 42 用已存 Direct-Zero、43/44 重新生成）+ 执行期变更（§4.1）后的实际冻结值 `frozen.json.operational_constants.draft_source="stored"`、`frozen.json.draft_cache` |
| F6 | **GPU 策略** | 批准时：`--gpus 2,3`，0/1 号卡不得有新进程（验收 #4）。**执行期已依次放宽到 0/1/2/3 四张卡全部可用**（`frozen.json.gpu_policy.effective_rule`），因此验收 #4 被显式取代 | v5 §2 末行 + 验收 #4；`frozen.json.gpu_policy`（含 4 步授权历史）；代码常量 `core/__init__.py` `ALLOWED_GPUS=('0','1','2','3')`、`MAX_GPUS=4` |
| F7 | **无编码模型** | 检索为**自建 BM25**（per-query min-max 字段归一化，CJK 字符 1–2 gram），**不引入任何句向量/BERT/编码器**；依赖仅 `sacrebleu/nltk/rouge_score` | v5 验收 #1；`frozen.json.encoders.used="none"`；用户原始方案 §11 的 MiniLM 方案被 v1 §3 显式取消 |
| F8 | 其他随批准固定的数值 | 经验条数上限 **k=4**（每 verdict 类 ≤⌈k/2⌉）、经验 prompt ≤2048 token、refinement 上限 **3 轮**、controller/judge `temperature=0`、生成 `temperature=0.1` / `top_p=1.0`、judge 双顺序且**两次都判 candidate 更优才接受**、测试集前 N（1000/1000/1000/100）、辅助切分 32/32/128、seed 42/43/44（**仅 2×2 跑 3 seed**）、主表 2000 次 paired bootstrap | v5 §3 R4/R6、§7、§8、验收 #12；`frozen.json.operational_constants`（`retrieval_k=4`、`max_rounds=3`、`temperature_controller_and_judge=0.0`）；`core/__init__.py`（`TEST_SAMPLES`、`AUX_SIZES`） |

### 2.1 与 `configs/frozen.json` 的一致性核对（结论）

- **一致**：α=0.5 及其"dev 指标 + 并列取大"规则、forced-refinement/16 条 dev 的选型方式、
  `draft_source="stored"`、controller prompt 的 defect-based 判据、`max_rounds=3`、`retrieval_k=4`、
  controller/judge temperature=0、无编码模型、BM25 字段归一化、测试集 1000/1000/1000/100 与辅助切分 32/128……
  全部与 v5 及用户原始方案相符（逐项证据见 §5.3）。
- **冲突**：dev 条数 40 vs 32（§5.4 冲突 1）、GPU 验收 #4 被取代（§5.4 冲突 2）、
  D2 的"43/44 重新生成初稿"与实际全程 stored（§5.4 冲突 3）、
  GLM 模型目录被覆盖为 `/home/ymb/glm_local/model`（§5.4 冲突 4）。
  这四项均已在 §4 执行期变更记录中留有授权与理由，**不是**无记录的漂移。

---

## 3. 方案原文之外、由上级方案与实现固定的定义【重建注】

v5 把某些早先已批准的定义压缩成了简称。下列两条在本文件其他小节被引用，故在此固定其来源；
**它们是重建者的整理，不是 v5 的原文**，逐字依据一并给出。

### 3.1 【重建注 R1】Phase 5 的 8 个配置

v5 §8 只写「核心消融：**8 配置** × 2 模型（2×2 跑 3 seed，其余 1 seed）」，没有逐条列举。可恢复的定义来自：

- **v1 方案 D7（逐字）**：「**D7｜消融正交化**：`--experience {full,random,none,no_outcome,positive_only}` × `--stop {adaptive,judge,fixed}`。
  Full=(full,adaptive)、FixedRounds=(full,fixed)、NoExperience=(none,adaptive)、SR-J-Fixed=(none,fixed) → 即 2×2；RandomRetrieve=(random,adaptive)、NoOutcome=(no_outcome,adaptive)、NoNegativeExperience=(positive_only,adaptive)。」
- **v1 方案 D8（逐字）**：「**D8｜SR-J-Stop 与 NoExperience 的区分**：**SR-J-Stop** = 无 controller，仅以 judge 是否接受为停止信号；**NoExperience** = 有 controller 但看不到经验。」
- **实现代码 `run_experiment.py` 的 arm 表**（9 个 arm）：`full_static / full_online / random_retrieve / outcome_hidden / positive_only / no_experience / fixed_rounds / sr_j_fixed / sr_j_stop`，
  去掉由 Phase 4 另行承担的 `full_online` 后正好 **8** 个。

**结论（推断，非原文）**：Phase 5 的 8 配置 = **2×2 四格（Full / FixedRounds / NoExperience / SR-J-Fixed）
+ RandomRetrieve + OutcomeHidden（v5 前叫 NoOutcome）+ PositiveOnly（v5 前叫 NoNegativeExperience）+ SR-J-Stop**。
其中 2×2 跑 seed 42/43/44，其余 4 个跑 seed 42（`configs/queue.json` 的 `T5-s43-*` / `T5-s44-*` 只覆盖 2×2 四格，与此一致）。

### 3.2 【重建注 R2】"两个代表模型" = GLM4-9B + Qwen3-8B

v5 未点名（§8 Phase 1c「2 消融模型」、§6 D9「两个代表模型」、Phase 8「2 模型」）。依据：

- **用户原始方案 §2.1（逐字，transcript 行 341；原文行尾为 CRLF）**：
  > 消融及补充实验只使用两个代表模型：
  >
  > * GLM4-9B
  > * Qwen3-8B

  同节列出五模型主表名单：GLM4-9B、Llama3.1-8B、Qwen3-4B、Qwen3-8B、Qwen3-32B。
- **v3 方案 Phase 10 行（逐字）**：「人工盲评材料导出（≤200 对，GLM4+Qwen8B，四任务）」。

**执行期影响**：Phase 5 曾一度计划把 glm4-9b 换成 qwen3-4b（因 glm4-9b 无法加载），该替换**已于 2026-09-12 撤回**，
故 3.2 的定义继续有效（见 §4.6）。

---

## 4. 执行期变更记录

> 本节列出自 2026-09-11 08:00 方案 v5 获批之后，**实际发生并被授权或记录**的每一项偏离。
> 每条给出：方案原文 → 实际做法 → 理由 → 授权/记录来源。来源优先级：用户原话 > `configs/frozen.json` > `reports/progress.md` > 代码注释。
> 本节的"方案原文"若来自 v5，用 §1 的逐字正文为准；若来自用户原始方案或更早稿，会注明行号。

### 4.1 y₀ 复用：复现门禁 77/80 判定 FAIL 之后仍继续使用 stored 初稿

- **方案原文（v5 §2.6）**：「**Phase 0 复现门禁**：每 model-task 重跑 5 条 Direct-Zero 与已存 `final_output` 精确比对；**不一致则放弃 `y₀` 复用**。」
- **方案原文（v5 §6 D2）**：「**D2** seed 42 的 `y₀` 用已存 Direct-Zero（经门禁），**43/44 重新生成**。」
- **实际结果**：门禁 `FAIL`——`reports/repro_gate.json`：`"matched": 77, "total": 80, "verdict": "FAIL"`；
  16/20 个 model-task 完成（qwen3-32b 因 OOM 未完成），13 个 5/5 相同、3 个 4/5，**3 处不一致全部落在 `wmt19_zh_en`**（llama3.1-8b / qwen3-4b / qwen3-8b）。
- **实际做法（偏离）**：**全部模型、全部 seed 继续 `draft_source="stored"`**；`progress.md` B2：
  「**已与负责人确认：偏离字面规定，全部模型继续使用 `draft_source="stored"`。**」
- **理由（原文）**：`frozen.json.y0_reuse.reason`：「reusing the stored text is the only way to guarantee a byte-identical y0 across arms; regenerating is subject to the same nondeterminism and would align worse with the published Direct-Zero baseline」；
  `progress.md` B2 补充：「若改为重新生成，新初稿同样受上述非确定性影响（~4–20% 的样本会变），既不能更接近 baseline，又丧失了跨臂同一性——反而更差。」
- **授权来源**：`frozen.json.y0_reuse`（`"authorised": true`、`"user_decision": "keep stored drafts despite the 77/80 gate result"`）；
  `progress.md` B2 记「已与负责人确认」。用户在收到该偏离汇报后的原话是第 1674 行「继续继续」与第 1776 行「继续执行你的计划并实时监控执行情况」——**没有任何用户消息逐字讨论过 stored 初稿本身**，此点如实标注。
- **论文必须报告**（`frozen.json.y0_reuse.must_report`）：「77/80 = 96.25% exact reproduction, all 3 mismatches on wmt19_zh_en, plus the same-seed repeat probe」；
  探针证据 `probe_determinism.py`：同进程同 prompt 同 seed 重复 5 次，item 3 出现 **distinct=2/5**，而已存输出 **5/5 可达** → vLLM 在 T=0.1 下非逐位可复现，不是配置错误。

### 4.2 controller prompt：替换为 defect-based 判据（含一次已披露的测试集 STOP 诊断）

- **首版 prompt（原文，`progress.md` §三）**：
  > - Refining an already-good answer often makes it worse. Stop when the answer already satisfies the task.
  > - Only ask for a revision when you can name a concrete, specific defect.
- **问题**：第一句**把论文结论写进了提示词**，controller 在 **98.3%** 的决策上 STOP；`Full` 与 `NoExperience` 会行为趋同，
  `How/When` 对比变成空转。由此 **qwen3-8b / wmt19_en_zh 旧 prompt 下产出的首批 522 条被判污染并作废重跑**（`progress.md` §二）。
- **替换后的判据（原文，`frozen.json.controller_prompt.criterion` / `core/controller.py:51-54`）**：
  > First try to state one concrete, specific defect in the current answer.
  > If you cannot name such a defect, choose STOP.
  > If you can name one, choose REFINE only when fixing it is likely to be worth another revision; otherwise choose STOP.
- **协议失误（已披露）**：`frozen.json.controller_prompt.protocol_note`：「an earlier STOP-rate diagnostic was run on the **TEST** split before the prompt was finalised; that was a protocol slip. The final tuning above used dev only, and the test split has not been used to select any method component.」
  `progress.md` §三同样如实记录。`frozen.json.freeze_discipline.already_disclosed_slip` 要求它**必须留在论文 limitations**。
- **dev 效果（`frozen.json.controller_prompt.dev_effect`，与 `reports/stop_rate_check_qwen3-8b_wmt19_en_zh.json` 一致）**：
  `none` stop_rate=1.0 / meanΔQ=0.0；`full` 0.683 / +5.59；`outcome_hidden` 0.70 / +3.91。
- **经验库不受影响（判定依据，原文）**：`frozen.json.controller_prompt.experience_libraries_still_valid`：
  「libraries were built with stop_mode='fixed' (forced REFINE), so the removed STOP bias was inert during bootstrap; both old and new guidance ask for a concrete defect, so the stored interventions remain representative」。
- **授权来源**：`frozen.json.controller_prompt`（`frozen_at: 2026-09-12`，`tuned_on: dev split only`）；`progress.md` §三。

### 4.3 `max_model_len` 8192 → 4096（回到 baseline 配置）

- **方案原文**：v5 未写 context 长度；**用户原始方案 §12** 写「统一 context：8192 tokens」。
- **实际做法（偏离）**：`run_experiment.py` / `build_experience.py` 初版用 8192；发现 baseline 全部使用
  `--vllm-max-model-len 4096`（在 `results_full_main_qwen35_vllm/logs/*__attempt0.log` 中逐一核对）后，
  新增 `core.MAX_MODEL_LEN = 4096` 并统一引用。
- **理由（原文，`progress.md` 6.1）**：「这属于**静默偏离 baseline 配置**——正是"模型一致性"约束要防的问题，而且把 KV cache profiling 的开销放大了一倍。」
  已生成的经验库无需重建：「我们的 prompt 约 600–800 token，远低于 4096，**不影响已生成内容**」。
- **遗留未闭合项（原文，`frozen.json.operational_constants.max_model_len_note`）**：
  「NOTE: `check_stop_rates.py` and `calibrate_judge.py` still hardcode 8192 and must be aligned before either is re-run.」
  （核对属实：`check_stop_rates.py:26` 仍写 `max_model_len=8192`。）
- **授权来源**：`frozen.json.operational_constants`（`recorded_at 2026-09-12T05:10-04:00`）；`progress.md` 6.1。

### 4.4 batch size 1 → 32（锁步批处理，5.6× 加速）

- **方案原文**：v5 未规定并发度；成本锚点是「Full ≈ 13 calls/样本 ≈ 13.4 s」，预算是按串行延迟外推的（v5 §8 锚点）。
- **根因（原文，`progress.md` 6.3）**：「`baseline/core/llm.py:149` 是 `self.model.generate([text_input], params)`——**每次只传 1 个 prompt**，而我们的 pipeline 又逐样本串行，所以引擎一直跑在 **batch size = 1**」。
- **实际做法**：新增 `core/batch_llm.py`（每条请求独立 `SamplingParams`，保留逐样本种子）与 `core/batched_pipeline.py`（轮次锁步批处理），
  `run_experiment.py` 新增 `--batch-size`（默认 **32**）；实测 3.0 s/样本 → **0.53 s/样本（5.6×）**。
- **连带后果（重要）**：`progress.md` 6.3：「由于提速显著，Phase 4 已**全部作废重跑为批处理版本**，以保证全实验方法学一致」。
  另有 04:09 误杀生产引擎事故（清理脚本未限卡号，杀掉 GPU 0/2 引擎，coedit_gec 已写 64 条丢失），已新增 `free_gpu.sh` 按 GPU uuid 过滤（`progress.md` 6.4）。
- **授权来源**：用户第 2490 行「…我希望你可以把我们的batch数或者并发数开的高一点，现在显存还很多，你不拉满有点浪费，而且推理引擎用的是vllm吗？」；
  `frozen.json.operational_constants.batch_size=32`（`batch_size_note`：engine 日志实测 controller 32/32、judge 至多 42/42）。

### 4.5 GPU 卡授权历史（四张卡全部放开，取代验收标准 #4）

- **方案原文（v5 §2 末行）**：「**GPU 白名单**：`--gpus 2,3`，必须在**未设置 `CUDA_VISIBLE_DEVICES`** 的 shell 中启动调度器。」
- **方案原文（v5 验收 #4）**：「**GPU 白名单**：所有运行 metadata `CUDA_VISIBLE_DEVICES ∈ {2,3}`；0/1 号卡无新进程。」
- **授权历史（时间顺序，`frozen.json.gpu_policy.history` + 用户原话）**：
  1. 最初由操作员限制为 GPU 2、3 —— 用户第 534 行：「为什么还需要编码模型？请你只使用后两张卡进行实验，及2，3号卡，不要用0，1号卡」
  2. 放宽为 0–3 中任意两张、同时最多两张 —— 用户第 2112 行：「…我现在给你权限，你可以占据0，1，2，3四张卡中的任意两张，但最多两张卡，卡号不限制你」
  3. 第 2194 行：「我也允许你用一下卡2加一下速」
  4. 第 2689/2690 行：「把卡1也用上」→ **四张卡全部授权**
- **生效规则（原文，`frozen.json.gpu_policy.effective_rule`）**：「all four cards 0,1,2,3 may be used concurrently for this run」；
  `supersedes`：「plan acceptance criterion #4 (CUDA_VISIBLE_DEVICES in {2,3}) and the earlier 'at most two cards' rule」。
- **代码常量**：`core/__init__.py`：`ALLOWED_GPUS = ("0","1","2","3")`、`MAX_GPUS = 4`
  （注意 `ensure_gpu_whitelist()` 的 docstring 仍残留旧措辞「any two of GPUs 0-3 may be occupied at a time」，与常量不符——**代码注释层面的未清理残留**，如实记录）。
- **另一条调度纪律（原文，`frozen.json.gpu_policy.scheduling_rule`）**：「when a job finishes the next must start immediately with no idle gap, otherwise the cards get taken by other tenants」；
  `progress.md` §六记录了实现方式（`supervisor.py` 常驻 + `dispatcher.py` 无空档派发）。
- **记录来源**：`frozen.json.gpu_policy`（`updated_at 2026-09-12T05:10-04:00`，`recorded_by`：「operator, after the compliance audit flagged the conflict between frozen.json and the live state」）。

### 4.6 Phase 5 模型替换：**已撤回**（glm4-9b + qwen3-8b 的原方案继续有效）

- **方案原文（用户原始方案 §2.1）**：「消融及补充实验只使用两个代表模型：GLM4-9B、Qwen3-8B」；v5 §8 Phase 5「8 配置 × 2 模型」。
- **中途的替换计划（`frozen.json.phase5_ablation_substitution`）**：`"interim_substitution_that_was_planned": ["qwen3-8b", "qwen3-4b"]`，
  起因是 glm4-9b 无法完成 vLLM engine init。
- **现状：OBSOLETE / WITHDRAWN 2026-09-12**。理由（原文）：「the substitution existed only because glm4-9b could not complete vLLM engine init. Root cause was one transiently unreadable NFS shard (model-00002-of-00004.safetensors); a sha256-verified local copy at `/home/ymb/glm_local/model` loads in ~3 s and is running the Phase 4 main experiment… The plan's original two models are therefore used, and **qwen3-4b is NOT substituted**。」
  `"applied_to": "nothing -- no Phase 5 arm was ever run on qwen3-4b, so withdrawing the substitution costs no work"`、`"design_unchanged": true`。
- **残留不一致（如实报告）**：`run_ablation_chain.sh:11-14` 的注释仍写「Model substitution: … qwen3-8b is kept and qwen3-4b is substituted. Recorded as a deviation.」——**该注释已过期**，与 `frozen.json` 的撤回声明矛盾；实际队列 `configs/queue.json` 的 Phase 5 作业已恢复为 **glm4-9b + qwen3-8b**（如 `T1-sr_j_fixed-glm4-9b`、`T2-no_experience-glm4-9b`、`T2-random_retrieve-qwen3-8b` 等）。
- **seed 策略不变**（原文，`frozen.json.phase5_ablation_substitution.seed_policy_note`）：「the plan has the 2x2 at 3 seeds (42/43/44) and the remaining arms at 1 seed; coverage is established at seed 42 first so partial progress stays informative」；`queue.json` 的 `T5-s43-*` / `T5-s44-*` 只对 2×2 四格排了 43/44。

### 4.7 GLM 模型目录覆盖（v5 §2.3 规定路径已不可读）

- **方案原文（v5 §2.3）**：「**GLM 特例**：只用 baseline 本地 `/mnt/huawei/ymb/model/glm-4-9b/model`，不得用 icml 的 `THUDM/glm-4-9b-chat-hf`。」
- **实际做法（偏离）**：`core.MODEL_PATH_OVERRIDES = {"glm4-9b": "/home/ymb/glm_local/model"}`，`resolve_model_config()` 应用覆盖，覆盖路径不存在时报错而非静默回退。
- **理由（原文，`progress.md` 6.5/6.7）**：NFS 服务端 `model-00002-of-00004.safetensors` 在偏移 **2,769,682,432 B** 之后不可读，任何越过该点的读取永久阻塞（D 状态）；正确仓库是 **`zai-org/glm-4-9b-chat-hf` @ `8599336f`**（不是 `zai-org/glm-4-9b-chat`——那是 ChatGLM 架构）；
  4 个 shard 的 **LFS sha256 与本地 `manifest_sha256.json` 逐一完全一致**，`AutoConfig` 解析为 `GlmConfig`（hidden 4096 / 40 层 / bf16）。
- **等价性口径（原文，`core/__init__.py`）**：「The MODEL, dtype, tokenizer and engine version are unchanged; only the directory differs.」
- **授权来源**：`frozen.json.phase5_ablation_substitution.why_withdrawn`；`reports/glm4_nfs_root_cause.md`；`progress.md` 6.5–6.7。

### 4.8 qwen3-32b 阻塞（唯一未开工的 model-task 组合）

- **方案原文（v5 §8）**：Phase 3「**Initial Experience：32/model-task × 5 模型 × 4 任务**」、Phase 4「**Full-Static × 5 模型**」。
- **实际阻塞（原文，`progress.md` 附录 B3）**：
  > CUDA out of memory. GPU 0 (physical GPU 2) has 303.50 MiB free. Process 3749967 has 22.54 GiB in memory use.
  「qwen3-32b 需要 `0.86 × 79 GiB ≈ 68 GiB` 仅放权重。另一用户 `qxr` 在**全部 4 张卡**上都有任务，没有任何一张能腾出 68 GiB。」
- **处置（原文）**：「**已与负责人确认：不与 qxr 争抢，先推进其余模型；qwen3-32b 待其任务结束后补跑。** 因此当前 Phase 3/4 覆盖 4 个模型…这不改变方案的方法学，只影响完成顺序。」
  `progress.md` §五（Round 6 时点）仍记：「**qwen3-32b 仍被阻塞**：需约 68 GiB 连续显存，且其自身加载同样会遇到上述挂死。」
- **影响面**：`repro_gate.json` 缺 qwen3-32b 的 4 个 model-task（因此门禁分母是 80 而非 100）；初始经验库与 Phase 4 主表目前为 4 模型。
- **授权来源**：`progress.md` B3（"已与负责人确认"）、A3-4、§五；用户第 2570 行「glm一定要跑上好吗？即使等到最后也要跑一下，因为baseline里面含glm」（针对 glm，非 32b）。

### 4.9 其他已记录的执行期变更（非上述七项，但已写入 `frozen.json`/`progress.md`）

| # | 变更 | 内容与理由（原文摘要） | 来源 |
|---|---|---|---|
| A | **dev 初稿缓存 `CachedDraftSource`** | vLLM 在 T=0.1 下非逐位可复现，每轮 sweep 重新生成初稿导致起点漂移（同一 16 条 dev 的 corpus BLEU 两次分别 32.51 / 38.21）。改为"dev 初稿只生成一次并落盘，所有 sweep 点回放同一批初稿"。 | `frozen.json.draft_cache`；`progress.md` A4-2 |
| B | **dev 门禁的 random 对照判为无效** | `status: inconclusive_as_run`——「the random_retrieve arm uses adaptive stopping, so the controller stopped immediately (rounds=1.0-1.12, accept=0%) and never exercised retrieval」；真正的对照改由 **Phase 5 的 RandomRetrieve 核心消融**承担。 | `frozen.json.dev_gate_random_control`；`progress.md` A4-4 |
| C | **经验库 bootstrap 由 adaptive 改 forced REFINE** | adaptive 下 32×2=64 次尝试只产出 20 条经验（产出率 31%、positive 仅 2）；改 `stop_mode="fixed"` 后产出率 100%。**属于实现修复，不改方法定义**。 | `progress.md` B1-3、A4 附表 |
| D | **positive pool 预检触发 K 提升** | 按 v5 §4.4 执行：llama en_zh K=5、llama zh_en K=4、llama gigaword K=3、qwen3-8b zh_en K=3（GEC 对所有模型 K=8）；**K 只在库构建阶段按 model-task 全局确定一次**，不按臂调整。最终 16/16 model-task positive ≥8。 | v5 §4.4；`progress.md` §二、A4-3 |
| E | **judge tie 修复** | 「core/judge.py now returns 'tie' without engine calls when candidate == current byte-identically. Previously 84 rounds (81 with byte-identical reasons) were mislabelled 'uncertain' because a consistent slot preference was mapped through two contradictory arrangements.」该修复是两段文本的纯函数，可同样施加于已写 trace。 | `frozen.json.number_provenance.judge_tie_fix` |
| F | **批处理路径的三个真实缺陷修复（F1/F2/F3）** | 批处理一度丢失 controller 修复重试（系统性偏向 STOP）、丢失 judge 修复重试、`full_online` 静默丢失在线写入；修复后 Phase 4 重启。另 `aggregate.py` 增加完整性护栏（避免把 64 条半成品与 1000 条完整结果混算）。 | `progress.md` 6.x 与父会话对审计的回复（transcript 行 2995） |
| G | **`scores/live_runs.STALE-2026-09-12.csv` 作废** | 「describes runs that were overwritten by supervisor restarts while run_experiment.py still opened outputs with mode 'w'. Never cite it.」现 `run_experiment.py` 改为 resume（`_load_prior`/`_persist_merged`，config_hash 不匹配则不复用），并有 `tests/test_resume_equivalence.py` 证明与从零重跑等价。 | `frozen.json.number_provenance`；`scores/README_STALE.md` |
| H | **`progress.md` §6.8 的比率已被部分前缀污染** | 独立重算：llama3.1-8b/en_zh 55.6%、qwen3-8b/en_zh 44.6% 可复现；glm4-9b/en_zh 78.5% 是 32 条前缀（800 条时 68.3%）；llama zh_en 52.7%→57.2%、qwen3-8b zh_en 36.3%→37.9%。池化的 "7344 refines, 50.7% worse" 无法从现存产物复现。**规则：所有论文数字必须在 runs 完成后一次性重算。** | `frozen.json.number_provenance` |
| I | **冻结纪律（自查约束）** | 「Phase 7 stopping diagnostics have already been computed on TEST-split traces. Therefore NO method component may change from now on -- not the controller prompt, not the stop rule, not the acceptance rule, not alpha, not the retrieval function, not the experience-library builder.」 | `frozen.json.freeze_discipline` |

### 4.10 尚未闭合的执行期事项（供后续审计直接核对）

1. `check_stop_rates.py:26` 与 `calibrate_judge.py` 仍硬编码 `max_model_len=8192`（`frozen.json` 要求"再跑前必须对齐"）。
2. `core/__init__.py.ensure_gpu_whitelist()` docstring 仍写"任意两张卡"，与 `ALLOWED_GPUS/MAX_GPUS` 四卡常量不符。
3. `run_ablation_chain.sh` 仍保留 qwen3-4b 替换注释（§4.6 已撤回）。
4. v5 验收 #16（`PositiveOnly` 须附两臂检索相似度分布）与 #17（缩水项在论文显式标注）尚无产物。
5. `reports/progress.md` §6.8 的表格数字按 `frozen.json.number_provenance` 不得直接引用。
6. Phase 6 的脚本已于 05:02 出现（`run_scaling.py` / `bon_judge` arm / `tests/test_scaling.py`），但 `runs/scaling/` 无产物、队列无作业；
   Phase 9 的图件（`make_plots.py` 已存在）与 Phase 10 的实际导出同样未产出（见 §5 对应关系表，核对时点 05:20）。

---

## 5. 与代码的对应关系

> 时点说明：本表核对于 **2026-09-12 05:12–05:20 EDT（最终复核 05:20）**。仓库当时是活动状态（父 agent 与 supervisor 均在写盘），
> 核对期间先后出现了 `export_human_eval.py`（04:59）、`run_scaling.py` / `tests/test_scaling.py`（05:02）、`make_plots.py`（05:05）
> 等文件；`runs/**` 的行数也在持续增长（本表给出的是核对瞬间的值）。该挂载点的 mtime 存在偏移，不要用 mtime 判断先后。
> 表中"状态"以该时点我在磁盘上**实际读到**的内容为准，并给出证据文件。

### 5.1 阶段 → 脚本

| Phase（v5 §8） | 实现的脚本 / 模块 | 状态与证据 |
|---|---|---|
| **0** 环境/GPU 断言 + 复现门禁 + 6 万步动机表与 oracle-stop 上界 | `run_phase0.py`（环境断言、动机表、220 次 baseline 统一重算）、`repro_gate.py` + `run_repro_gate.sh`（门禁）、`probe_determinism.py`（同 seed 重复探针） | ⚠️ 大部分完成：`reports/phase0_env.json`、`scores/motivation_steps.csv`、`scores/baseline_rescored*.csv`、`reports/repro_gate.json`（77/80 FAIL → 见 §4.1）。**oracle-stop 上界未见实现**（全仓库 `grep -i oracle` 仅命中 `tests/test_accumulation_gaps.py` 的一句注释） |
| **1a** manifest（**过滤式**）+ 隔离断言 + 辅助切分 | `build_manifests.py`、`core/manifest.py` | ✅ 产物：`data/manifests/*.jsonl`（test 1000/1000/1000/100；aux initial 32 / dev 32 / accumulation 128）、`data/manifests/_summary.json`（含各 split sha256） |
| **1b** 依赖安装（sacrebleu/nltk/rouge_score） | 无脚本（一次性 `pip install`） | ✅ 记于 `progress.md` 与 `core/__init__.py` 注释 |
| **1c** dev 切分轨迹（2 消融模型 × 4 任务 × 32） | `run_dev_gate.sh`、`core/pipeline.py` | ✅ 产物：`runs/gate_alpha/*`、`runs/gate_warmup/*`、`scores/gate_runs.csv`、`scores/gate_bootstrap.csv` |
| **1d/1e** BM25 检索 + 归一化 | `core/bm25_fields.py`（per-query min-max 字段归一化、CJK 1–2 gram）、`core/retrieval_cache.py` | ✅ 实现存在；规范化落点见 `core/bm25_fields.py` |
| **1f** positive pool 预检（§4.4，≥8） | `build_experience.py` 的 precheck 路径 | ✅ 16/16 通过（`progress.md` §二；提升 K 的四个 model-task 见 §4.9-D） |
| **1g** α 选定并冻结（验收 #8） | `run_dev_gate.sh` + `core/`（α 传入 `RunConfig.alpha`） | ✅ `configs/frozen.json`（α=0.5）；证据 `runs/gate_alpha/fixed_rounds/seed42/*/glm4-9b/dev/a{0.0,0.5,1.0}/` |
| **1（不变量测试）** | `tests/test_invariants.py` | ✅ 存在（会话记录称 29 项断言通过；本重建未运行测试） |
| **2** 每 model-task 3 条真实 smoke | `run_experiment.py --limit 3` 路径 | ✅ 产物：`runs/smoke/`、`scores/pilot_*.csv`；判定器校准 `calibrate_judge.py` |
| **3** Initial Experience：32/model-task × 5 模型 × 4 任务 | `build_experience.py`；驱动 `run_build_all_models.sh`、`run_rebuild_*.sh` | ⚠️ **4 模型完成**（glm4-9b / llama3.1-8b / qwen3-4b / qwen3-8b × 4 任务 = 16 个库）；**qwen3-32b 缺**（§4.8） |
| **4** Full-Static × 5 模型（主表，1 seed）+ Full-Online × 2 模型 × 3 seed | `run_experiment.py`（arm `full_static` / `full_online`）、`core/batched_pipeline.py`、`core/batch_llm.py`、`supervisor.py`、`dispatcher.py`、`watchdog_run.sh`、`run_main.sh`、`run_main_chain.sh` | 🔄 运行中。05:20 证据：`runs/main/full_static/seed42/**` 共 **12** 个任务级 jsonl（目标 20 = 5 模型 × 4 任务）——`wmt19_en_zh` 4 模型全 **1000/1000**；`wmt19_zh_en` llama/qwen3-8b **1000**、glm4-9b **832**、qwen3-4b **896**（仍在增长）；`coedit_gec`/`gigaword` 仅 llama3.1-8b 与 qwen3-8b（**1000** / **100**）；`full_online` **0 个**；`configs/queue.json` 中 glm4-9b/qwen3-4b 的 full_static 为 running，full_online 为 pending |
| **5** 核心消融：8 配置 × 2 模型（2×2 跑 3 seed，其余 1 seed） | `run_experiment.py` 的 arm 注册表（见 §3.1）、`run_ablation_chain.sh` | 🔄 部分。`runs/ablation/` 05:20 共 **8** 个 jsonl（目标 16 = 8 配置 × 2 模型，现覆盖 3 个 arm-model 组合，全部为 qwen3-8b）：`no_experience` × 4 任务全部完成（1000/1000/1000/100）、`outcome_hidden`（en_zh 1000、zh_en 928 增长中）、`sr_j_fixed`（en_zh 1000、zh_en 32 增长中）；其余 arm/模型为 pending |
| **6** Scaling：K∈{1,2,3 复用}+K=5；BoN N∈{1,2,4 复用}+N=8 | `run_scaling.py`（Phase 6 驱动：K∈{1,2,3,5} 中 K≤3 由 K=3 锚点前缀截断复用；BoN-J N∈{1,2,4,8} 中 N=1 复用 `full_static`）、`run_experiment.py` 的 `bon_judge` arm / `--n-candidates`（`BONJ_SELECTION_RULE`：同一 pairwise judge 上的 round-robin Copeland 选择）、`tests/test_scaling.py` | ⚠️ **脚本已于 05:02 出现，但无任何运行产物**：`runs/scaling/` 为空，`configs/queue.json` 无 scaling/BoN 作业。（本审计开始时的 05:12 快照里这些脚本尚不存在，故本行结论以 05:20 为准） |
| **7** Stopping Diagnostics：≤100 个 STOP 状态强制补一次 refinement | `run_stopping_diagnostics.py`（GPU-free 的不必要修订率 + `--counterfactual` 强制补一轮）、`tests/test_stopping_counterfactual.py` | ⚠️ **GPU-free 半边已产出**：`scores/stopping_diagnostics.csv`、`stopping_diagnostics_all_prerestart.csv`、`stopping_diagnostics_main_prerestart.csv`；**`--counterfactual`（过早停止率）未跑**（`runs/stopping/` 为空）。另有 `check_stop_rates.py`（dev 停止率核对，脚本仍硬编码 8192，见 §4.3/§4.10） |
| **8** Accumulation：5 snapshot × 2 任务 × 3 order × 3 method × 2 模型（8B 档，净 152 次评价） | `run_accumulation.py`、`tests/test_accumulation_loop.py`、`tests/test_accumulation_gaps.py` | ⚠️ 脚本已交付并通过 stub 引擎端到端验证；**生成模式未运行**（`runs/accumulation/` 为空；`progress.md` §6.8 末句："生成模式因四卡满载尚未运行"） |
| **9** 统一重算评分、bootstrap、绘图 | `aggregate.py`（2000 次 paired bootstrap、Holm 校正、写 `scores/`）、`make_plots.py`（05:05 出现的绘图层：fig1 初始/最终质量、fig2 Full−最强无经验臂、fig3 质量–compute 曲线）、`core/scoring.py`、`core/stats.py` | ⚠️ **部分**：`scores/all_runs.csv`、`gate_bootstrap.csv`、`pilot_bootstrap.csv`、`check_0500_runs.csv`、`check_0515_runs.csv`、`check_0515_bootstrap.csv` 已产出；`plots/` 目前**只有聚合中间产物 `_corpus_metrics.json`，没有任何图件**（全仓库无 `*.png/*.pdf/*.svg`） |
| **10** 人工盲评材料导出（≤200 对） | `export_human_eval.py`、`tests/test_human_eval_export.py` | ⚠️ **脚本已存在（04:59）但未实际导出**：`human_eval/` 为空 |

### 5.2 实现端与 §3 两条重建注的对应

- Phase 5 的 8 个配置 → 见 **§3.1**（含 v1 D7/D8 逐字依据与 `run_experiment.py` arm 表）。
- Phase 1c/5/D9 的"两个代表模型" → 见 **§3.2**（GLM4-9B + Qwen3-8B；被撤回的替换见 §4.6）。
- 五模型主表名单 → 见 **§3.2**（GLM4-9B、Llama3.1-8B、Qwen3-4B、Qwen3-8B、Qwen3-32B；其中 qwen3-32b 仍被阻塞，见 §4.8）。

### 5.3 v5 验收标准 → 当前证据（可机械核对）

| v5 验收项 | 现状 |
|---|---|
| #1 无编码模型 | ✅ `frozen.json.encoders.used="none"`；`core/bm25_fields.py` 自建 BM25 |
| #2 模型一致性 / `baseline/` 无改动 | ✅ 见 `core/__init__.py` 注释与 PINNED_VERSIONS；GLM 路径例外见 §4.7 |
| #3 复现门禁通过或已按规则放弃 y₀ | ⚠️ 门禁 **FAIL**（77/80），已按 §4.1 授权继续 stored |
| #4 GPU ∈ {2,3}、0/1 无新进程 | ❌ 已被 §4.5 的授权取代（现为 0–3 四卡） |
| #5 隔离断言 / 排除 icml KB 0–999 / Gigaword 仅 val+test | ✅ `build_manifests.py` + `tests/test_invariants.py`；manifest 由过滤操作生成 |
| #6 按臂对不变量 | ✅ 实现于 `core/retrieval_cache.py` + `tests/test_invariants.py`；跨臂一致性随 trace 记录 |
| #7 归一化后相似度 ∈ [0,1] | ✅ `core/bm25_fields.py` per-query min-max |
| #8 α 由预注册 dev 指标选定并冻结 | ✅ `frozen.json`（α=0.5；并列取大；不要求单调） |
| #9 positive pool ≥8 / K 全局一次 | ✅ 16/16；K 提升记录见 §4.9-D |
| #10 渲染器白名单（`delta_offline`/`reference_text` 不入 prompt） | ✅ `tests/test_invariants.py` 断言 |
| #11 smoke 4 任务各 3 条 | ✅ `runs/smoke/`（会话记录称通过） |
| #12 主表 2000 次 paired bootstrap、可溯源 | ⚠️ `aggregate.py --bootstrap 2000` 已实现；Phase 4 未完成，主表尚不可出 |
| #13 成本表含 controller/judge 分解 | ⚠️ trace 已记 token/调用；聚合表的分解未产出 |
| #14 2×2 四格 y₀ 完全一致（哈希校验） | ⚠️ 由 stored 初稿在构造上保证；尚无汇总产物 |
| #15 3 seed 方差或说明 | ❌ 未到（只有 seed 42 的部分结果） |
| #16 `PositiveOnly` 两臂检索相似度分布 | ❌ 无产物 |
| #17 缩水项在论文显式标注 | ❌ 无产物（论文未写） |

### 5.4 与 `configs/frozen.json` 的一致性核对：冲突清单（逐字引文）

**冲突 1 —— dev 条数：v5 §1 写 40，v5 §8 与实现为 32。**
- v5 §1（逐字）：「Gigaword 分配：initial **32** / dev 40 / accumulation 128；任何一行被剔除即报错中止。」
- v5 §8 Phase 1c（逐字）：「| **1c** | dev 切分轨迹（2 消融模型 × 4 任务 × **32**） | 有 | 0.5 GPU·h | — |」
- `configs/frozen.json.controller_prompt.tuned_on`（逐字）：「dev split only (**40 dev items**, qwen3-8b/wmt19_en_zh)」
- `reports/progress.md` §一（逐字）：「在 **dev 辅助切分**（**40 条**，qwen3-8b / wmt19_en_zh）上：」
- 磁盘事实：`data/manifests/*__dev.jsonl` **每个都是 32 行**；`core/__init__.py` `AUX_SIZES = {"initial": 32, "dev": 32, "accumulation": 128}`；
  `check_stop_rates.py` 默认 `N=40`，但它对 32 行的 dev manifest 做 `[:40]` 切片，所以实际只用到 **32** 条
  （证据：`reports/stop_rate_check_qwen3-8b_wmt19_en_zh.json` 中 `none` 臂 `stop=32`）。
- **判定**：40 这个数字在 v5 内部（§1 对 Gigaword）与 `frozen.json`/`progress.md` 的表述中都出现，但**没有任何 40 行的 dev 产物**。
  实际生效值是 32。此处不作取舍，如实并列；后续若引用"40 条 dev"必须改写为 32 并说明。

**冲突 2 —— GPU 白名单。**
- v5 §2 / 验收 #4（逐字）：「`--gpus 2,3`」「所有运行 metadata `CUDA_VISIBLE_DEVICES ∈ {2,3}`；0/1 号卡无新进程。」
- `frozen.json.gpu_policy`（逐字）：`"effective_rule": "all four cards 0,1,2,3 may be used concurrently for this run"`，且 `"supersedes": "plan acceptance criterion #4 (CUDA_VISIBLE_DEVICES in {2,3}) and the earlier 'at most two cards' rule"`。
- **判定**：**验收 #4 已被显式取代**，取代过程有 4 步用户授权（§4.5）。审计时应以 `frozen.json` 为准，并把它记为"计划外但已授权"。

**冲突 3 —— seed 43/44 的初稿来源。**
- v5 §6 D2（逐字）：「**D2** seed 42 的 `y₀` 用已存 Direct-Zero（经门禁），**43/44 重新生成**。」
- `frozen.json.operational_constants.draft_source`（逐字）：`"stored"`；`run_experiment.py` 的 `--draft-source` 默认即 `"stored"`（choices: stored/generate/cached）。
- **判定**：实际全程使用 stored（含种子 43/44 的 2×2 计划），与 D2 的字面规定不同；该偏离由 §4.1 的 y₀ 复用决定覆盖（同一理由：跨臂逐字同一的 y₀），但 **D2 的后半句没有被单独撤销**——这是一处应当补记的文书缺口。

**冲突 4 —— GLM 模型目录。**
- v5 §2.3（逐字）：「只用 baseline 本地 `/mnt/huawei/ymb/model/glm-4-9b/model`，不得用 icml 的 `THUDM/glm-4-9b-chat-hf`。」
- `core/__init__.py`（逐字）：`MODEL_PATH_OVERRIDES = {"glm4-9b": "/home/ymb/glm_local/model"}`。
- **判定**：**不是违反**（没有使用 icml 的 checkpoint；权重 sha256 与 baseline manifest 逐一相同），但路径确实与 v5 字面不同，已由 §4.7 记录。

### 5.5 结论

- 方案 v5 的文本已可被任何审计者从本文件 §1 逐字核对；**所有执行期偏离都收敛到 §4.1–§4.8 八项 + §4.9 的九条记录**，其中 4 项改变了
  v5 的字面要求（y₀ 复用、controller prompt、max_model_len、GPU 白名单），2 项改变了实现细节（batch 32、GLM 路径），
  1 项为被撤回的中途计划（Phase 5 模型替换），1 项为环境阻塞（qwen3-32b），9 条为记录性变更（§4.9）。
- **尚未兑现的方案交付物（05:20 核对）**：Phase 0 的 oracle-stop 上界未见实现；Phase 6 的脚本已在核对窗口内出现但
  **没有任何运行产物**（`runs/scaling/` 为空）；Phase 7 的反事实分支、Phase 8 的生成模式、Phase 9 的图件、Phase 10 的实际导出
  **均未产出**；Phase 4 主表（12/20 个 model-task 文件）与 Phase 5 消融（3/16 个 arm-model 组合）仍在运行。
- **本文件只做重建与对照，不修改任何既有文件**；重建者未运行任何 GPU 作业。
