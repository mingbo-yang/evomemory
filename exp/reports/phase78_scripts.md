# Phase 7/8 脚本：停止诊断与经验累积

作者：子代理（Harness），2026-09-12 04:50 EDT（数字快照 04:47:55；trace 仍在增长，
每次重跑该命令都会刷新 `scores/stopping_diagnostics.csv`）。
交付物：`run_stopping_diagnostics.py`（Phase 7）、`run_accumulation.py`（Phase 8），
外加两个 GPU-free 自检脚本 `tests/test_stopping_counterfactual.py`、
`tests/test_accumulation_loop.py`。

## TL;DR（中文）

- 两个脚本均已写好，`--help` 与 `--dry-run` 全部通过，且在**不触碰任何 GPU** 的前提下
  完成了尽可能多的验证（详见第 4 节）。
- **(A) 默认模式已实跑 `runs/main`**：不必要的修订率（REFINE 后离线指标反而变差的比例）
  在 wmt19_en_zh 上是 **llama3.1-8b 55.6%（2693 次修订中 1496 次变差）**、
  **qwen3-8b 44.6%（1113 次中 496 次）**、**glm4-9b 78.5%（93 次中 73 次）**；
  wmt19_zh_en 上是 **llama3.1-8b 52.7%**、**qwen3-8b 36.3%**。
  若把"没变好"（变差或完全没变）都算作浪费，则各为 **75.0% / 69.3% / 92.5% / 76.1% / 80.4%**。
  这与 motivation 表的核心结论一致：
  **在强制/自适应修订下，多数修订轮次在离线指标上是负收益。**
- **`runs/main` 尚未全部完成**：截至 04:47，`wmt19_en_zh` 的两个模型已跑满 1000 条，
  其余仍在追写；我 04:2x 抓到的 7 文件 / 4536 样本快照随后被生产链重启删除
  （04:19–04:33），该快照的 CSV 已保留为 `scores/stopping_diagnostics_main_prerestart.csv`。
- **(A) 的 `--counterfactual`（过早停止率）没有在真实 GPU 上跑**（0/1/2/3 号卡都在跑生产任务），
  但我用 stub engine 把该路径**端到端跑通并断言**了关键不变量（refine seed 必须等于
  `call_seed(sample, round, "refine")`、judge 必须是双序 judge、指标必须来自共享 `Scorer`）。
- **(B) 的生成模式没有跑**（需要 GPU）；`--dry-run` 与 stub 编排测试（真实 manifest、
  真实经验库、真实快照/评估循环，仅文本来自 stub）全部通过。
- 两条需要你知道的事故/发现写在**第 6 节**：并发编辑 `core/batched_pipeline.py` 曾引入的
  `latency_s` 崩溃 bug（我复现后立即上报，主代理随即修复，现已确认修好），以及我一次误把
  `--gpu 0,1,2,3` 传给策略检查、导致 glm4-9b 在 GPU 0 上尝试加载后**初始化失败即退出**
  （无残留进程，生产任务未受影响）。

---

## 1. (A) `run_stopping_diagnostics.py` — Phase 7: stopping diagnostics

### 1.1 两个方向的定义

| 方向 | 定义 | 是否需要生成 |
|---|---|---|
| **Unnecessary refinement rate** | 所有 REFINE 决策中，候选答案在**离线参考指标**下比当时的 current answer **更差**的比例（等价于 trace 里的 `delta_offline < 0`） | 不需要（默认模式） |
| **Premature stop rate** | 对 STOP 状态**强制多修一轮**后，该轮会**改善**离线指标的比例 | 需要（`--counterfactual`，默认关闭） |

默认模式还同时报告：
- `same`（`delta_offline == 0`，白跑一轮）与 `useless_rate = worse + same`；
- `accept_rate`（双序 judge 接受率）与 judge 判定分布，用于对比"离线指标视角 vs judge 视角"；
- `useful_but_rejected`（离线变好但被 judge 拒绝）与 `worse_but_accepted`（离线变差但被 judge 接受）；
- `n_stop`（可用的 STOP 状态数），供反事实模式估计规模。

`--recompute` 会用**共享 `Scorer`** 从 `(reference, candidate)` 重算 `metric_offline` 与
`delta_offline`（current answer 由 trace 的 `accepted` 标志精确回放），并输出与 trace 中
存储值的最大偏差，作为一致性检查。不引入任何新指标。

### 1.2 CLI

```bash
# 默认（GPU-free）：扫描 runs/ 下所有 trace
python run_stopping_diagnostics.py
python run_stopping_diagnostics.py --runs runs/main
python run_stopping_diagnostics.py --runs runs/main --models qwen3-8b --tasks wmt19_en_zh
python run_stopping_diagnostics.py --runs runs --arms full_static --limit 200
python run_stopping_diagnostics.py --runs runs/main --recompute      # Scorer 一致性检查
python run_stopping_diagnostics.py --runs runs/main --counterfactual --dry-run   # 只打印计划

# 反事实（需要一张空闲卡；默认上限 100 个 STOP 状态）
python run_stopping_diagnostics.py --runs runs/main --counterfactual \
    --max-stop-states 100 --gpu 2 --sampling-seed 0
```

输出：`scores/stopping_diagnostics.csv`（`scope=run` 逐文件、`scope=arm` 逐臂汇总、
`scope=task_model` 逐 task×model 汇总；率永远是"重新计算"而不是"对率取平均"），
反事实模式额外写 `scores/stopping_diagnostics_counterfactual.csv`。

### 1.3 反事实模式如何保证可比性（全部 import，无复制）

1. STOP 状态定位：`(task, model, arm)` 分组后**轮转采样**（保证 cap 覆盖每个 run 而不是
   把第一个 run 抽满），每组的桶用 `--sampling-seed` 派生的种子打乱。
2. 状态复原：`reconstruct_current()` 按 trace 的 `accepted` 标志回放，得到该轮 controller
   真正看到的 current answer；经验块用该轮**实际检索到的 `exp_ids`** 重新渲染
   （`render_experience_block`，`include_outcome` 由该 run 的 `experience_mode` 决定），
   因此不重新检索、不引入新的状态漂移。
3. 指令：`Controller.decide(..., stop_mode="fixed")` —— 与 pipeline 内同一份 helper，
   只是把 STOP 从 schema 里拿掉；seed salt 用 `"counterfactual"`，与真实决策的 seed 区分。
4. 生成：`build_refine_prompt`（`core.pipeline`），seed = `call_seed(sample_id, round, "refine")`，
   max_tokens/temperature/top_p 取自 `Pipeline.budget`，与 pipeline 若真的继续时的调用**逐参数一致**。
5. 评判与打分：`PairwiseJudge.judge`（同一确定性 A/B 顺序与 seed）+ 共享 `Scorer`。
6. 统计：`premature = int(delta > 0)`；按 task×model 汇总并给出均值 delta 的 95% CI
   （`core.stats.paired_bootstrap` 对 0 基线）。

## 2. (B) `run_accumulation.py` — Phase 8: experience accumulation

### 2.1 设计

- **流**：`data/manifests/<task>__accumulation.jsonl`（128 条/model-task，`--limit` 可截断）。
- **评估集**：该 task 的 **TEST manifest** 前 `--eval-limit`（默认 200；gigaword 只有 100）。
  评估用 `draft_source="stored"` + `BatchedPipeline`，所以每个 checkpoint/条件的 y0
  逐字节相同，是严格配对的比较。
- **checkpoint**：`0, chunk, 2*chunk, …, n`（默认 chunk=32 → 0/32/64/96/128）。
  每个 checkpoint：先把当前库快照落盘（`<out-dir>/<model>/<task>/order<k>/library_c####.jsonl`），
  再评估，**然后**才处理下一段流。流的每条：跑正常自适应循环 → `transitions_to_experiences`
  → 追加进库并重建 retriever（与 `full_online` 臂完全相同的在线更新路径）。
- **两个条件**：`dynamic`（已累积的库）vs `frozen`（只有 initial 库）。
  frozen 与流的顺序无关，因此默认每个 model-task 只测一次并复用（`--no-reuse-frozen` 可关掉）。
- **顺序方差**：`--orders N`，order 0 = manifest 原序，1..N-1 = 种子化打乱；
  逐 (model, task, checkpoint) 用 `core.stats.order_variance` 汇总并写
  `scores/accumulation_order_variance.json`。
- **两条报告曲线**：(i) 固定算力下的质量（同一评估集、同一 y0、同一 max_rounds/batch，
  每个 checkpoint 的 corpus 指标、逐样本均值、以及 dynamic 相对 frozen 的配对 bootstrap
  delta/CI/p）；(ii) 经验增长下的平均修订轮数、tokens/sample、calls/sample、接受率。
- **评估绝不写经验库**：评估只用快照（`cache=None`，不碰 `experience/indexes`），
  流里的新经验只追加到内存库并写进 `out-dir` 的快照文件。

### 2.2 CLI

```bash
# 校验 setup 并打印计划（不加载模型、不碰 GPU）
python run_accumulation.py --dry-run
python run_accumulation.py --dry-run --models qwen3-8b --tasks wmt19_en_zh --orders 3

# 真实运行（需要一张空闲卡）
python run_accumulation.py --gpu 2 --models qwen3-8b --tasks wmt19_en_zh \
    --orders 3 --eval-limit 200 --chunk-size 32 --batch-size 32
python run_accumulation.py --gpu 2 --models all --tasks all --orders 3 \
    --stream-draft-cache data/draft_cache/accumulation.jsonl   # 流草稿只生成一次，跨 order 回放
```

输出：`runs/accumulation/<model>/<task>/order<k>/{library_c####.jsonl, stream.{jsonl,csv},
eval_c####_{dynamic}.jsonl}`、`runs/accumulation/<model>/<task>/frozen/eval_c0000_frozen.jsonl`、
`scores/accumulation_curve.csv`、`scores/accumulation_order_variance.json`。

## 3. 验证（全部 GPU-free）

| 检查 | 结果 |
|---|---|
| 两个脚本 `--help` | 通过 |
| (A) `--dry-run`（含 `--counterfactual --dry-run`） | 通过，打印 STOP 状态可用量与调用量估计，不加载模型 |
| (B) `--dry-run`（全模型/单模型、`--orders 3`） | 通过；含 4 个 task 的 `check_isolation`（test ∩ aux = ∅ 全部 ok），并警告 qwen3-32b 无 initial 库 |
| GPU 预算检查（`--gpu 7`） | (A)/(B) 均 `REFUSING: ... outside ('0','1','2','3')`，退出码 2，未加载模型 |
| (A) 默认模式实跑 `runs/main` 与整个 `runs/` | 通过（数字见第 4 节） |
| (A) `--recompute` 一致性 | **max |重算 − 存储| = 0.000e+00**，覆盖 7344 轮（重启前）/ 1402 轮（当前） |
| (A) 反事实路径（stub engine） | `tests/test_stopping_counterfactual.py`：20/20 PASS |
| (B) 编排路径（stub engine + torch stub） | `tests/test_accumulation_loop.py`：14/14 PASS |
| 经验库未被评估写入 | 自检中对 `experience/{initial,snapshots}` 做哈希前后比对，通过 |
| 未触碰 GPU | 自检进程内 `torch` 被替换为 no-op 模块；未 import vLLM |

`tests/test_stopping_counterfactual.py` 的断言包括：refine 调用恰好一次且 seed 等于
`call_seed(sample, round, "refine")`；judge 双序一致并给出 `better`；`premature` 与
`cf_delta` 自洽；指令来自 controller；`n_missing_experiences == 0`（库里能找到该轮全部
`exp_ids`）；`reconstruct_current` 与 trace 回放一致；`current_metric` 等于
`Scorer.primary(reference, current)`。

## 4. (A) 在 `runs/main` 上实跑得到的真实数字

### 4.1 口径与注意事项

- `delta_offline` 是 **逐样本** 差值：对 wmt19 用 `Scorer.score_one` 的 **sentence-BLEU proxy**
  （`core/scoring.py` 明确说明它只用于 per-step 诊断，corpus BLEU 才是报告值），
  GEC 用 sentence GLEU，gigaword 用 ROUGE-1。所以 `meanΔ` 是"proxy 分"而非 corpus 分。
- **`runs/main` 尚未全部完成**：目标是每 task 1000 条；04:47:55 时 `wmt19_en_zh` 的
  llama3.1-8b 与 qwen3-8b 已各满 1000 条，`wmt19_zh_en`（llama 256 / qwen 512）与
  glm4-9b（32 条）仍在追写。下面的数字是**实时快照**（trace 正在被生产进程追加），
  每行的 `n_samples` 都已写进 CSV；重跑同一命令会得到略有差异的数字。
- 04:19–04:33 之间生产链重启（并在 04:33 前删掉了旧 trace），因此我保留了重启前那份
  7 文件 / 4536 样本的快照 CSV。

### 4.2 当前快照（2026-09-12 04:47:55，`runs/main`，arm=full_static，seed=42）

| task | model | samples | REFINE | worse | same | better | **unnec%** | no-gain% | accept% | meanΔ | 可用 STOP 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wmt19_en_zh | glm4-9b | 32 | 93 | 73 | 13 | 7 | **78.5%** | 92.5% | 8.6% | −11.09 | 1 |
| wmt19_en_zh | llama3.1-8b | 1000 | 2693 | 1496 | 525 | 672 | **55.6%** | 75.0% | 5.4% | −2.74 | 109 |
| wmt19_en_zh | qwen3-8b | 1000 | 1113 | 496 | 275 | 342 | **44.6%** | 69.3% | 10.6% | −0.97 | 642 |
| wmt19_zh_en | llama3.1-8b | 256 | 658 | 347 | 154 | 157 | **52.7%** | 76.1% | 4.9% | −2.65 | 38 |
| wmt19_zh_en | qwen3-8b | 512 | 780 | 283 | 344 | 153 | **36.3%** | 80.4% | 9.5% | −1.95 | 261 |

（`identical_text`：13 / 382 / 194 / 20 / 130，即一部分"same"是候选与当前答案逐字相同；
zh_en/qwen3-8b 上"完全没变"的比例最高，占 44.1%。glm4-9b 只有 32 条样本、
93 次修订，行内数值波动大，仅供参考。）

### 4.3 重启前快照（04:2x；该批 trace 已被删除，CSV 保留）

`scores/stopping_diagnostics_main_prerestart.csv`（arm=full_static，seed=42）：

| task | model | REFINE | worse | same | better | **unnec%** | no-gain% | accept% | meanΔ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wmt19_en_zh | llama3.1-8b | 2694 | 1466 | 526 | 702 | **54.4%** | 73.9% | 5.3% | −2.64 |
| wmt19_en_zh | qwen3-4b | 682 | 306 | 165 | 211 | **44.9%** | 69.1% | 9.2% | −1.37 |
| wmt19_en_zh | qwen3-8b | 1110 | 499 | 277 | 334 | **45.0%** | 69.9% | 11.1% | −1.15 |
| wmt19_zh_en | llama3.1-8b | 2018 | 1142 | 401 | 475 | **56.6%** | 76.5% | 5.0% | −3.10 |
| wmt19_zh_en | qwen3-4b | 260 | 94 | 96 | 70 | **36.2%** | 73.1% | 8.5% | −1.56 |
| wmt19_zh_en | qwen3-8b | 580 | 217 | 249 | 114 | **37.4%** | 80.3% | 9.0% | −1.68 |
| coedit_gec | qwen3-8b | 0 | 0 | 0 | 0 | — | — | — | —（64 条全部 STOP，无 REFINE） |

合计 7344 次 REFINE，**50.7% 变差**，**74.0% 没有任何离线增益**，接受率 6.9%，
meanΔ −2.31。`--recompute` 在这次快照上重算全部 7344 轮，与存储值偏差 **0.000e+00**。

### 4.4 整个 `runs/`（含 dev/gate 与 glm4-9b）

`scores/stopping_diagnostics_all_prerestart.csv`（同样是被删除前的快照）：

| task | model | REFINE | **unnec%** | accept% | meanΔ | 备注 |
|---|---|---:|---:|---:|---:|---|
| wmt19_en_zh | glm4-9b | 150 | **56.7%** | 2.7% | −4.12 | dev + smoke |
| wmt19_zh_en | glm4-9b | 139 | **66.2%** | 5.8% | −3.94 | dev + smoke |
| coedit_gec | glm4-9b | 144 | **47.2%** | 3.5% | −14.03 | 全部来自 `fixed_rounds`（强制修订）× α 扫描 |
| gigaword | glm4-9b | 144 | **69.4%** | 7.6% | −8.99 | 全部来自 `fixed_rounds` |
| wmt19_en_zh | qwen3-8b | 1110 | 45.0% | 11.1% | −1.15 | main |
| wmt19_zh_en | qwen3-8b | 580 | 37.4% | 9.0% | −1.68 | main |
| coedit_gec | qwen3-8b | 0 | — | — | — | main，64 条全 STOP |

按臂拆开看（同一份 CSV 的 `scope=run` 行）：
- **`fixed_rounds`（强制修订）**：GEC *unnec* = 43.8–52.1%（α=0.0/0.5/1.0），
  gigaword 60.4–75.0%，en_zh 42.2–75.0%，zh_en 62.2–68.8% —— 与冻结记录里
  "强制修订下三个任务被改坏"的结论一致，但这里给出的是**逐轮**的比例。
- **`no_experience` / `random_retrieve`**：几乎不修订（0–3 次 REFINE，97–100% STOP），
  所以"无经验 ⇒ 不动手"，正是 progress.md 表一的机制。

### 4.5 离线指标与 judge 的分歧（当前快照）

| task/model | REFINE | judge: better | tie | worse | uncertain | 离线变好却被拒 | 离线变差却被接受 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wmt19_en_zh / glm4-9b | 93 | 8 | 36 | 13 | 36 | 6 | 7 |
| wmt19_en_zh / llama3.1-8b | 2693 | 145 | 449 | 481 | 1618 | 626 | 93 |
| wmt19_en_zh / qwen3-8b | 1113 | 118 | 208 | 185 | 602 | 296 | 65 |
| wmt19_zh_en / llama3.1-8b | 658 | 32 | 29 | 125 | 472 | 143 | 13 |
| wmt19_zh_en / qwen3-8b | 780 | 74 | 148 | 141 | 417 | 126 | 24 |

两点值得写进论文：**(i)** 双序 judge 有 **54–72%** 的轮次是 `uncertain`（两次顺序不一致），
这正是"参考无关 judge 不可靠"的直接证据；**(ii)** "离线变好却被拒"（626/296/143/126）
远多于"离线变差却被接受"（93/65/13/24），说明当前的接受判据系统性偏保守 —— 这正是
`--counterfactual` 想量化的"漏修"来源。

## 5. 无法验证 / 未验证的部分

1. **(A) `--counterfactual` 的真实生成轮没有跑。** 0/1/2/3 号卡都被生产任务占用
   （`nvidia-smi` 显示 GPU 0/1 上是生产链的 33 GB engine，2/3 上亦有人），遵守"不启动
   GPU 任务"的约束。该路径只用 stub engine 验证过接线。等有卡时的命令与规模：
   `--counterfactual --max-stop-states 100`；当前 `runs/main` 有 1051 个 STOP 状态可用
   （1 + 109 + 642 + 38 + 261），100 个状态约 400–500 次调用（每个状态 1 controller + 1 refine
   + 2 judge），按串行实测 ~3 s/样本换算约 **4–8 分钟 + 模型加载**。
2. **(B) 的生成模式没有跑**（同上）。已跑的只有 `--dry-run`（全 20 个 model-task）与
   stub 编排测试（8 条流、6 条评估、2 个 order）。
3. **真实运行的规模估算未实测**：`--dry-run` 给出的全量估计是
   82,800–243,280 次 LLM 调用（orders=1）；`--orders 3` 大约翻三倍（流 3×、评估 3×，
   frozen 只测 1 次）。实际耗时取决于自适应停止率，未实测。
4. **重启后的 trace 与新 batched 控制器修复的关系未验证**：04:19 的重启发生在
   `core/batched_pipeline.py` 加入 controller repair retry 之后，所以 4.2 的数字反映的是
   "修复后"的行为，4.3 是"修复前"。我没有做两者的正式对比。
5. **(B) 的 `--stream-draft-cache` 跨 order 回放只在代码层确认**（`CachedDraftSource.put`
   幂等、按 sample_id 索引），未在真实生成下跑过。

## 6. 事故与并发发现（需要你知道）

1. **`core/batched_pipeline.py` 的 `latency_s` 崩溃 bug（04:13:10 版本，已修复）**：
   该版本在 308-309 / 336-337 / 455-456 行写成
   `for k in (..., "latency_s"): s.cost[k] += getattr(g, k)`，但
   `baseline_core.types.Generation` 的字段是 `latency` → `AttributeError`。
   我在 04:2x 用 stub 编排测试复现并立即通知了主代理；**当前文件（04:14 之后）已经修好**
   （309-311 / 339-340 / 459-460 行改为先累加三个 token 字段、再 `s.cost["latency_s"] += g.latency`），
   已用 `getattr(Generation(...), k)` 逐键探针确认不再抛错，两个自检脚本也都在修复后的文件上通过。
   我全程没有改这个文件。
2. **一次误触发的 GPU 尝试（我的操作失误）**：为测试 GPU 预算检查，我执行了
   `run_stopping_diagnostics.py --runs runs/smoke --counterfactual --gpu 0,1,2,3`。
   `0,1,2,3` 恰好**在** `MAX_GPUS=4` 的政策之内，所以检查通过、脚本继续去加载 glm4-9b，
   在 GPU 0 上（当时生产链刚在同一张卡上起了 qwen3-8b）engine 初始化失败并抛
   `RuntimeError`，进程随即退出。事后核查：`ps` 无残留进程、`nvidia-smi` 里的
   compute app 全是生产链的 PID（912888/912997 → supervisor.py/run_experiment.py），
   生产 engine 从 04:30:57 起一直存活。**未重试**。教训值得记下：政策检查只能拦住
   越界的卡号，拦不住"合法的卡但正忙"。
3. **`runs/main` 旧 trace 被删除**：重启清除后，基于旧 trace 的复算无法再做；
   相关 CSV 已保留（见第 7 节）。

## 7. 改动与产物清单

新增：
- `run_stopping_diagnostics.py`（Phase 7）
- `run_accumulation.py`（Phase 8）
- `tests/test_stopping_counterfactual.py`、`tests/test_accumulation_loop.py`（GPU-free 自检）

对既有文件的**最小**改动（两处，均为消除重复的常量提取，行为不变）：
- `core/pipeline.py`：新增模块常量 `DEFAULT_REFINE_INSTRUCTION`，并让原先的内联字面量引用它；
- `core/batched_pipeline.py`：import 该常量并替换同一条内联字面量（2 行）。
  （该文件里 04:13 版本引入的 `latency_s` 问题由主代理在 04:14 自行修复，我没有碰它。）

产物：
- `scores/stopping_diagnostics.csv` —— 当前（04:47:55）`runs/main` 快照（每次重跑都会刷新）
- `scores/stopping_diagnostics_main_prerestart.csv` —— 重启前 7 文件 / 4536 样本快照
- `scores/stopping_diagnostics_all_prerestart.csv` —— 重启前整个 `runs/`（含 dev/gate）
- `runs/accumulation/`、`scores/accumulation_curve.csv`、`scores/accumulation_order_variance.json`
  —— 真实运行尚未产生（未跑生成模式）

复现（GPU-free）：

```bash
cd /mnt/huawei/ymb/aaai2027/exp
/home/ymb/miniconda3/envs/qwen35/bin/python run_stopping_diagnostics.py --runs runs/main
/home/ymb/miniconda3/envs/qwen35/bin/python run_stopping_diagnostics.py --runs runs/main --recompute
/home/ymb/miniconda3/envs/qwen35/bin/python run_accumulation.py --dry-run
/home/ymb/miniconda3/envs/qwen35/bin/python tests/test_stopping_counterfactual.py
/home/ymb/miniconda3/envs/qwen35/bin/python tests/test_accumulation_loop.py
```
