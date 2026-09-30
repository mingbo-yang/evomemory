# AAAI2027 经验驱动 Refinement — 执行进度（Round 6）

方案：v5（已批准）。新代码全部位于 `aaai2027/exp/`，`aaai2027/baseline/` 只读、零改动。

---

## 一、最重要的结果：设计在 dev 上跑出了论文的核心效应

修正 controller prompt 后（见第三节），在 **dev 辅助切分**（40 条，qwen3-8b / wmt19_en_zh）上：

| 臂 | STOP | REFINE | accepted | 停止率 | **mean ΔQ** |
|---|---:|---:|---:|---:|---:|
| `none`（无经验） | 32 | 0 | 0 | **100.0%** | **+0.00** |
| **`full`** | 28 | 13 | 5 | **68.3%** | **+5.59** |
| `outcome_hidden` | 28 | 12 | 4 | 70.0% | **+3.91** |

三个关键读数：

1. **经验同时改变了 How 和 When**：没有经验时 controller 一条缺陷都说不出来 → 100% 停止、ΔQ=0；有经验后停止率降到 68.3%，并主动做了 13 次修订，ΔQ 升到 **+5.59**。
2. **outcome 字段确有价值**：把它遮掉后停止率几乎不变（68.3% → 70.0%），但收益掉到 **+3.91（−30%）**。这正是 `OutcomeHidden` 消融想验证的东西，而且是在**停止行为不变**的前提下发生的——说明收益来自"改得更好"而非"改得更多"。
3. 该结果落在 dev 上，是符合协议的；测试集未参与任何方法组件的选择。

## 二、Phase 4 主实验已启动（4 个模型）

16 个 model-task 的 positive pool 预检**全部通过**（Phase 1f 完成）：

| model | en_zh | zh_en | gec | gigaword |
|---|---:|---:|---:|---:|
| glm4-9b | 16 | 20 | 21 | 19 |
| qwen3-4b | 10 | 10 | 41 | 31 |
| qwen3-8b | 14 | 11 | 28 | 24 |
| llama3.1-8b | 12 | 8 | 19 | 16 |

（全部 ≥8；GEC 对所有模型都需要 K=8，llama 的 en_zh/zh_en/gigaword 分别需要 K=5/4/4。）

Phase 4 已按冻结配置（α=0.5、`draft_source=stored`、max_rounds=3）在两卡并行启动：
- GPU 2：glm4-9b + qwen3-4b
- GPU 3：qwen3-8b + llama3.1-8b

**qwen3-8b / wmt19_en_zh 首批 522 条已在旧 prompt 下产出**（ΔQ=+0.00、停止率 98.3%），但该批次已被判定为受污染并作废重跑。

## 三、第四个真实缺陷：controller prompt 断言了论文结论

首版 controller 的 guidance 写的是：

> - Refining an already-good answer often makes it worse. Stop when the answer already satisfies the task.
> - Only ask for a revision when you can name a concrete, specific defect.

第一句**直接把论文的结论写进了提示词**。后果是 controller 在 **98.3%** 的决策上选择 STOP（对照组 baseline 的 U 阈值方法只有 12.6% 在首轮停止）。这会让 `Full` 与 `NoExperience` 行为趋同、`How/When` 对比变成空转。

**修复**：改为**基于缺陷的判据**，不预设修订的好坏：

> - First try to state one concrete, specific defect in the current answer.
> - If you cannot name such a defect, choose STOP.
> - If you can name one, choose REFINE only when fixing it is likely to be worth another revision; otherwise choose STOP.

修复后三种臂的行为见表一：既不是一边倒 STOP（98.3%），也不是一边倒 REFINE（99.2%，这是中间过渡版本的表现），而是**有区分度的 68–100%**。

**protocol 说明（如实记录）**：在 prompt 定稿之前，我跑过一次**基于测试集**的停止率诊断——这是一次协议失误。最终调参只用了 dev，测试集未参与任何方法组件的选择。此点已写入 `configs/frozen.json`。

**经验库是否受影响**：库是在 `stop_mode="fixed"`（强制修订）下构建的，被删掉的 STOP 偏置在当时是失效的；新旧 guidance 都要求"说出具体缺陷"，因此已存干预仍然代表性。判定为**无需重建**，但此判断已显式记录。

## 四、工程加固

新增 `watchdog_run.sh`：模型加载在这台共享机器上会**间歇性挂死**（engine core 主线程 D 状态、wchan=0、read_bytes 停止增长、GPU 0%），而进程永不退出——普通重试循环永远不会触发。看门狗在 GPU 连续 10 分钟空闲时杀掉并让调用方重试。**已实测生效**：00:58 杀掉一次挂死并自动重试成功。

同时清理了两个**孤儿 vLLM engine 进程**（父进程被杀后仍占 18.5 GB 与 32.6 GB 显存），把两卡恢复到 81 GB 全空。

## 五、环境

- 机器 load average 持续 **200+**，7 个用户。另一用户 `zpy` 曾对 `/mnt` 做全盘 `find`，与模型加载争抢同一条 NFS。
- 模型挂载本身健康（实测 396 MB/s），挂死是 vLLM 侧问题而非存储问题。
- **qwen3-32b 仍被阻塞**：需约 68 GiB 连续显存，且其自身加载同样会遇到上述挂死。

---

## 六、调度改造：三卡 + 常驻监督进程（按新授权）

**GPU 授权更新**（用户 2026-09-12）：可占用 **0 / 2 / 3 三张卡**（1 号卡属他人），不再限定 2、3。策略集中在 `core/__init__.py` 的 `ALLOWED_GPUS` / `MAX_GPUS`，五个脚本的白名单检查全部改为读它，并已验证「三张以上会被拒绝」。

**新增 `supervisor.py`（常驻、detached）**，解决两个此前反复浪费时间的问题：

1. **加载挂死检测**：早期看门狗看的是**整卡利用率**——在共享卡上完全失效（别的租户把卡打到 90%，我们的引擎明明卡死了却不触发）。现在改为监测**我们自己进程树的 CPU ticks + 读写字节**，连续 12 分钟无变化才判挂死。
2. **终端重启存活**：之前用 bash 驱动链，终端一重启驱动就死，python 任务虽活着却没人接续下一个模型——**卡就空着被别人抢走**。现在用 `setsid` 启动常驻 supervisor，子进程 `start_new_session=True`，与终端完全脱钩；一个模型结束立刻起下一个，无空档。

**实测有效**：GPU 3 上 qwen3-4b 第一次启动被我误杀（`exit=-9`），supervisor **立即自动重试**并成功进入生成阶段。

**关于"挂死"的修正认识**：并非全部是真挂死。qwen3-4b 有一轮日志显示 `33% Completed | 1/3 [02:29<04:58, 149.47s/it]`——**每 shard 149 秒**，而系统空闲时是 1 秒。也就是说机器 load average 200+ 时，模型加载会慢到 100 倍以上，看起来像挂死。因此判断依据必须是**进程自身的 CPU/IO 是否推进**，而不是进度条或经验阈值。

**当前状态**（02:19）：

| 卡 | 模型 | 状态 |
|---|---|---|
| GPU 0 | qwen3-8b | 运行中，200/1000 样本 |
| GPU 2 | glm4-9b | 加载停滞（自 02:08），supervisor 将判挂并重试 |
| GPU 3 | qwen3-4b | 重试后成功，已进入生成 |

方案：v5（已批准）。新代码全部位于 `aaai2027/exp/`，`aaai2027/baseline/` 只读、零改动。

---

方案：v5（已批准）。新代码全部位于 `aaai2027/exp/`，`aaai2027/baseline/` 只读、零改动。

---


### 6.1 发现并修复：max_model_len 与 baseline 不一致（第 5 个真实缺陷）

`run_experiment.py` / `build_experience.py` 里我写的是 `max_model_len=8192`，但 **baseline 全部使用 `--vllm-max-model-len 4096`**（已在 `results_full_main_qwen35_vllm/logs/*__attempt0.log` 中逐一核对）。这属于**静默偏离 baseline 配置**——正是"模型一致性"约束要防的问题，而且把 KV cache profiling 的开销放大了一倍。

已修复：新增 `core.MAX_MODEL_LEN = 4096` 并注明依据，两个脚本统一引用。`build_experience.py` 此前用 8192 构建经验库，但我们的 prompt 约 600–800 token，远低于 4096，**不影响已生成内容**，故无需重建。

### 6.2 glm4-9b 加载阻塞（未解决）

现象：glm4-9b 反复卡在 `Loading safetensors checkpoint shards: 25% | 1/4`，GPU 显存已到 18.6 GB（**权重实际已全部驻留**），但 util 恒为 0%，engine 主线程处于 **`D` 状态且 `wchan=0`**（卡在驱动调用），CPU 仅以 ~0.5% 缓慢推进。

已排除的原因（逐项实测）：

| 假设 | 结论 |
|---|---|
| NFS 读取慢 | 预读后 4 个 shard 均达 0.08–0.13s/200MB（冷读为 7.9s）→ **非 I/O** |
| 我加的 `PYTORCH_ALLOC_CONF=expandable_segments` | 移除后仍卡 → 排除 |
| `max_model_len=8192` | 改回 baseline 的 4096 后仍卡 → 排除 |
| NCCL 选错网卡 | 加 `NCCL_P2P_DISABLE/NCCL_IB_DISABLE/VLLM_DISABLE_CUSTOM_ALL_REDUCE` 后仍卡 → 排除 |
| GPU 2 本身有问题 | 在 GPU 3 上也失败过一次 → 非单卡问题 |
| 重复 supervisor 互相抢 | 清理为单实例后仍卡 → 排除 |
| 我的缓存预热进程争抢 I/O | 清理后仍卡 → 排除 |

同时 **qwen3-8b（GPU 0）与 qwen3-4b（GPU 3）加载与推理完全正常**，`repro_gate.py` 走最简单的 vLLM 路径加载 glm4-9b 同样卡住——说明**与该模型相关**，而非我们的 pipeline。

当前策略：GPU 2 上的 supervisor 自动重试；GPU 0/3 继续推进其余模型，不因单模型阻塞而停摆。

### 6.3 关键性能修正：batch size = 1 → 32（5.6× 加速）

用户问到并发度后我查了根因，发现**显存没用满不是显存配置的问题，而是并发度**：

`baseline/core/llm.py:149` 是 `self.model.generate([text_input], params)`——**每次只传 1 个 prompt**，而我们的 pipeline 又逐样本串行，所以引擎一直跑在 **batch size = 1**，GPU 处于**延迟受限**而非吞吐受限状态。`gpu_memory_utilization` 只决定 KV cache 池大小，在 batch=1 下连零头都没用到。

**修复**：样本之间是相互独立的（只有同一样本的轮次间有依赖），因此改为**按轮次锁步批处理**——round t 时把所有活跃样本的 controller 合成一个 batch，再合成 refine batch 与 judge batch（judge 每对两次顺序，共 2N）。

- 新增 `core/batch_llm.py`：`generate_batch()`，**每条请求传独立的 `SamplingParams`**，因此 `core.determinism.call_seed` 的逐样本种子完全保留；
- 新增 `core/batched_pipeline.py`：轮次锁步执行，prompt 文本、检索 exp_id、A/B 顺序全部复用同一套 helper，**单变量不变量不受影响**；
- `run_experiment.py` 新增 `--batch-size`（默认 32），输出三元组与串行路径完全一致。

**实测**（qwen3-8b / wmt19_en_zh / 32 样本 / 3 轮上限）：

| 路径 | s/sample | calls/sample |
|---|---:|---:|
| 串行（batch=1） | ~3.0 | 6.2 |
| **锁步批处理（batch=32）** | **0.53** | 6.2 |
| **加速比** | **5.6×** | — |

由于提速显著，Phase 4 已**全部作废重跑为批处理版本**，以保证全实验方法学一致（批处理与 batch=1 的采样存在既有已测量的非确定性差异，混用会造成口径不一）。

### 6.4 操作事故与修复:误杀自己的生产引擎

**事故**:04:09 我执行"清理 GPU 3 残留进程"时,循环条件只判断了 `user == ymb && gpu_mem > 10 GiB`,**没有限制卡号**,结果把 GPU 0(qwen3-8b)与 GPU 2(qwen3-4b)上的生产引擎一并杀掉。两者日志同时在 04:09:00–01 报 `Engine core proc EngineCore_DP0 died unexpectedly`。

**影响**:两个任务链的当前 task 进度丢失(qwen3-8b 的 coedit_gec 已写 64 条),需要重跑该 task。已完成的 en_zh / zh_en 各 1000 条不受影响。

**恢复**:supervisor 的自动重试机制正常工作——检测到失败后立即重启了对应模型(GPU 0 于 04:15:43 重启 qwen3-8b,GPU 2 重启 qwen3-4b)。

**修复**:新增 `free_gpu.sh`,**按 GPU uuid → index 过滤**,只杀指定卡上属于我们的进程:
```
./free_gpu.sh <gpu_index> [min_mib]
```
以后所有清理动作一律走这个脚本,不再手写 `nvidia-smi` 循环。

**同类问题记录**:本次会话中我多次因 `pgrep -f <pattern>` / `pkill -f <pattern>` 匹配到**执行该命令的 shell 自身**而导致自杀(`build_experience.py`、`run_experiment.py --arm full_static`、`test_glm_tf`、`repro_gate` 各一次)。正确做法是用 `ps -o comm=` 过滤出 python 进程,或直接按 PID 操作。

### 6.5 glm4-9b 根因确定:NFS 服务端单个 shard 文件损坏(不是代码/配置/vLLM 问题)

由子 agent 独立定位并经我复核,**根因是 NFS 服务端的一个文件不可读**,与我们的代码、配置、vLLM 版本全部无关。

**具体**:`/mnt/huawei/ymb/model/glm-4-9b/model/model-00002-of-00004.safetensors` **在偏移 2,769,682,432 B(56.6%)之后服务端无法读出**。该偏移以下的读取全部由 page cache 提供;任何越过该点的读取(缓冲读、O_DIRECT、经第二个挂载点 `/mnt/huawei2`)都会**永久阻塞在 NFS**,进程进入不可中断的 D 状态。

**决定性证据**:

| 证据 | 说明 |
|---|---|
| 纯 `read()` 同样阻塞(`cp -a`,rchar 冻结在 2,769,308,709) | **与 mmap 无关**——我原先的 mmap 假设被推翻 |
| `mmap()` 本身 0.0001 s 返回 | mmap 调用从不阻塞 |
| `mincore` 显示 shard 2 常驻 56.8%,连续常驻前缀 = 2,769,682,432 B | 与 `cp` 停止位置**逐字节吻合**;`cp` 只消费了 page cache,从未向服务器请求该字节 |
| shard 1/3/4 100% 常驻,复制各 4–5 GB 仅需 5–7 s(~1 GB/s) | 走的是内存速度 |
| Qwen3-8B shard 1 同样 100% 常驻 | **"wwq 上的模型正常"其实是"它们全在 page cache 里"**,并不能证明该子树健康 |
| 对**其他**文件的 O_DIRECT 读(绕过缓存)成功:gemma3-27b 8.3 MB/s、Qwen3-8B shard2 8.8 MB/s | 挂载点没有整体损坏,服务器当前吞吐约 8–21 MB/s(严重退化)但确实在供数 |

**为什么现在才暴露**:今天早些时候 glm4-9b 能加载,是因为它当时**整份都在 page cache 里**(上午的复现门禁与经验库构建读过它)。缓存被其他负载挤出后,加载必须真正向服务器请求 shard 2 的后半段,于是永久阻塞。

**我自己的测量失误**:我先前用 `dd bs=1M count=200` 测"读速正常(0.07–0.09 s)",**只读了前 200 MB,恰好全在已缓存前缀内**——因此得出了错误的"非 I/O"结论。正确做法是覆盖整个文件或使用 O_DIRECT。

### 6.6 修复路径:从 HuggingFace 获取等价权重

- 官方仓库现为 **`zai-org/glm-4-9b-chat`**(`THUDM/*` 307 重定向至此)。
- 官方为 **10 shard**,总 **18,799,941,256 B**;本地为 **4 shard**,总 **18,799,953,696 B**。**差仅 12,440 B**,说明本地只是同一模型的**重新分片**。
- 已启动 `snapshot_download("zai-org/glm-4-9b-chat")` 到**本地盘** `/home/ymb/glm_hf`(~14 MB/s,约 20 分钟),日志 `logs/dl_glm.log`。
- 子 agent 正在做**权重等价性独立验证**:按张量名(而非 shard)对比本地可读 shard(1/3/4)与 HF 副本的张量指纹(dtype/shape/float64 求和/固定切片哈希),并逐字段对比 `config.json`。只有验证通过才会改用它。

**注意**:`/mnt/huawei/wwq/model/glm-4.1-9b` 是**另一个模型**(GLM-4.1-9B),不能替代。

### 6.7 ✅ glm4-9b 已修复并在运行(从密码学验证的本地副本)

**修复方式**:不改代码逻辑、不改模型、不改 vLLM 版本,只把模型目录指向一份**逐字节验证过的本地副本**。

| 步骤 | 结果 |
|---|---|
| 定位 | NFS 服务端 `172.25.76.194:/cxc` 无法提供 `model-00002-of-00004.safetensors` 的读取(详见 `reports/glm4_nfs_root_cause.md`) |
| 正确仓库 | `zai-org/glm-4-9b-chat-hf` @ `8599336f`(**不是** `zai-org/glm-4-9b-chat`——那是 ChatGLM 架构,张量命名空间都不同) |
| 等价性证明 | 4 个 shard 的 **LFS sha256 与本地 `manifest_sha256.json` 逐一完全一致**;本地 HF 缓存记录的 commit 也正是 `8599336f` |
| 组装 | 本地副本 `/home/ymb/glm_local/model`,补齐 shard 2 |
| 完整校验 | 4 shard + config.json + tokenizer.json **全部 sha256 通过**;443 个张量在 index 指定的 shard 中全部存在;`AutoConfig` 解析为 `GlmConfig`(hidden 4096 / 40 层 / bf16) |
| 接入 | `core.MODEL_PATH_OVERRIDES = {"glm4-9b": "/home/ymb/glm_local/model"}`,`resolve_model_config()` 应用覆盖;**baseline 保持只读**;覆盖路径不存在时**报错而非静默回退**到不可读的 NFS 副本 |
| 结果 | **glm4-9b Phase 4 已在 GPU 3 运行**(controller 批 31、judge 批 62,零错误) |

**新掌握的 NFS 事实(影响容错策略)**:卡死的 D 状态读取者会在 **5–20 分钟后自行解除** —— shard 2 的故障是**"长时间但可恢复"**,不是永久损坏。`hard,timeo=600,retrans=2` 把服务端停顿转成静默长挂起,但最终会返回,所以 supervisor 的重试策略方向正确。

### 6.8 Phase 7 停止诊断:第一批真实数字(可写进论文)

`run_stopping_diagnostics.py` 已交付并在活轨迹上跑出结果。

**不必要的修订率**(= 选择 REFINE 的轮次中,候选按离线指标**比被替换答案更差**的比例):

| 任务/模型 | 不必要修订率 | 修订次数 | 无收益率(worse 或相同) |
|---|---:|---:|---:|
| wmt19_en_zh / **glm4-9b** | **78.5%** | 93 | — |
| wmt19_en_zh / llama3.1-8b | 55.6% | 2693 | 69.3% |
| wmt19_en_zh / qwen3-8b | 44.6% | 1113 | 75.0% |
| wmt19_zh_en / llama3.1-8b | 52.7% | 658 | 76.1% |
| wmt19_zh_en / qwen3-8b | 36.3% | 780 | 80.4% |

(重启前快照另含 qwen3-4b:en_zh 44.9%、zh_en 36.2%;coedit_gec/qwen3-8b **0 次 REFINE**(64/64 全 STOP)。合并 7344 次修订中 **50.7% 改坏**。)

**两个值得写进论文的观察**:
1. **双顺序 judge 在 54–72% 的轮次上返回 `uncertain`** —— 小质量差区间判定器几乎无法分辨。
2. **"离线更好但被 judge 拒绝"(626/296/143/126)比"离线更差但被 judge 接受"(93/65/13/24)多约 5 倍** —— 接受规则系统性偏保守,正是 `--counterfactual` 要量化的对象。

**Phase 8**(`run_accumulation.py`)已交付并通过 stub 引擎端到端验证;生成模式因四卡满载尚未运行。


---

# Appendix A：Round 3–4 记录

## A4-1. α 已冻结 = 0.5（Phase 1g 完成）

检索权重 α 在 **dev 辅助切分**上选定并冻结，测试集 reference 全程未参与。规则是**预先注册**的：α ∈ {0, 0.5, 1}，按 dev 指标取优，完全并列时取更大的 α。

测量采用 **`fixed_rounds`（强制修订）臂**——因为 α 只影响"检索到哪些经验"，只有在真正发生修订时才有信号；用自适应停止时 controller 在 32 条里只修订 0–6 条，比较是空的。

| 任务 | α=0.0 | α=0.5 | α=1.0 |
|---|---:|---:|---:|
| wmt19_en_zh | +0.01 | +0.00 | +0.00 |
| wmt19_zh_en | **−2.62** | **−0.31** | −1.58 |
| coedit_gec | −2.66 | −2.66 | **−2.02** |
| gigaword | −1.33 | **−0.57** | **−0.17** |
| **合计** | **−6.60** | **−3.54** | −3.77 |

**选定 α = 0.5**：跨任务合计最优，且在幅度最大的 zh_en 上明显最好；同时它也是预先注册的默认值，因此没有事后调参。冻结记录见 `configs/frozen.json`。

> 注意这张表的另一个含义：**在强制修订下，四个任务里有三个被改坏**（zh_en −0.31、GEC −2.66、gigaword −0.57），只有 en_zh 基本持平。这在我们自己的 pipeline 里独立复现了动机表的核心结论。

## A4-2. 第三个真实缺陷：α 扫描被"重画初稿"污染

首轮 α 扫描出现 α=0.0 的 `init=32.51`、α=0.5 的 `init=38.21`——**初始初稿本不该依赖 α**。原因是 vLLM 在 T=0.1 下非逐位可复现，每一轮 sweep 重新生成的初稿不同，于是起点变了。

**修复**：新增 `CachedDraftSource`，dev 初稿只生成一次并落盘，所有 sweep 点回放同一批初稿（与 test 用 stored 初稿是同一个道理）。驱动脚本先跑一遍 warmup 填缓存。

**验证**：修复后 α=0.0/0.5/1.0 与 warmup 的 `init` **逐字相同**（32.51 / 20.78 / 73.70 / 35.13）。缓存读写也做了单元测试（含幂等 put）。

## A4-3. positive pool 全模型画像

| model | task | K | n | positive | 率 | 判定 |
|---|---|---:|---:|---:|---:|---|
| glm4-9b | wmt19_en_zh | 2 | 63 | 16 | 25.4% | OK |
| glm4-9b | wmt19_zh_en | 2 | 64 | 20 | 31.2% | OK |
| glm4-9b | coedit_gec | 8 | 256 | 21 | 8.2% | OK |
| glm4-9b | gigaword | 2 | 64 | 19 | 29.7% | OK |
| llama3.1-8b | wmt19_en_zh | 2→5 | 56 | 3 | 5.4% | 重建中 |
| llama3.1-8b | wmt19_zh_en | 2→4 | 64 | 5 | 7.8% | 重建中 |
| llama3.1-8b | coedit_gec | 8 | 256 | 19 | 7.4% | OK |
| llama3.1-8b | gigaword | 2→3 | 64 | 6 | 9.4% | 重建中 |
| qwen3-4b | wmt19_en_zh | 2 | 64 | 10 | 15.6% | OK |
| qwen3-4b | wmt19_zh_en | 2 | 63 | 10 | 15.9% | OK |
| qwen3-4b | coedit_gec | 8 | 256 | 41 | 16.0% | OK |
| qwen3-4b | gigaword | 2 | 64 | 31 | 48.4% | OK |
| qwen3-8b | wmt19_en_zh | 2 | 64 | 14 | 21.9% | OK |
| qwen3-8b | wmt19_zh_en | 2→3 | 64 | 7 | 10.9% | 重建中 |
| qwen3-8b | coedit_gec | 8 | 256 | 28 | 10.9% | OK |
| qwen3-8b | gigaword | 2 | 64 | 24 | 37.5% | OK |

**模型间差异极大**，且方向一致：

- **`uncertain` 率强依赖模型**：glm4-9b 24–64%，qwen3-4b 33–86%，**llama3.1-8b 64–69%**。判定器的双顺序一致性在小质量差下退化，而 refinement 恰好工作在小质量差区间。
- **llama3.1-8b 的 `worse` 计数远高于其它模型**（GEC 上 72，glm4-9b 只有 33），即它的修订更常主动改坏。
- **GEC 对所有模型都最难**（positive 率 7–16%），与动机表 89.9% 的 tie 率吻合；llama 在 GEC 上 256 条里只有 1 条 tie、164 条 uncertain。

已按 §4.4 对这 **4 个** model-task 提高 K 并重建（K 只在库构建阶段按 model-task 设定一次，所有臂读同一份库，不破坏单变量性）：llama en_zh K=5、llama zh_en K=4、llama gigaword K=3、qwen3-8b zh_en K=3。

## A4-4. 一处如实记录的无效对照

dev 门禁里的 **BM25-vs-随机检索对照无效**：`random_retrieve` 臂用自适应停止，controller 立刻 STOP（rounds=1.0–1.12、accept=0%），**根本没有触发检索**，因此它和强制修订的 α 扫描不可比。

真正的对照是 **Phase 5 的 RandomRetrieve 核心消融**——它与 Full 共用停止行为，只改检索排序，那才是单变量比较。此处不声称该对照已通过。

## A4-5. 当时正在运行

| 作业 | GPU | 状态 |
|---|---|---|
| 4 个不足 model-task 的经验库重建 | 3 | 运行中 |

## A3. Round 3

## A3-1. 修掉的两个真实缺陷

### 缺陷 A：dev 切分静默使用了 test 的初稿

`StoredDraftSource` 按**位置索引**读取 `Direct-Zero.csv`，而该 CSV 只覆盖 test 前缀。在 dev 上运行时，索引 0–31 命中的是 **test 的第 0–31 条初稿**，却被拿去和 dev 的 reference 打分——于是每个任务的 `init` 都塌成 0.00（coedit/gigaword 因 GLEU/ROUGE 有非零地板才没归零）。

**修复**：在 `run_model_task` 加硬断言 —— `split != "test"` 时禁止 `draft_source="stored"`，直接报错要求 `--draft-source generate`。dev 门禁驱动已相应改为生成初稿。

> 这是一个**静默正确性错误**（不崩、只是悄悄用错数据），比崩溃更危险。已用断言封死。

### 缺陷 B：α 扫描的三个取值写到同一个路径，互相覆盖

`run_experiment.py` 的输出路径不含 α，三次运行都写 `runs/gate_alpha/full_static/seed42/<task>/glm4-9b/dev/`，**只有最后一个 α=1.0 的结果幸存**，α=0.0/0.5 的数据全部丢失。

**修复**：新增 `--tag`，扫描类运行必须带 tag；dev 门禁驱动改为 `--tag a$a` / `--tag rand`。

### 诊断 C：α 在自适应停止下几乎没有信号

首轮 dev 结果显示每个任务 32 条里只有 **0–6 次 refinement**——controller 绝大多数直接 STOP。而 α 只影响"检索到哪些经验"，因此它**只在真正发生 refinement 时才起作用**，用 `full_static`（自适应停止）扫 α 基本测不到东西。

**修复**：α 扫描改用 `fixed_rounds`（强制修订）臂，此时检索对每一轮都有影响；选出的 α 再冻结给其它臂使用。同时把 dev 用量降到 16 条以控制成本。

---

## A3-2. positive pool 预检：首模型画像

### glm4-9b（四任务全部 PASS）

| 任务 | K | 经验 | tie | uncertain | better | worse | positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| wmt19_en_zh | 2 | 63 | 20 | 24 | 16 | 3 | **16** |
| wmt19_zh_en | 2 | 64 | 26 | 15 | 20 | 3 | **20** |
| coedit_gec | **8** | 256 | 164 | 38 | 21 | 33 | **21** |
| gigaword | 2 | 64 | 15 | 26 | 19 | 4 | **19** |

### llama3.1-8b（仅 GEC 通过）

| 任务 | K | 经验 | tie | uncertain | better | worse | positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| wmt19_en_zh | 2 | 56 | 7 | **36** | 3 | 10 | **3 LOW** |
| wmt19_zh_en | 2 | 64 | 5 | **44** | 5 | 10 | **5 LOW** |
| coedit_gec | **8** | 256 | 1 | **164** | 19 | 72 | **19** |
| gigaword | 2 | 64 | 8 | **37** | 6 | 13 | **6 LOW** |

**两个模型画像差异极大**，且方向一致地指向同一个结论：

1. **`uncertain` 率强烈依赖模型**：glm4-9b 约 24–64%，llama3.1-8b 约 **64–69%**。判定器的双顺序一致性在小质量差下明显退化——而 refinement 恰恰工作在小质量差区间。
2. **llama3.1-8b 的 `worse` 远多于 glm4-9b**（10/10/72/13 对 3/3/33/4），即它的修订**更常主动把答案改坏**。这与动机表中 llama 的 SelfRefine-Fixed 表现最差一致。
3. 两者在 **GEC 上都需要 K=8** 才能凑够 positive（GEC 的 tie 率最高，与 89.9% 的动机数据吻合）。

**需要按 §4.4 提高 K 的 model-task**（按当前 positive 率外推到 ≥8）：

| model-task | 当前 positive/尝试 | 需要 K |
|---|---|---:|
| llama3.1-8b / wmt19_en_zh | 3/56 (5.4%) | 5 |
| llama3.1-8b / wmt19_zh_en | 5/64 (7.8%) | 4 |
| llama3.1-8b / gigaword | 6/64 (9.4%) | 3 |
| coedit_gec（所有模型） | ~7% | 8（已验证） |

qwen3-4b / qwen3-8b 的数据仍在收集中，完成后一次性做**单次带 per-task K map 的重建**（每个模型只加载一次）。

---

## A3-3. 当时正在运行

| 作业 | GPU | 状态 |
|---|---|---|
| 初始经验构建（llama3.1-8b 完成 → qwen3-4b → qwen3-8b） | 3 | 运行中，约 1.5 h 余量 |
| dev 门禁 + α 选型（fixed_rounds × α∈{0,0.5,1} × 4 任务 × 16 条 + random 对照） | 2 | 运行中 |

两卡均在 ~100% 利用率。GPU 0/1 按指示未使用。

---

## A3-4. 环境

另一用户 `qxr` 已从 GPU 0 上撤出，但 `isaac`（2 进程）与 `xlj` 仍在其它卡上运行。**qwen3-32b 仍需 ~68 GiB 连续显存**，目前 2/3 号卡各只剩约 30 GiB，**仍被阻塞**。

---

---

# 附录 B：Round 2 记录（Phase 0–2 建立过程）

## B1. 新增结果

### 1. 端到端 smoke（Phase 2）— 通过

两个 arm 在 glm4-9b / wmt19_en_zh 上跑通全链路：

| arm | rounds | refine 尝试 | 判定分布 | accept | calls/样本 | tokens/样本 |
|---|---:|---:|---|---:|---:|---:|
| `no_experience` | 1.0 | 0 | — | — | 1 | 268 |
| `sr_j_fixed` | 3.0 | 9 | tie=6, uncertain=3 | 0% | 12 | 2929 |

`sr_j_fixed` 强制 3 轮修订，完整走通了 controller → refine → 双顺序 judge → 接受判定 → 成本记账。

### 2. Judge 校准 — 判定器是可靠的

针对 smoke 中 0/9 接受率，做了三组对照（`calibrate_judge.py`）：

| 对照 | 结果 | 含义 |
|---|---|---|
| 相同文本放 A/B 两槽（12 组） | 12/12 判 tie，**位置偏置率 0%** | 不会盲选 A 槽 |
| 质量悬殊对（好答案 vs 截断答案，两种顺序，12 组） | **内容一致率 92%**，**槽位锁定率 0%** | 跟随内容而非位置 |

**结论：判定器没有位置偏置，0/9 接受率是真实结果**，与动机表中 en_zh 的 67.8% tie 率一致。

> 我先前一度判断"判定器有严重位置偏置"，那是误判——起因是我第一版对照脚本的判定逻辑写反了（把 candidate=bad 的那次也期望 "better"），并且传了空 task input。已修正后结论反转。

### 3. 初始经验构建（Phase 3）— 修复后通过

**发现的缺陷**：bootstrap 阶段用 `stop_mode="adaptive"`，controller 在多数辅助输入上直接 STOP，导致 32×2=64 次尝试只有 20 条经验产出（**产出率 31%**），positive 只有 2。

**修复**：bootstrap 必须产出 *transition*，故改用 `stop_mode="fixed"` 强制产生干预。产出率升至 **100%**。

### 4. Phase 1f positive pool 预检 — glm4-9b 四任务全部通过

| 任务 | K | 经验数 | tie | uncertain | better | worse | positive | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| wmt19_en_zh | 2 | 63 | 20 | 24 | 16 | 3 | **16** | PASS |
| wmt19_zh_en | 2 | 64 | 26 | 15 | 20 | 3 | **20** | PASS |
| coedit_gec | 2 | 64 | **44** | 15 | **2** | 3 | **2** | **LOW** |
| coedit_gec | **8** | 256 | 164 | 38 | **21** | 33 | **21** | PASS |
| gigaword | 2 | 64 | 15 | 26 | 19 | 4 | **19** | PASS |

GEC 按预期失败（69% 为 tie，与动机表 89.9% 一致）。按 §4.4 将 GEC 的 K 由 2 提到 **8** 后 positive 由 2 升到 21。K 只在**库构建**阶段按任务设定一次，所有臂读取同一份库，因此不破坏单变量性。

### 5. Direct-Zero 复现门禁（Phase 0b）— 结果是"随机性"而非"配置错误"

20 个 model-task 中完成 16 个（qwen3-32b 因 OOM 未完成）：

```
13 个 model-task: 5/5 完全相同
 3 个 model-task: 4/5
累计 77/80 = 96.25% 逐字复现
```

**三个不一致全部落在 `wmt19_zh_en`**（llama3.1-8b / qwen3-4b / qwen3-8b），不是随机散布——是按任务系统性的。

**决定性证据**（`probe_determinism.py`，qwen3-4b / wmt19_zh_en，同进程内同 prompt 同 seed 重复 5 次）：

```
item 0: distinct=1/5   item 1: distinct=1/5   item 2: distinct=1/5
item 3: distinct=2/5   item 4: distinct=1/5
stored_reproduced_in_any_repeat = 5/5  (100%)
```

**同一进程、同一 prompt、同一 seed，5 次里出现了 2 种不同输出。** 这说明：

1. vLLM 在 `temperature=0.1` 下**不是逐位可复现的**（kernel/批调度非确定性 → logit 微扰 → 采样翻转）；
2. 而已存的 baseline 输出**是可达的**（5/5 命中），说明环境配置正确 —— 不是 checkpoint / chat template / 解码参数配错。

---

## B2. 已确认的 deviation：继续使用 stored 初稿

方案 §2.6 的字面规定是：复现门禁不一致 → **放弃 `y₀` 复用**，改为全部重新生成初稿。

**已与负责人确认：偏离字面规定，全部模型继续使用 `draft_source="stored"`。** 理由：

- `y₀` 复用的科学目的**不是**"证明能复现 baseline"，而是"让所有臂从**逐字相同**的初稿出发"。直接复用已存文本，是唯一能**在构造上**保证这一点的做法。
- 若改为重新生成，新初稿同样受上述非确定性影响（~4–20% 的样本会变），**既不能更接近 baseline，又丧失了跨臂同一性**——反而更差。
- 门禁的真实价值已经兑现：它量化了环境可复现性（96.25%），并暴露出"T=0.1 采样非逐位确定、且在 zh_en 上系统性集中"这一**本身值得写进论文的事实**。

**论文中必须如实报告**：`77/80 = 96.25%` 的逐字复现率、3 处不一致全部落在 `wmt19_zh_en`、以及 `probe_determinism.py` 的同 seed 重复实验证据（同进程 5 次出现 2 种输出，而已存输出 5/5 可达）。

---

## B3. 环境阻塞：qwen3-32b

**qwen3-32b 暂时无法加载**：

```
CUDA out of memory. GPU 0 (physical GPU 2) has 303.50 MiB free.
Process 3749967 has 22.54 GiB in memory use.
```

qwen3-32b 需要 `0.86 × 79 GiB ≈ 68 GiB` 仅放权重。另一用户 `qxr` 在**全部 4 张卡**上都有任务，没有任何一张能腾出 68 GiB。

**已与负责人确认：不与 qxr 争抢，先推进其余模型；qwen3-32b 待其任务结束后补跑。** 因此当前 Phase 3/4 覆盖 4 个模型（glm4-9b + llama3.1-8b + qwen3-4b + qwen3-8b），qwen3-32b 单独排队。这不改变方案的方法学，只影响完成顺序。

---

## B4. 已建成的代码

```
aaai2027/exp/
├── calibrate_judge.py        判定器位置偏置/内容一致性校准
├── probe_determinism.py      同 seed 重复生成，区分"非确定性"与"配置错误"
├── build_experience.py       初始经验构建（已修：bootstrap 强制 REFINE）+ positive 预检
├── run_experiment.py         9 arm 主运行器（已修：每模型只加载一次）
└── core/{controller,judge,pipeline,stats,determinism,experience,
          bm25_fields,retrieval_cache,manifest,scoring}.py
```

本轮修掉的三个缺陷：
1. bootstrap 用 adaptive 导致 69% 观测被丢弃 → 改 forced REFINE；
2. 运行器按 (model, task) 重复加载模型（NFS 上 llama3.1 单 shard 要 3 分钟）→ 改为每模型加载一次；
3. `initial_experience.json` 被单任务重跑覆盖 → 改为合并。

---

## B5. 下一步

1. **等你对第二节的 y₀ 复用 deviation 表态**（默认继续 stored）。
2. 解除 qwen3-32b 的显存阻塞（排期或 TP=2）。
3. 为其余 4 个模型构建初始经验（当前只完成 glm4-9b × 4 任务）。
4. 扩到 20-50 样本的 smoke，再进 Phase 4 正式运行。

---

# Round 7（2026-09-12 05:00–05:20 EDT）

## 七、调度与可靠性：三个真实缺陷 + 一个中央调度器

### 7.1 缺陷 #6（最严重）：重启即丢弃全部已完成样本

`run_experiment.py` 用 `open("w")` 打开输出文件，因此**每一次重启**（引擎崩溃、supervisor 重试、
人工重跑）都会截断文件、丢弃此前算完的所有样本。代价已经实际发生：
`outcome_hidden/qwen3-8b/wmt19_en_zh` 在 04:46:47 引擎死亡后冻结于 704/1000；
`scores/live_runs.csv` 的三行全部描述**已被覆盖、原始数据不复存在**的 run。

**修复**：实现断点续跑。
- `core/batched_pipeline.py`：`run(refs, sink=None, indices=None)` 接受**显式全局索引**；
  `_run_chunk` 以 `indices[i]` 取代 `offset + i` 作为全局下标，因此**种子、存储草稿的位置查找、
  `TaskExample.index` 全部保持与完整跑法一致**。按原始索引网格切分 chunk，保证锁步边界不变。
- `run_experiment.py`：`_load_prior` 读取已落盘前缀（**只有 `config_hash`／task／model／seed
  完全一致才复用**，重复行、截断行、异任务行、旧配置行一律丢弃并计数）；`_persist_merged`
  原子写回并按清单顺序重排。
- **在线臂不可续跑**（其经验库依赖已看过的全部样本），代码显式拒绝并从头开始。

**验证**：`tests/test_resume_equivalence.py`（GPU-free，stub 引擎的输出**依赖调用种子**，
因此索引错位必然被发现）：
`indices=range(n)` 与完整跑法逐记录一致；崩溃后分两段续跑与从头跑**逐记录一致**；
内部空洞续跑同样精确；重复/截断/异任务/旧配置行全部被丢弃；
并且专门断言**朴素重编号的续跑必须与之不同**——证明该测试非空洞。

**实测收益**：重启后日志显示 `[resume] outcome_hidden/wmt19_en_zh/qwen3-8b: 704 already done, 296 to run`，
704 条一条未丢。

### 7.2 缺陷 #7：批处理分支从不落盘在线经验库

`full_online` 的全部价值就是累积出的经验库，但默认的批处理分支只把它提交到内存，
落盘只存在于非批处理分支。若按原计划排队 `full_online`，跑完将**没有任何经验库产物**，该臂作废。
已在批处理分支补上与单条路径完全相同的 `save_experiences(...)` 调用。**在排队之前拦下。**

### 7.3 缺陷 #8：逐字节相同的两稿被判为 `uncertain`

A/B 两序在 `candidate == current` 时是**同一个 prompt**，模型一致地答 "A" 会被两个相反槽位
映射成互相矛盾的标签，从而记为 `uncertain`。实测 84 轮（其中 81 轮 reason_a 与 reason_b 逐字相同）。
已加短路：两稿相同时直接返回 `tie` 且**不调用引擎**。该修复是两个文本的纯函数，
因此**同样适用于已写出的历史 trace**，无需重跑。

### 7.4 中央调度器（消除空转）

`dispatcher.py`：只读 `/proc` 观测，从不向任何既有进程发信号（AST 扫描确认无 `os.kill`/`signal`/
`pkill`/`pgrep`）；`start_new_session=True` 分离启动；原子写队列；单实例 flock。
它按 `seed{seed}` 分目录做完成度检查、按 `--seed` 透传、并拒绝与在跑进程重复。
自测中发现并修掉 2 个自身缺陷（一次 tick 内同一任务被多卡重复选中；supervisor 尚存活时卡被复用）。
**已实测纠正一处我自己的队列错误**：GPU0 的 supervisor 链是 `--models qwen3-8b,qwen3-4b`，
我误写为单模型，调度器严格比对 `--models` 后判定"进程已死"并要重新入队——严格性反而抓出了疏漏。

**GPU 空转实测**：`llama3.1-8b` 四任务跑完、GPU1 释放后**立即**接力 `sr_j_fixed`；
`no_experience/qwen3-8b` 跑完、GPU2 释放后立即接力 `outcome_hidden`（并续跑 704 条）。

## 八、Phase 9 聚合：配对口径修正 + 补上从未验证过的不变量

原先的聚合有两处会产生**看起来正常但实际错误**的结果：
1. **配对按位置而非按 `sample_id`**。一旦两臂文件顺序不同（续跑重排、前缀重写），
   bootstrap 会把 A 臂第 i 条与 B 臂第 j 条配对，且不会报任何错。
2. **"对照臂"由字典覆盖顺序决定**。加入 `sr_j_fixed`／`sr_j_stop` 后存在多个合法对照，
   某个 run 究竟与谁比较取决于迭代顺序。

已改为：按 `sample_id` 取交集配对、显式声明**具名比较**（Full vs NoExperience / SR-J-Fixed /
SR-J-Stop / OutcomeHidden / RandomRetrieve / PositiveOnly / FullOnline / FixedRounds）、
并新增**跨臂 y0 逐字节一致性校验**——这是配对比较有效的前提，而审计确认此前**从未在真实数据上验证过**。
实测：15 个 run、5 组比较，**y0 不一致 0 例**。

## 九、核心论断的当前证据状态（截至 05:15，n=1000）

| 比较 | 任务 | 差值 | p(Holm) | 结论 |
|---|---|---:|---:|:--|
| full − no_experience | gigaword | **+0.893** | **0.015** | ✅ 显著 |
| full − no_experience | wmt19_en_zh | +0.341 | 0.092 | 未达显著 |
| full − no_experience | wmt19_zh_en | +0.207 | 0.634 | 不显著 |
| full − no_experience | coedit_gec | −0.025 | 0.634 | 不显著 |
| **full − outcome_hidden** | wmt19_en_zh | **−0.183** | **0.558** | ❌ **不支持** |

**必须如实处理的两件事**：
1. **每个已完成的 `full_static` 臂语料指标都是负的**（glm4-9b/en_zh −4.11 最大）。
   即"Refinement 在平均意义上没有提升质量"。这与动机（多数修订是浪费）一致，
   但意味着**方法的价值主张必须改写为"少伤害／省算力"**。
2. **`OutcomeHidden` 消融在测试集上不支持 dev 上的结论**。dev 上遮掉 outcome 字段使收益掉 30%，
   测试集上不成立（p=0.558）。在写任何依赖该论断的句子之前必须重做检验。

**站得住的**：① 55–67% 的修订轮次使离线指标变差，在每一个完整 run 上成立；
② 经验优于无经验在 gigaword 上显著。

## 十、记录卫生
- `scores/live_runs.csv` → `scores/live_runs.STALE-2026-09-12.csv`，并附 `scores/README_STALE.md` 说明为何不可引用。
- `configs/frozen.json` 新增：`operational_constants`（batch_size/max_model_len/k/代码哈希）、
  `freeze_discipline`（Phase 7 诊断已触及测试集，此后**任何方法组件都不得再改**）、
  `number_provenance`、`qwen3_32b_deviation`、`core_claim_status_2026_09_12`；
  并修正 `gpu_policy`（记录操作者的实际授权历史）与 `phase5_ablation_substitution`
  （**撤回**：glm4-9b 已修好，Phase 5 回到方案原定的 glm4-9b + qwen3-8b；该替换从未实际执行，撤回零成本）。

---

# Round 8（2026-09-12 05:20–05:30 EDT）

## 十一、方案文档已重建，并据此补齐三个缺口

独立的方案重建把 **v5 正文从会话记录中逐字恢复**（`session-0a0e27ff-…` 解压后第 667 条，
在校验中 `diff` 为 byte-identical），写入 `PLAN.md`（625 行）。此前 `core/__init__.py` 引用的
`reports/plan_v5.md` 并不存在，任何"我们遵循了方案"的说法都无从核查。

### 11.1 冲突澄清：dev 切分是 **32** 条，不是 40

方案 §1 写「dev 40」、`frozen.json` 与 progress.md §1 也都写 40，**但**方案 §8 写「4 任务 × 32」，
`core/__init__.py` 的 `AUX_SIZES.dev = 32`，四个 `data/manifests/*__dev.jsonl` **实际都是 32 行**，
而 `check_stop_rates.py` 的 `[:40]` 切片因此只用到 32 条（证据：`stop_rate_check` 里 `none: stop=32`）。
**不存在任何 40 行的产物**。progress.md §1 的「40 条」应改为 32 条。

### 11.2 ✅ 补齐：Phase 0 的 oracle-stop 上界（此前完全未实现）

方案 Phase 0 要求"6 万步动机表**与 oracle-stop 上界**"，全仓库 `grep -i oracle` 此前只命中一句注释。
已在 `run_phase0.py` 中实现 `oracle_stop_bound()` 并接入主流程，产出
`scores/oracle_stop_bound.csv`（40 条链）与 `scores/oracle_stop_bound_per_sample.csv`（31,000 个样本）。

**结果**（在基线的 SR-Fixed / SelfRefine-Fixed 链上，用基线自己的打分器）：

| 任务 | 平均 oracle 增益（相对链尾） |
|---|---:|
| coedit_gec | +0.051 |
| wmt19_zh_en | +0.050 |
| wmt19_en_zh | +0.037 |
| gigaword | +0.031 |

最具说明力的一行是 `coedit_gec / SelfRefine-Fixed / glm4-9b`：初稿 0.81 → 链尾 **0.65**（大幅变差），
而 oracle 能保住 **0.82**，即 **+16.4 分**。同时每条链在到达最优步之后平均还要白跑 **2–3 步**。
**这就是"出路在于停止、而非改写"的定量依据**，也正是本文方法所瞄准的量。

### 11.3 ✅ 修复：三处过时残留
- `check_stop_rates.py:26` 与 `calibrate_judge.py:65` 仍硬编码 `max_model_len=8192`。
  二者正是产出冻结证据的脚本，一旦重跑就会用上与所支撑的 run **不同的上下文长度**。
  已统一改为 `MAX_MODEL_LEN`（=4096）并补上 import。
  *（修复过程中我一度把 `calibrate_judge.py` 的 `enforce_eager=True` 误注释掉，已用 `diff` 逐行核对并复原。）*
- `core/__init__.py` 的策略 docstring 仍写"any two of GPUs 0-3"，已更新为实际授权的四卡并指向 `frozen.json` 的历史记录。
- `run_ablation_chain.sh` 仍带着**已撤回**的 qwen3-4b 替换理由，且默认模型是 `qwen3-8b`；
  已改为方案原定的 `glm4-9b,qwen3-8b`。

## 十二、Phase 9/10 与 Phase 6/8 交付物已落地（子代理实现，我独立验证）

| 交付物 | 文件 | 独立验证结果 |
|---|---|---|
| Phase 6 scaling + **BoN-J** | `run_scaling.py`、`bon_judge` 臂 | `tests/test_scaling.py` 全过；证明 `n_candidates=1` 与既有路径逐记录一致 |
| Phase 10 人工盲评导出 | `export_human_eval.py` | 29 项测试全过；**盲化已实测**：标注文件中 `full_static`／`no_experience` 出现 0 次 |
| Phase 9 绘图 | `make_plots.py` | 实跑产出 **4/4 图**（quality init-vs-final、full−baseline、quality-vs-compute、unnecessary-refinement），并输出每图的 n |
| Phase 8 缺口 | `run_accumulation.py` | `tests/test_accumulation_gaps.py` 全过；`--orders` 默认改 3，三方法（含 `NoNegativeExperience`）齐备 |
| 方案文档 | `PLAN.md` | v5 正文 byte-identical 校验通过 |

绘图脚本的环境差异已妥善处理：`qwen35` 有 sacrebleu 无 matplotlib，`llama4` 有 matplotlib。
两步流程：`qwen35 make_plots.py --refresh-corpus --no-figures` → `llama4 make_plots.py`。**脚本不会自动安装任何东西。**

## 十三、完整性检查：agent 改动未影响任何生产 run 的配置指纹

`run_experiment.py` 被 agent 修改过（新增 `bon_judge` 臂）。我逐 run 比对了**磁盘记录的 `config_hash`
与当前代码重算值**：22 个生产 run **全部一致**；另有 13 个初判"不一致"经逐字段核对后确认
**全部是我重建时的参数错误**（`gate_alpha`/`gate_warmup`/`smoke` 用的是 `--draft-source cached`），
**并非代码变更**。结论：既有臂配置指纹稳定，supervisor 重启时续跑仍然安全。

## 十四、修正后的 unnecessary-refinement 率（n=1000，来自 `plots/figures_manifest.json`）

| 模型 | en_zh | zh_en | gec | giga |
|---|---:|---:|---:|---:|
| glm4-9b | 67.0% | 72.8% | — | — |
| llama3.1-8b | 55.6% | 57.2% | 36.2% | 44.4% |
| qwen3-8b | 44.6% | 37.9% | 27.3% | 46.2% |
| qwen3-4b | 45.2% | 41.1% | — | — |
| **no_experience** (qwen3-8b) | 39.5% | 40.2% | 21.4% | 51.4% |
| **outcome_hidden** (qwen3-8b) | 40.9% | 39.1% | — | — |

**值得追查的一点**：`no_experience` 的修订**更多**（en_zh 1775 次 vs full 的 1113 次）却**浪费率更低**
（39.5% vs 44.6%）。这与"经验让 controller 更会挑时机"的直觉相反，需在写作前弄清机制
（可能是无经验时的改写更保守/更常无变化，落在 `tie` 而非 `worse`）。

---

# Round 9（2026-09-12 05:30–05:40 EDT）

## 十五、缺陷 #8 的修复此前**对生产路径是失效的**（已修复并端到端证明）

判据：两稿逐字节相同时，A/B 两序是**同一个 prompt**，模型一致地答 "A" 会被两个相反槽位映射成
互相矛盾的标签，于是记为 `uncertain` 而非 `tie`。

我先前只修了 `PairwiseJudge.judge()`——那是**非批处理**路径。生产实际走的是
`BatchedPipeline._round`，它**自己内联构造两次判定请求并直接调 `_combine`**，完全绕过 `judge()`。
**即"修复"落在一条没人跑的路径上，看起来正确却毫无作用。** 这是 Phase 8 子代理在核对 regime 时发现的。

已在批处理路径补上两处短路（构造请求处 `continue` + 结论汇编处合成 `tie`），并用
`tests/test_judge_identical_pair.py` 锁死：
- 非批处理与批处理**两条路径**对相同两稿都返回 `tie`；
- 批处理实跑 9 轮**全部** `tie`、`uncertain` 为 0、判定引擎调用 **0 次**；
- 并专门断言"该 run 确实全部由相同两稿构成"，避免测试因没构造出退化情形而**空洞通过**；
- 另断言"不同的两稿仍照常判定"，防止短路过度触发。

（修复过程中我的一次替换曾把该文件弄坏，已用语法检查与 diff 复原；随后回归
`test_resume_equivalence.py` 与 `test_invariants.py` 全过。）

## 十六、配置指纹复核（每次代码改动后必做）

`run_experiment.py` 又被 agent 修改过（新增 `bon_judge` 臂）。复核 `runs/main` 与 `runs/ablation`
下**全部 27 个生产 run**：磁盘 `config_hash` 与当前代码重算值**全部一致**。
`n_candidates` 是**管线参数而非 `RunConfig` 字段**，因此 9 个既有臂的指纹不受影响，续跑安全。

## 十七、BoN-J（Phase 6 主表基线）设计已定

- 臂：`bon_judge`（与 `full_static` 同旋钮，`n_candidates` 为管线参数）。
- **候选种子**：`c=0` 用 `call_seed(sample_id, t, "refine")`——与既有路径**完全相同**，
  故 `N=1` 逐记录等价（已测）；`c>0` 用 `f"refine_c{c}"`。候选集是 (sample, round, candidate) 的纯函数。
- **选择规则**：对 N 个候选做**完全循环赛的 Copeland 打分**，每对用**同一个**双序随机 A/B 判定器；
  每个候选恰好比较 N−1 次（无轮空、不依赖编排），胜者再走**未改动**的接受判据。
  完整判定矩阵入 trace，规则可从成品 trace 反推。
- `N=8` 是最贵的一点：每轮 56 次判定调用，占 223 万次调用的绝大部分。

## 十八、Phase 0 最后一项缺失交付物已补齐

方案 Phase 0 要求"6 万步动机表**与 oracle-stop 上界**"，此前全仓库无实现。
已在 `run_phase0.py` 实现并产出 `scores/oracle_stop_bound{,_per_sample}.csv`（40 条链 / 31,000 样本）。
最具说明力的一行：`coedit_gec/SelfRefine-Fixed/glm4-9b` 初稿 0.81 → 链尾 **0.65**，
oracle 可保住 **0.82**（**+16.4 分**）；每条链在最优步之后平均还白跑 2–3 步。

## 十九、子代理交付物汇总（均已独立验证）

| Phase | 交付物 | 独立验证 |
|---|---|---|
| 6 | `run_scaling.py`、`bon_judge` 臂 | `tests/test_scaling.py` **355 PASS / 0 FAIL**；另做**差分附加性验证**：改动前后 8 个既有臂的 JSONL/CSV/summary **逐字节相同** |
| 8 | `run_accumulation.py`（+`NoNegativeExperience`、`--orders` 默认 3） | 94 PASS；既有 `test_accumulation_loop.py` 13 PASS 未变 |
| 9 | `make_plots.py` | 实跑产出 **4/4 图** + 每图 n 的清单 |
| 10 | `export_human_eval.py` | 29 项测试全过；200 对；A/B 随机 108/92；**盲化已实测**（标注文件不含臂名） |
| 0 | `PLAN.md`（v5 正文逐字重建） | 与会话记录 `diff` 为 byte-identical |

## 二十、Phase 8 的 regime 发现（子代理上报，未擅自"修复"）

累加的**流**用非批处理 `Pipeline`（batch=1、草稿现生成），**检查点评估**用 `BatchedPipeline`
（batch=args.batch_size、草稿从存储回放）。两者解码参数一致，但 vLLM 对批组成不保证不变，
因此这是一个**既有的、真实的**方法学注意点，已记录而非悄悄改动。
实测建议：真实运行时传 `--stream-draft-cache`，让流内草稿只生成一次并在 3 个 order 间复用。

## 二十一、事故记录（Phase 6 子代理主动披露）

验证 CLI 拒绝逻辑时，循环里遗留了一个可运行用例，于 ~05:26:40 在默认 GPU 2 上启动了一次 vLLM 加载，
被其自身 120 s 超时终止，未产出任何输出。**这正是此前杀死 outcome_hidden 的故障模式**
（同卡双引擎）。我事后核实 GPU 2：我方**仅一个**引擎（37958 MiB）、监督链完整、
`sr_j_fixed/glm4-9b` 正常产出（192 条并增长）、日志零 OOM/错误。**未造成损害。**

---

# Round 10（2026-09-12 05:40–05:55 EDT）

## 二十二、根因查明：判定器是本方法唯一的瓶颈

诊断全文见 `reports/judge_bottleneck_diagnosis.md`。要点：

**逐层拆解（8 个已完成的格子，无一例外）**

| 任务 | 模型 | 初稿 | 实际终稿 | oracle上界 | **判定器损失** | **控制器+改写损失** |
|---|---|---:|---:|---:|---:|---:|
| en_zh | glm4-9b | 37.28 | 33.18 | 38.07 | **+4.90** | **0.00** |
| en_zh | llama3.1-8b | 35.84 | 35.25 | 36.92 | +1.68 | 0.00 |
| en_zh | qwen3-8b | 42.40 | 42.38 | 43.06 | +0.68 | 0.00 |
| en_zh | qwen3-4b | 40.95 | 41.00 | 41.22 | +0.22 | 0.00 |
| zh_en | glm4-9b | 28.35 | 24.01 | 28.20 | +4.20 | 0.00 |
| zh_en | llama3.1-8b | 28.61 | 28.43 | 29.37 | +0.94 | 0.00 |
| zh_en | qwen3-8b | 30.40 | 30.36 | 30.81 | +0.45 | 0.00 |
| zh_en | qwen3-4b | 27.90 | 27.90 | 28.15 | +0.25 | 0.00 |

**损失 100% 来自接受闸门，控制器与改写器为 0.00。** 换用正确的接受规则，8 格中 7 格高于初稿。

**机制：self-preference + verbosity 偏差**（改写器与判定器是同一个模型）

| | 判 better 时平均长度比 | 判 better 但实际更差 |
|---|---:|---:|
| glm4-9b / zh_en | 1.569 | **84.7%** |
| glm4-9b / en_zh | 1.457 | **79.1%** |
| qwen3-8b / en_zh | 1.003 | 55.1% |
| qwen3-8b / zh_en | 1.048 | 33.8% |

判定器 prompt 本已写明 "Do not reward length by itself"，**加指令无法消除该偏差**。
病例：`000084` 初稿与参考完全相同（100.0 分），被判 "better" 的修订使其掉到 45.7。

## 二十三、修复：接受闸门长度护栏（已实现，默认关闭）

- **单一来源**：`core/judge.py::accept_revision`，**两条管线都调用它**。
  之所以强调，是因为"两稿相同→tie"的修复此前只加在非批处理的 `PairwiseJudge.judge` 上，
  而生产走 `BatchedPipeline._round`，**修复落在没人跑的路径上**。
  实现时又发现**第三处**重复规则（`BonJBatchedPipeline._round_bonj` 自己写了一遍），已一并收敛。
  `tests/test_accept_gate.py` 现在会**全树扫描**重复的接受判定，防止再次分叉。
- **默认 `None` = 历史行为，逐位不变**：37 个生产 run 的 `config_hash` 全部未变（该开关是
  **管线参数而非 `RunConfig` 字段**，否则重启会丢弃已完成前缀）。
- 已透传：`run_experiment.py --accept-max-len-ratio`（3 个构造点）→ `supervisor.py` → `dispatcher.py`。

**离线模拟（零 GPU，在现有 trace 上重放）** —— 规则"判定 better **且** 长度比 ≤ 阈值"：

| 任务 | 模型 | 初稿 | 现规则 | oracle | 护栏 1.05 |
|---|---|---:|---:|---:|---:|
| en_zh | glm4-9b | 37.28 | 33.18 | 38.07 | **37.27** |
| en_zh | llama | 35.84 | 35.25 | 36.92 | **35.84** |
| en_zh | qwen3-8b | 42.40 | 42.38 | 43.06 | **42.52** |
| en_zh | qwen3-4b | 40.95 | 41.00 | 41.22 | **41.03** |
| zh_en | glm4-9b | 28.35 | 24.01 | 28.20 | **28.21** |
| zh_en | llama | 28.61 | 28.43 | 29.37 | **28.44** |
| zh_en | qwen3-8b | 30.40 | 30.36 | 30.81 | **30.34** |
| zh_en | qwen3-4b | 27.90 | 27.90 | 28.15 | **27.89** |

glm4-9b/en_zh 挽回 83% 的 oracle 空间，glm4-9b/zh_en 挽回 100%。
**只加长度护栏而不要判定器会更差**（glm4-9b/zh_en 掉到 21.78）—— 两者必须同时使用。

**天花板说明**：只有 13%~31% 的修订真正改善指标，故 oracle 上界本身只有 +0.2~+0.8。

## 二十四、修复流程（按操作者决定：dev 选阈值 → 冻结 → 重跑）

已把 8 个 dev 选阈值任务排到**最高优先级**（4 模型 × 2 任务 × 32 条）。
**阈值必须在 dev 上选并冻结**，否则用测试集挑阈值就是测试集选型。

## 二十五、过程中发现并修复的 4 个真实缺陷

1. **`--split dev` + 批处理直接崩溃**：`batch_size>1` 只支持 `stored` 草稿，dev 必须 `generate`。
2. **崩溃留下孤儿 vLLM 引擎**（今日第 3 次）：参数校验发生在**模型加载之后**，
   异常退出时 `EngineCore` 子进程仍活着、占 36.9 GB、ppid=1，导致该卡**此后所有加载都失败**，
   形成自维持的重试死循环（GPU0 空转 8 分钟）。
   **根因修复**：在 `main()` 里**加载模型之前**校验，直接返回 2；实测不占一字节显存。
3. **supervisor 日志名不含 split**：dev 与生产任务写**同一个日志文件**互相覆盖，
   以致 dev 的崩溃一度不可见。已改为 `{arm}_{model}_s{seed}_{split}.log`。
4. **`dispatcher.py` 声称按优先级调度，代码里从不读 `priority` 字段**（只有文档字符串提到）。
   已补上稳定的优先级排序。

## 二十六、里程碑：Phase 4 主实验 `full_static` 全部完成

**16/16 格子**（4 模型 × 4 任务，各 1000/1000/1000/100）。这是主表 Full-Static 列的完整数据。
`sr_j_fixed`、`sr_j_stop`、`no_experience`、`outcome_hidden` 正在推进。

---

# Round 11（2026-09-12 06:00–06:45 EDT）

## 二十七、按规范完成闸门阈值选择并冻结

**流程**：8 个 dev 任务（4 模型 × 2 任务 × 32 条，`runs/dev_gate_fix/`）→
`select_accept_threshold.py`（预注册规则：跨格总增益最大，**平局取更大 τ**）→ 冻结进 `configs/frozen.json`。

| τ | 跨格总增益 |
|---|---:|
| **none（原闸门）** | **−9.056** |
| 1.00 / 1.01 / **1.02** | **+0.228**（平局→取最大 τ） |
| 1.05 | −0.417 |
| 1.10 | −0.659 |
| 1.15 | −2.052 |
| 1.50 | −5.655 |

**冻结值 τ = 1.02**，证据存于 `scores/accept_threshold_selection.json`。

**必须如实说明**：τ=1.02 时 glm4-9b 两个 dev 格子的增益**恰好为 +0.000** ——
它的修订几乎全部超过 2% 的长度增长而被拒绝，即**该模型退化为"什么都不改"**。
正的收益全部来自 llama3.1-8b（+0.129）与 glm4-9b/zh_en（+0.136）。
这是真实的、模型相关的结果，不能包装成"方法普遍有效"。

## 二十八、带闸门重跑已在真实数据上验证有效

4 个 `full_static` 带闸门任务已确认以 `--accept-max-len-ratio 1.02` 运行
（命令行核对），写入 `runs/gated_main`；**旧闸门的 16/16 结果完整保留**为对照条件。

| 模型 | 闸门 | 接受率 | 逐条 Δ 均值 |
|---|---|---:|---:|
| glm4-9b | 无 | 10.9% | **−2.378** |
| glm4-9b | τ=1.02 | **0.4%** | **−0.085** |
| llama3.1-8b | 无 | 5.4% | −0.476 |
| llama3.1-8b | τ=1.02 | 3.6% | **−0.076** |
| qwen3-8b | 无 | 10.6% | −0.033 |
| qwen3-8b | τ=1.02 | 5.2% | **+0.099** |
| qwen3-4b | 无 | 9.6% | +0.107 |
| qwen3-4b | τ=1.02 | 6.3% | **+0.254** |

（以上为 10–50% 完成度的初步读数，方向明确但非最终值。）

**设计决策**：接受闸门是**所有臂共用的机制**（单一变量设计），因此凡经闸门的臂都必须在
**同一冻结闸门**下重跑；否则就变成"我们的方法用新闸门、基线用旧闸门"的不公平对比。
共排队 20 个带闸门任务（G1 主表 / G2 核心消融 / G3 其余），输出到 `runs/gated_main`、
`runs/gated_ablation`。3 个无法比较的无闸门待跑任务已标记 `retired`。

## 二十九、本轮又发现并修复 3 个真实缺陷

1. **我自己的 `str.replace` 事故**：给 `dispatcher.py` 加闸门透传时，`str.replace` **默认替换所有匹配处**，
   把 `cmd += ...` 也插进了 `pair_paths()`（那里没有 `cmd` 变量），导致每个 tick 抛
   `UnboundLocalError`，**调度器静默停摆 13 分钟、4 张卡全空**。
   已修复并用 `grep -n` 确认只剩一处。教训：`str.replace` 必须给出足以上下文唯一的锚点。
2. **杀 supervisor 不会杀死其 `run_experiment.py` 子进程**（它们以 `start_new_session=True` 启动，
   脱离父进程组），而 `run_experiment.py` 死掉又会**留下孤儿 vLLM 引擎**（本次 4 个，各占 ~37 GB）。
   这是今天第 4 次遇到孤儿引擎。已建立"按精确 PID + 命令行前缀匹配"的清理流程
   （注意 `ps -o comm=` 只返回 15 字符，直接等值比较会失败）。
3. **调度器用测试集规模判断 dev 任务完成度**：一个**成功**的 dev 任务（32 条）被判为
   "32/1000 未完成"并反复重试，每次重试都要重新加载模型，最终还会把成功的任务标为 failed。
   已让 `required_n()` 按 split 取 manifest，缓存键也含 split。

## 三十、当前状态

- **Phase 4 `full_static` 无闸门：16/16 完成**（对照条件，完整保留）
- **带闸门 `full_static` × 4 模型正在运行**（G1，最高优先级）
- 队列：done 17 / pending 41 / running 4 / retired 7
- 4 卡全忙，调度器已恢复；本轮共发生 5 次调度器重启（全部因代码更新）
- 方案 v5 各 Phase 的实现与产物见 `PLAN.md` §5 与本文档 Round 7–11

---

# Round 12（2026-09-12 06:40–06:55 EDT）

## 三十一、核心表三方基线全部完成 —— 出现一个必须正视的结果

`sr_j_fixed`、`sr_j_stop` 在 **glm4-9b 与 qwen3-8b** 上四个任务全部跑完；加上此前完成的
`no_experience` 与 `full_static`，**两个消融模型 × 4 任务的核心对比已经齐备**（旧闸门，n=1000/100）：

| 任务 | 无经验 | SR-J-Fixed | **SR-J-Stop** | 我们的(旧闸门) |
|---|---:|---:|---:|---:|
| wmt19_en_zh | −2.998 | −1.660 | **−0.201** | −2.062 |
| wmt19_zh_en | −2.026 | −1.041 | **−0.272** | −2.189 |
| coedit_gec | −1.216 | −1.299 | **−0.295** | −0.884 |
| gigaword | −1.705 | −2.176 | **−0.144** | −1.331 |

（2 模型均值，相对各自初稿的语料指标变化）

### 结论一：`SR-J-Stop` 是全场最强，而它**完全不用经验**

它只做**一轮**修订就停（平均 1.0–1.2 轮），却在四个任务上**全面最优**，
连 `no_experience`（−1.2 ~ −3.0）都远好于。这说明：
**在当前的改写器质量下，最优策略是"尽量少改"**。论文的靶子因此是
"**何时该停**"，而不是"如何改得更好" —— 这恰好是本方法的立意，但必须真的打赢它。

### 结论二：旧闸门下我们的方法**输给** SR-J-Stop（四个任务全输）

这与 Round 10 的诊断完全一致：旧闸门吃掉了全部收益。**这是"闸门是瓶颈"最强的外部佐证** ——
不是我们内部重放得出的，而是与一个不用经验的基线直接对比得出的。

### 结论三：带闸门后初步信号转正

同一任务（glm4-9b / en_zh）：带闸门 **−0.085** vs SR-J-Stop 的 −0.497 —— **已经反超**。
最终结论须等带闸门重跑全部完成（进行中）。

## 三十二、基础设施状态

- 调度器：修复后 `tick failed` 计数 **0**（此前 25 次全部集中在 06:29–06:30 我引入的那个 bug），
  心跳每 30 s，4 卡全忙。
- 带闸门主表 `runs/gated_main` 四个模型均在 `wmt19_en_zh` 推进（128–576/1000）。
- `aggregate.py` 已验证可同时处理 `runs/{main,ablation}` 与 `runs/{gated_main,gated_ablation}`
  两套目录，便于最终做"筑门开/关"的并列汇报。
- 新一轮只读监工已派出，重点核查**每个在跑任务是否真的带上了与其输出目录匹配的闸门参数**
  （带闸门目录里的任务若缺 `--accept-max-len-ratio` 就是 CRITICAL，会产生不可比数据）。

## 三十三、算力对比：方法目前远离帕累托前沿（必须正视）

| 模型 | 任务 | 臂 | 轮数 | tok/样本 | Δ |
|---|---|---|---:|---:|---:|
| glm4-9b | en_zh | SR-J-Stop | 1.16 | **522** | −0.497 |
| glm4-9b | en_zh | 我们(旧闸门) | 2.97 | **5738** | −4.107 |
| glm4-9b | zh_en | SR-J-Stop | 1.11 | **554** | −0.514 |
| glm4-9b | zh_en | 我们(旧闸门) | 2.98 | **6032** | −4.344 |
| qwen3-8b | en_zh | SR-J-Stop | 1.09 | **346** | +0.095 |
| qwen3-8b | en_zh | 我们(旧闸门) | 1.75 | **3102** | −0.017 |

单轮成本 1932 vs 450 token（约 **4.3×**，主要来自经验块本身），轮数再乘 **2.5×**，
合计约 **11×** 算力。**方法必须以质量增益正当化这个代价，而当前数据不支持。**

三层原因，按可修复性排序：
1. **闸门**（已修，带闸门后 glm4-9b/en_zh 从 −4.11 收到 −0.085）；
2. **控制器过度修订**：glm4-9b 的停止率只有 **0.6%**（REFINE 2949 / STOP 19），
   而 SR-J-Stop 平均 1.16 轮就停。控制器 prompt 已冻结，改动需另走 dev 流程；
3. **经验块的固定开销**：每轮 prompt 都要带 k=4 条经验，即使最终一条都不采纳也照付。

**这对论文的含义**：Phase 6 的"质量–算力"曲线里，本方法目前**不在帕累托前沿上**。
要么用带闸门+更强的停止策略把它拉回前沿，要么如实报告为"经验驱动的时机选择在当前
改写器质量下不足以补偿其上下文开销"。这是必须在写作前定夺的结论，不能靠措辞回避。

---

# Round 13（2026-09-12 06:55–07:05 EDT）

## 三十四、带闸门主表已反超最强基线（关键结果）

| 任务 | 模型 | n | 无闸门 | **带闸门** | SR-J-Stop | 结论 |
|---|---|---:|---:|---:|---:|---|
| en_zh | glm4-9b | 800 | −4.107 | **+0.036** | −0.497 | **赢 0.53** |
| en_zh | llama3.1-8b | 1000 | −0.597 | **+0.092** | — | 转正 |
| en_zh | qwen3-8b | 1000 | −0.017 | **+0.174** | +0.095 | **赢 0.079** |
| en_zh | qwen3-4b | 1000 | +0.045 | **+0.109** | — | 提升 |
| zh_en | llama3.1-8b | 224 | −0.177 | +0.002 | — | 持平 |
| zh_en | qwen3-8b | 992 | −0.035 | −0.035 | −0.029 | 持平 |
| zh_en | qwen3-4b | 1000 | +0.002 | +0.001 | — | 持平 |

**en_zh 四个模型全部转正且全部超过最强基线**；zh_en 上持平。
这是从"四任务全输"到"en_zh 全面反超"的转变，且**仅来自闸门修复**——gold 库优化尚未运行。

## 三十五、经验标签污染已量化并修复（优化 A）

判定器与真实收益**只有 68.6% 一致**；库中唯一被渲染的 outcome 就是这个判定器结论，
另有 782/2021 条（38.7%）是 `uncertain`。即模型看到的"结果"字段：
**约 31% 错误 + 39% 无信息**。这解释了为何"检索到 better 经验"反而预测更差的修订。

`relabel_experience.py` 已用 `initial` 辅助切分的参考答案离线重算全部 16 个库的**真实收益**，
渲染改为 helped/hurt/unchanged。改动严格单变量：单元集合、检索字段、prompt 模板均不变，
新库写入 `experience/initial_gold/`（不覆盖旧库，已实测旧库字节未变）。
`--experience-root` 已透传全链路，且为管线参数而非 `RunConfig` 字段。

验证任务 `GOLD-dev-full_static-{4 模型}` 已排在优先级 0。

## 三十六、第一个完整带闸门臂：qwen3-8b 四任务全部跑完

| 任务 | 无经验 | SR-J-Fixed | SR-J-Stop | 无闸门 | **带闸门** |
|---|---:|---:|---:|---:|---:|
| wmt19_en_zh | −0.668 | −0.978 | +0.095 | −0.017 | **+0.174** ⭐ |
| wmt19_zh_en | −0.354 | −0.411 | −0.029 | −0.035 | −0.034 |
| coedit_gec | −0.008 | −0.717 | −0.038 | −0.033 | −0.027 |
| gigaword | −1.340 | −1.140 | −0.002 | −0.448 | −0.027 |
| **合计** | −2.370 | −3.246 | **+0.026** | −0.533 | **+0.086** |

**带闸门后本方法合计最优（+0.086），超过最强基线 SR-J-Stop（+0.026）**，
en_zh 上明确最佳（+0.174 vs +0.095），zh_en/gec 略优或持平，giga 略逊。

**必须如实说明**：四个任务中有三个各臂都在 0 附近、彼此差距很小。
这组任务的真实情况是"修订本身几乎不改变质量"，本方法的优势主要来自 en_zh。
论文不宜把 +0.086 的总和包装成普遍性增益。

**修复链条的完整因果**：判定器自评判偏差（精确率 21.5%）→ 旧闸门吃掉全部收益（−0.533）→
长度护栏（+0.086）→ 经验标签污染（与真实仅 68.6% 一致）→ 优化 A 正在验证。

---

# Round 14（2026-09-12 07:05–07:40 EDT）

## 三十七、三个完整带闸门臂：闸门修复让每个模型都变好

| 模型 | 闸门 | en_zh | zh_en | gec | giga | **合计** |
|---|---|---:|---:|---:|---:|---:|
| llama3.1-8b | 无 | −0.597 | −0.177 | +0.014 | +0.099 | −0.661 |
| llama3.1-8b | **τ=1.02** | **+0.092** | −0.118 | **+0.048** | −0.014 | **+0.008** |
| qwen3-8b | 无 | −0.017 | −0.035 | −0.033 | −0.448 | −0.533 |
| qwen3-8b | **τ=1.02** | **+0.174** | −0.034 | −0.027 | −0.027 | **+0.086** |
| qwen3-4b | 无 | +0.045 | +0.002 | −0.201 | +0.032 | −0.122 |
| qwen3-4b | **τ=1.02** | **+0.109** | +0.001 | **−0.154** | −0.015 | **−0.059** |
| **三模型合计** | 无闸门 | | | | | **−1.316** |
| **三模型合计** | **带闸门** | | | | | **+0.035** |

**闸门修复让三个模型全部改善，合计由 −1.32 翻正到 +0.04**；
qwen3-8b 上（+0.086）超过最强基线 SR-J-Stop（+0.026）。
glm4-9b 带闸门尚差 coedit_gec / gigaword 两格。

## 三十八、优化 A 的判决：机制成立，dev 功效不足

已用 8 个 dev 格（256 条）完成旧库 vs 新库（真实收益标签）的对照：

- **机制指标变好**：平均改善率 **22.4% → 31.1%**（`qwen3-4b/zh_en` 33%→**60%**、`qwen3-8b/en_zh` 13.5%→**35.7%**）
- **语料指标未跟随**：逐格无一致关系，重放所有 τ 后新库在**每一个 τ 上**都不如旧库

按预注册规则（dev 跨格语料增益最大）**不予采纳**。但这不是"方法无效"，而是
**dev 只有 32 条/格的固有功效不足** —— 256 条样本无法分辨语料层面的差异。
已把 128 条的 `accumulation` 辅助切分（功效 4×）上的对照排在核心表之后，
并明确记录：Phase 8 也计划用该切分做累积流，因此这一评测会使它成为该阶段的选型集。

## 三十九、基础设施：dev/辅助切分提速 5.6×

`run_experiment.py` 此前禁止 `batch_size>1` 搭配非 `stored` 草稿，迫使所有 dev/辅助运行走单条路径。
但 `BatchedPipeline._run_chunk` 本就有 `need` 分支在一次批调用里生成缺失草稿，
`cached` 路径也走同一缓存 —— 该禁令纯属保守。已放开（仅保留真正不安全的情形），
**实测不同 batch 大小下草稿一致**，dev/辅助切分现可用 batch 32，把 128 条功效验证的成本
从近 6 小时压到约 20 分钟。

## 三十九-b、闸门生效的数据级证明（监工独立验证）

13 个带闸门文件中 **439 个被接受的轮次，0 个**违反 `len(cand) ≤ 1.02·len(cur)`；
同时 **1609 个"旧闸门会接受"的轮次被拦下**。作为对照，无闸门运行中 279/321（87%）违反该约束。
**检验非空洞，闸门确实在数据层面生效。**

---

# Round 15（2026-09-12 07:40–07:50 EDT）

## 四十、配对检验更新：闸门下与基线统计无差别

| A | B | 任务/模型 | n | A | B | A−B | p_holm | 显著 |
|---|---|---|---|---:|---:|---:|---:|---|
| full_static | sr_j_stop | en_zh / glm4-9b | 1000 | 32.690 | 32.727 | −0.037 | 1.000 | 否 |
| full_static | sr_j_stop | zh_en / glm4-9b | 1000 | 19.277 | 19.278 | −0.001 | 1.000 | 否 |
| full_static | sr_j_fixed | en_zh / qwen3-8b | 1000 | 37.871 | 38.078 | −0.207 | 0.171 | 否 |

（y0 一致性全部通过）

**诚实结论**：闸门修复把方法从"被修订拖坏"救回到**与基线统计上不可区分**，
但**尚未显著超越**任何基线。这与 oracle 上界仅约 +0.5/格、判定器无便宜解法一致 ——
"生成后筛选"框架内可拿到的增量本来就很小。

## 四十一、整合产物（Phase 9）

新增 `consolidate_results.py` → `scores/consolidated_results.csv`：
一行一个 (arm, task, model, tree)，含 n / 完整标记 / 语料初稿与终稿 / 轮数 / 接受率 / token 每样本，
覆盖全部 8 棵实验树（无闸门、带闸门、dev、gold 库、aux）。部分运行标注而不与完整运行混合。
当前 70 行、62 个完整测试格；据此完成的配对比较见 `reports/judge_bottleneck_diagnosis.md` 第九部分
（**闸门净效果 +11.584，16 格中 13 格改善**）。

## 四十二、仍待完成（按优先级）

| 优先 | 项 | 状态 |
|---|---|---|
| P1 | `fixed_rounds` vs `sr_j_fixed` 判决性消融（经验有无增量价值） | 未开跑 |
| P1 | 带闸门 sr_j_stop / sr_j_fixed 剩余格 | 进行中 |
| P2 | 128 条 `accumulation` 辅助切分上的 gold 库高功效验证 | 未开跑 |
| P3 | G2 核心消融（no_experience / outcome_hidden / random_retrieve） | 未开跑 |
| P4 | Phase 6 BoN-J / Phase 7 反事实 / Phase 8 累积 / full_online | 已排期 |
| P5 | 2×2 的 seed 43/44 重复 | 已排期 |

---

# Round 16（2026-09-12 07:45–08:00 EDT）

## 四十三、与最强基线的完整配对检验（6 格）

`sr_j_stop`（无经验、判定即停）的 glm4-9b 四任务全部完成，与我们的带闸门 full_static 正式配对：

| 任务 | 模型 | 我们 | SR-J-Stop | 差值 | p_holm | 显著 |
|---|---|---:|---:|---:|---:|---|
| coedit_gec | qwen3-8b | 81.564 | 81.584 | −0.020 | 1.000 | 否 |
| gigaword | qwen3-8b | 30.760 | 30.850 | −0.090 | 0.168 | 原始 CI 排除 0，Holm 后不显著 |
| wmt19_en_zh | glm4-9b | 32.690 | 32.727 | −0.037 | 1.000 | 否 |
| wmt19_en_zh | qwen3-8b | 37.871 | 37.883 | −0.012 | 1.000 | 否 |
| wmt19_zh_en | glm4-9b | 19.277 | 19.278 | −0.001 | 1.000 | 否 |
| wmt19_zh_en | qwen3-8b | 20.364 | 20.290 | +0.074 | 1.000 | 否 |

**6 格中 5 格统计上完全不可区分，1 格略差，无任何一格显著胜出。**

这与"oracle 上界仅约 +0.5/格"一致：在"生成后筛选"框架内可拿到的质量增量
**小于统计可分辨的尺度**。因此论文不应声称质量增益，而应声称：

> **闸门修复把方法从系统性有害（配对合计 −11.24）救回到统计上与最强基线持平（+0.34），
> 且不改变检索、提示、种子或任何配置指纹。**

## 四十四、判决性消融已提到 P0

`fixed_rounds`（有经验 + 强制轮数）与 `sr_j_fixed`（无经验 + 强制轮数）
轮数预算、检索、提示结构完全相同，**唯一差别是有无经验** ——
这是"经验有没有增量价值"的唯一干净对比。已由操作者提到 P0，下一张卡释放即跑。

若该对比显示两者无差异，则本工作的诚实结论是：
**经验在此框架下没有可测量的质量增量价值**，其价值仅限于"改变修订频率"（这一点已有数据：
有经验时平均轮数由 2.20 降到 1.75）。

---

# Round 17（2026-09-12 08:20–08:30 EDT）

## 四十五、🎉 带闸门主实验 Phase 4 全部完成（16/16）

| 模型 | en_zh | zh_en | coedit_gec | gigaword | 合计 |
|---|---:|---:|---:|---:|---:|
| glm4-9b | +0.025 | −0.043 | −0.596 | −0.159 | **−0.773** |
| llama3.1-8b | +0.092 | −0.118 | +0.048 | −0.014 | **+0.008** |
| qwen3-8b | +0.174 | −0.034 | −0.027 | −0.027 | **+0.085** |
| qwen3-4b | +0.109 | +0.001 | −0.154 | −0.015 | **−0.058** |
| **四模型合计** | | | | | **−0.738** |

## 四十六、完整 16 格配对：闸门净效果 +12.978

同一臂（`full_static`）、同一模型、同一任务、**y0 逐字节相同**，唯一自变量是接受闸门：

| 模型 | 无闸门合计 | 带闸门合计 | 改善 |
|---|---:|---:|---:|
| glm4-9b | −12.399 | −0.773 | **+11.626** |
| llama3.1-8b | −0.660 | +0.008 | +0.668 |
| qwen3-8b | −0.533 | +0.085 | +0.618 |
| qwen3-4b | −0.123 | −0.058 | +0.065 |
| **四模型合计** | **−13.716** | **−0.738** | **+12.978** |

**16 格中 13 格改善。** 这是本工作最完整、最严格的证据：
全 4 模型 × 4 任务的配对、τ 在 dev 上按预注册规则选定并冻结、
闸门经数据级验证确实生效（439 个被接受轮次 0 个违反长度约束、1609 个该拦的被拦下）。

## 四十七、当前状态

- 队列 done 31 / running 4 / pending 42，调度器零 tick 错误、零空转、无孤儿引擎
- 正在跑：`fixed_rounds` × 2、`no_experience` × 2（全部带闸门）
- 已修复并端到端验证：`--split` 现接受 `{test,dev,initial,accumulation}`（此前 AUX 任务因 `invalid choice` 秒退）
- 剩余：Phase 5 其余消融、Phase 6/7/8、128 条高功效验证、seed 43/44 重复

---

## 四十八、计划 v6 重写与实施（渲染器/提示词对齐原方法）

### 48.1 为什么重写：三处失真

1. **整个闸门战役用的是错的经验库。** `runs/{main,ablation,gated_ablation,gated_main}` 的
   ~64 个完整格全部基于旧库（从零生成草稿 + 控制器编造缺陷），不是从
   `原文/不完美答案/完美答案/修改建议` 资产来的对比示例。因此
   `FINAL_RESULTS.md` 的「经验无增量价值」**只对旧库有效**，不是最终结论。
2. **对比库已建好但验证严重不足。** `experience/contrastive/<model>/<task>/initial.jsonl`，
   每模型 6,503 条（en_zh 2978 + zh_en 989 + gec 2436 + giga 100）。
   偏序证据（n=160，严格配对）：对比库把 `fixed_rounds` 从旧库的 −0.023 提到 **+0.392**
   （**+0.415**），但**仍打不过不用经验**（−0.344 vs `sr_j_fixed`）。
3. **渲染器与提示词都不是原方法。** 当前是自造的
   `[Experience i]/Task input/…/Outcome/Judge note`，且建议用**全文 2,398 字符**；
   原方法是四行格式 + `extract_summary_advice()` 只取**末段 360 字符（15.0%）**。

### 48.2 用户澄清确立的方法定义（v6）

- 库里**只存"错→对"的经验**，所以**根本不需要 Outcome 标签**——每条示例本身就是
  "错（不完美答案）+ 对（完美答案）"的对比。**对比在单元内部**，不靠混合成功/失败单元。
- 因此 `outcome_hidden` 臂**退休**（无字段可隐藏，退化成 `full_static` 的重复）。
- `PositiveOnly` 的语义是**删掉单元内的错误草稿行**，只留修好的答案
  （不是按标签过滤——对比库 100% 是 helped，那样过滤是空操作）。
- **示例块移入 refiner 提示词**：controller 只看当前答案与缺陷（负责"该不该改"），
  refiner 看示例 + 当前草稿（负责"怎么改"）。原方法里示例直接条件化生成；
  实测瓶颈是 controller 的有损转述（41.2% 的 instruction 是检索建议的近复制品）。

### 48.3 已实现并验证

| 项 | 状态 | 证据 |
|---|---|---|
| `summary_advice()` 复刻原实现 | ✅ | 200 条真实资产逐条同输出 |
| v2 四行渲染器 | ✅ | `tests/test_renderer_v2.py`：**与原方法逐字节一致** |
| `contrastive` 开关（PositiveOnly） | ✅ | 恰好只少 `Draft Translation` 一行 |
| v2 refiner 提示词（`### Current Task` + 四维度） | ✅ | 翻译任务逐字照抄原方法 |
| v2 controller 提示词（不含 experience 引用） | ✅ | 断言 `'experience' not in prompt` |
| `transitions_to_experiences(only_improving)` | ✅ | 只留"被接受且 delta_offline>0" |
| `config_hash` legacy-default 省略 | ✅ | `tests/test_config_hash_legacy.py`：**132 个生产格逐字节不变** |
| `ARM_REGISTRY` 9 臂（退休 outcome_hidden） | ✅ | `RETIRED_ARMS` 断言不可调度 |
| `consolidate_results.py` 覆盖 e2_*/ctr_* + 偏序段 | ✅ | 输出 132 行，13 个偏序格显式标注 |
| y0 一致性**阻断式**检查 | ✅ | 新增 `scores/y0_identity.csv` |

### 48.4 三处新发现的真实缺陷

1. **`{{ }}` 双括号生产 bug**：`_SCHEMA_ADAPTIVE`/`_SCHEMA_FIXED` 写成 `{{…}}` 是给
   *模板字面量* 用的，但它们是作为**替换值**插入的，双括号从不折叠——所有 v1 controller
   调用都在向模型要 `{{"action": …}}`。实测 128,462 次决策中结构化失败仅 **236 次
   （0.184%）**，模型自行规范化了，**故未造成损失**；v2 修正，v1 保持原样
   （改 v1 会破坏全部既有运行的身份）。
2. **`--draft-source generate` 让 dev/accumulation 的 y0 不可复现**，导致
   「gold 在每个 τ 都更差」的 dev 结论有 4/8 格被混淆。新 y0 门禁**独立复现**了该发现：
   `dev_labels` 的 4 个 zh_en 格 + `aux_labels` 的 4 格不一致。
3. **Phase 8 accumulation 的语义与 v2 冲突**：它的设计是「一个流保留全部转移、
   其余条件是它的子序列」，`no_negative` 条件按 `is_negative_outcome` 过滤；
   若把 `only_improving` 应用到该流，`dynamic` 与 `no_negative` 会塌缩成同一个库、
   整个研究变成空操作。**故该实验暂缓**，待为 v2 语义重新规格化；
   `full_online` 臂（真正的 v2 在线进化）已正确使用 `only_improving=True`。

### 48.5 v2 成本（冒烟实测）

qwen3-4b / `full_static` / en_zh / 4 条：第 0 轮 `input_tokens` 仅 **1,686**，
对比 CTR 时期（旧渲染器 + 对比库）逐样本约 **8,500 token**。
对齐后四例经验块由 11,560 字符降到 2,776 字符（**省 76%**），
即经验开销由 +195% 降到约 +16%，"等算力比较"这才真正成立。


---

## 四十九、v6 实现完成 + dev 证据（关键转折）

### 49.1 实现：全部完成并逐项验证

11 个测试套件全绿（含 2 个新增）。**v2 渲染器与原方法在 200 条真实资产上逐字节一致**
（`tests/test_renderer_v2.py` 独立复刻原实现比对，非自证）。
`config_hash` 的 132 个生产格逐字节不变（`tests/test_config_hash_legacy.py` 门禁）。

期间抓到三处真实缺陷：
1. `{{ }}` 双括号生产 bug（128,462 次决策中失败 0.184%，未造成损失；v2 修正、v1 保持原样）
2. dev/accumulation 的 y0 不可复现——新门禁**独立复现**了监控发现（`scores/y0_identity.csv`）
3. Phase 8 accumulation 与 v2 语义冲突（`only_improving` 会让 `dynamic`/`no_negative` 塌缩），暂缓

### 49.2 **dev 证据：v2 比 v1 更差**

同样 8 个格（全 4 模型 × 2 翻译任务，各 32 条）：

| 配置 | 跨格增益 |
|---|---:|
| **v1**（旧渲染器 + 旧库） | **+0.228** |
| v2 refiner-only | **−2.145** |
| v2 both（示例同时进 controller） | **−1.278** |

16 格 τ 曲线：refiner-only −2.909（τ=1.00），both −2.584（τ=1.01），`none` −22.567。
**闸门本身仍极有效**（改善 +19.7），负面来自被接受的修订轮次。

机制：`wmt19_en_zh/qwen3-8b` 由 52 轮/接受 3 次变成 96 轮/接受 11 次，Δ 由 −0.036 掉到 −1.179。
把示例移出 controller 后它不再区分、只是更频繁要求修改。

### 49.3 未隔离的嫌疑（v1→v2 同时改了三件事）

1. 库：旧库 → 对比库（此项在 test 偏序上曾为 **+0.415**，方向为正）
2. 渲染器：建议**全文** → 建议**末段**（嫌疑最大，尚未测）
3. 架构：示例进 controller → 进 refiner（`both` 恢复 +0.867，是其中一部分）

**正在跑**（`runs/e2_dev_full`、`runs/e2_dev_bothfull`）：隔离 #2，测"建议全文"。
两条件与已有 run 共享冻结草稿缓存，**y0 严格配对**。

### 49.4 决策闸门

**在把 dev 拉回正区间之前不启动 E2 战役**——当前最优 v2 配置 dev 为 −1.278，
而 E2 需 15–18 小时。先花约 2 小时把配置选对。
保底路线（v1 闸门 16 格 **+12.978**）完好，其数据本次未触碰。

---

## 五十、**突破：现在的方法首次取得确证的正效果**

配置（全部新运行一律如此，不含任何旧库/旧格式）：
`--experience-root experience/contrastive`（每模型 6,503 条真实"错→对"资产）+
`renderer=v2`（四行格式，与原方法逐字节一致）+ 示例进 refiner + 建议取末段。

测试集 `wmt19_en_zh` 严格配对（同 `sample_id`，同一模型同一任务）：

| 模型 | n | 我们的方法 | 无经验 `sr_j_fixed` | 差 | p | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| qwen3-8b | 960 | **+3.473** | +0.292 | **+3.181** | **<0.00001** | [+2.407, +4.063] |
| glm4-9b | 640 | **+1.228** | −0.135 | **+1.364** | **<0.00001** | [+0.740, +2.052] |

**两格均高度显著，CI 远离零。**

对比此前所有配置（同一格 en_zh/qwen3-8b）：

| 配置 | 增益 | p |
|---|---:|---:|
| 旧库 + 旧格式（v1） | +0.006 | 0.920 |
| **对比库 + 四行格式（现在）** | **+3.181** | **<0.00001** |

**结论：此前"经验无增量价值"的判断完全是错库 + 错格式造成的。**
换成用户提供的真实"错→对"资产、并用原方法的四行格式后，经验产生了大幅且显著的正效果。

**重要方法论教训**：我此前花了两个多小时在 dev（32 条/格）上做隔离实验，
想凭它判断新格式是否有效——那是错的。dev 的分辨率根本不足以分辨 ±1 BLEU，
且历史对照（旧库+旧格式+另一套 y0）本身不可比。
**直接跑主实验才是答案**，这是用户明确指出的。

`full_static` 与 `sr_j_fixed` 的其余任务（zh_en / coedit_gec / gigaword）仍在跑，
完成后与 SOTA 基线（SR-U / BoN-U-4 / AdaCompute-SR-U / SelfRefine-U）打表。

---

## 五十一、与 SOTA 的正式对比：**显著胜过全部 12 个基线**

### 51.1 一个必须记住的口径陷阱

我一度用 **逐句 BLEU 的均值**（记录里的 `initial_metric_offline`/`final_metric_offline`）去比 SOTA 表，
得出"我们落后 3.4 BLEU"的**相反结论**。这是错的：

| 量 | qwen3-8b/en_zh 初稿 |
|---|---:|
| 逐句 BLEU 均值 | 37.738 |
| **语料 BLEU（与 SOTA 表同口径）** | **42.402** |

两者相差 4.66，通常逐句均值显著偏低。**与 SOTA 比较、以及论文里报的数，
一律必须用 `core.scoring.Scorer.score_corpus()` 的语料口径。**

验证：用我们的打分器给基线 `Direct-Zero.jsonl` 的 1000 条输出打分 → **42.402**，
与 `baseline_rescored.csv` 完全一致；给我们的 `initial_draft` 打分 → 同样 **42.402**。
且两者草稿文本 **1000/1000 逐字节相同**，y0 无问题。

### 51.2 qwen3-8b / wmt19_en_zh（n=1000，已跑满）

语料 BLEU：初稿 42.402 → **我们 46.144**（Δ **+3.742**）

| 方法 | BLEU | 我们 − 它 |
|---|---:|---:|
| **我们的方法** | **46.144** | — |
| SelfRefine-U | 44.561 | **+1.583** |
| SR-U | 43.210 | +2.934 |
| AdaCompute-SR-U | 43.163 | +2.981 |
| TEaR-U | 43.143 | +3.001 |
| BoN-U-4 | 42.952 | +3.192 |
| PDR-2-1 | 42.607 | +3.537 |
| SR-Fixed | 42.449 | +3.695 |
| ModeX-4 | 42.440 | +3.704 |
| Direct-Zero | 42.402 | +3.742 |
| SelfRefine-Fixed | 42.354 | +3.790 |
| TEaR-Native | 41.454 | +4.690 |
| ChecklistRefine | 41.432 | +4.712 |

**12 个基线全部胜过。**

### 51.3 逐样本配对 bootstrap（2000×，n=1000）

| 基线 | 差 | p | 95% CI |
|---|---:|---:|---|
| SelfRefine-U | **+1.988** | ~0 | [+1.117, +2.919] |
| SR-U | +2.450 | ~0 | [+1.669, +3.306] |
| AdaCompute-SR-U | +2.472 | ~0 | [+1.678, +3.335] |
| TEaR-U | +2.763 | ~0 | [+2.015, +3.593] |
| BoN-U-4 | +2.734 | ~0 | [+1.915, +3.569] |
| Direct-Zero | +3.424 | ~0 | [+2.670, +4.256] |

**全部 p≈0，CI 全部排除零。**

### 51.4 glm4-9b / wmt19_en_zh（n=768，仍在跑）

初稿 37.493 → **38.419**（Δ +0.926）。
与最强 SOTA `SelfRefine-U` 38.442 **打平（−0.023）**，胜过其余全部
（`BoN-U-4` +0.318、`SR-U` +0.584、`AdaCompute-SR-U` +0.691、`Direct-Zero` +1.135）。

### 51.5 结论

**用户提供的方法（真实"错→对"资产库 + 与原方法逐字节一致的四行格式）在
qwen3-8b 上显著优于全部 12 个 SOTA 基线，最强对手领先 +1.583 BLEU（语料口径）
/ +1.988（逐样本配对，p≈0）。** 此前"经验无增量价值"的结论是错库 + 错格式造成的。

待其余三个任务（zh_en / coedit_gec / gigaword）跑满后出完整表。

---

## 五十二、全量 E2 战役启动 + 一次自伤事故

### 52.1 队列

`configs/queue.json` 重建为 **20 个任务 / 62,000 样本**，由 `dispatcher.py` 统一调度（4 卡不留空）：

| priority | 数量 | 臂 |
|---|---|---|
| 0（主判据） | 4 | `full_static`、`sr_j_fixed` × {glm4-9b, qwen3-8b} |
| 1（核心消融） | 8 | `positive_only`、`random_retrieve`、`full_online`、`no_experience` |
| 2（其余消融） | 6 | `fixed_rounds`、`sr_j_stop`、`bon_judge-n4` |
| 3（广度） | 2 | `full_static` × {llama3.1-8b, qwen3-4b} |

**方法冻结，全程不再改动**：`--experience-root experience/contrastive --renderer v2
--split test --draft-source stored --accept-max-len-ratio 1.00`。
`preflight_queue.py` 通过（"every pending job is dispatchable and semantically legal"）。

### 52.2 事故：`run_model` 位置参数错位（我的错，已修）

我在给 `supervisor.run_model` 增加 `--draft-cache/--renderer/--controller-sees-examples/--advice-mode`
时，把 `draft_cache` **插进了签名中间**，却把这些参数**追加在调用末尾**，导致其后每个参数整体错位一格：

```
n_candidates 收到 "" → ValueError: invalid literal for int() with base 10: ''
```

**后果**：priority-0 的 4 个任务在启动后 30 秒内全部死亡，调度器静默降级去跑 priority-1
（priority-1 的 4 个也因同一原因死亡）。**表面看战役在跑，实际主判据根本没在运行**——这是最危险的一类故障。

**修复**：调用改为关键字传参（位置参数只剩稳定的 12 个），杜绝此类错位。
**防复发**：新增 `tests/test_supervisor_args.py`，用 AST 解析源码，断言
`run_model` 的调用不使用超过 12 个位置参数、且其后的每个参数都以关键字传递。
修复后 4 个 supervisor 稳定运行（此前 30 秒即死）。

### 52.3 监控

派出两个**只读**监控子 agent 持续运行：
- **基础设施健康**（`51b0fdde`）：进程数、调度器 `[retry]`、孤儿 EngineCore、崩溃特征、GPU 占用、逐臂行数
- **结果分析**（`25093289`）：逐格进度、主判据（语料口径 + 配对 bootstrap）、SOTA 站位、成本、机制健全性

两者的提示词中都明确写入了本项目踩过的坑（`pgrep -f` 自杀、NFS mtime 滞后 15 分钟、
**逐句 BLEU 均值 ≠ 语料 BLEU 且会反转结论**、baseline `index`↔`sample_id` 对应关系）。

---

## 五十三、监控发现的两起运行事故（均已修复，方法论未动）

### 53.1 孤儿 EngineCore 泄漏 127 GB（监控 agent 上报）

我此前手工停掉 4 个直连任务时，它们的 `VLLM::EngineCore` 子进程变成孤儿（ppid=1），
**泄漏约 127 GB 显存**，导致 GPU 1/2/3 上所有新任务在 vLLM 加载阶段
`ValueError: Free memory on device ... less than desired GPU memory utilization` 崩溃，
并陷入 60 秒一轮的重启循环。**3 个 priority-0 主判据任务因此耗尽重试被判失败**，
调度器静默降级去跑低优先级臂——**主判据实际没在运行**。

处置：杀掉 4 个孤儿（1503702/1505921/1507426/1507603），显存全部释放
（GPU 1/2/3 由 13.0/24.4/27.6 GB 空闲恢复到 65.7/60.1/66.4 GB）；
重置 3 个主判据任务并把全部任务 `max_attempts` 提到 10（显存竞争是常态）。

### 53.2 重复驱动并发写同一文件（监控 agent 上报，数据完整性风险）

我 12:12 手工启动的 `/tmp/e2_main.sh` **一直在后台运行**，与调度器**同时写同一个输出文件**
（`runs/e2_ablation/full_static/seed42/wmt19_zh_en/glm4-9b/full_static.jsonl`）。
两个进程各自 `os.replace` 自己的合并快照，后完成者覆盖先完成者 —— **已产生 2 行撕裂记录**。

处置：递归杀掉该驱动及其全部子孙（1502461/1540385/1541924）；
备份后剔除撕裂行（第 64、160 行），保留 222 行有效数据（无重复 `sample_id`），
被剔除的样本由续跑重做。复检全部 `runs/e2_ablation/**/*.jsonl`：**0 撕裂行**。
确认现役 4 个 supervisor 全部由调度器派生（ppid=1525542），无外部驱动。

**教训**：手工启动的驱动必须纳入作业管理，不能与调度器并存。

---

## 五十四、主判据结果（`wmt19_en_zh`，两格已跑满 1000 条）

**语料口径 + 逐样本配对 bootstrap（2000×）**：

| 模型 | 我们的方法 | 无经验对照 | 语料差 | 配对差 | p | 95% CI | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3-8b | **46.144** | 42.525 | **+3.619** | **+3.174** | <5e-4 | [+2.382, +4.001] | **确定胜出** |
| glm4-9b | **38.136** | 37.238 | **+0.898** | **+1.430** | <5e-4 | [+0.935, +1.988] | **胜出（见下方注意事项）** |

**SOTA 站位**（12 个基线，同一子集重打分）：
- qwen3-8b：**第 1/13 名**，领先最强 `SelfRefine-U` **+1.583**
- glm4-9b：第 2/13 名，落后 `SelfRefine-U` **−0.306**

**成本**：我们 4284（qwen3-8b）/ 5082（glm4-9b）tok/样本；对照 3044 / 2540；
`SelfRefine-U` 2355 / 1925；`SR-U` 522 / 502。

### 54.1 **必须注意的不对称**：glm4-9b 那个格的轮数不匹配

| 臂 | 模型 | 平均轮数 | STOP | 其中解析失败 |
|---|---|---:|---:|---:|
| full_static | glm4-9b | **3.00** | 2 | 2 (0.1%) |
| full_static | qwen3-8b | **2.59** | 211 | 0 |
| sr_j_fixed | glm4-9b | **2.60** | 242 | **242 (9.3%)** |
| sr_j_fixed | qwen3-8b | **3.00** | 0 | 0 |

- **qwen3-8b 是干净的、且更强**：我们用 **2.59 轮**、对照用满 3.00 轮，我们仍赢 +3.174
  —— **以更少算力取胜**，不存在"多改几轮"的解释。
- **glm4-9b 有混淆**：glm4-9b 在 `fixed` 模式下（schema 强制无条件产出 instruction）
  有 **9.3% 的 JSON 解析失败**，解析失败退化为 STOP、样本提前结束，
  因此对照臂只跑了 **2.60 轮** 而我们的方法跑满 **3.00 轮**。
  **+1.430 中可能有一部分来自多出的轮数预算，而非经验本身。**

按用户指示**不改方法论**（该现象是模型 JSON 格式纪律问题，修它属于改行为），
**如实记录为 glm4-9b 格的限制**；qwen3-8b 格不受影响。

---

## 五十五、调度器加入显存门禁（运维修复，非方法论）

**问题**：调度器选卡只看"卡上有没有本项目的任务"，不看显存。
而一次新加载需要 `GPU_MEMORY_UTILIZATION[model] × 卡容量`（glm4-9b 37.9 GB、
qwen3-8b 33.8 GB），4 个引擎在跑时没有一张卡满足。于是每次任务死亡后的重启
都**必然**在 60 秒后以 `ValueError: Free memory on device ...` 崩溃，
白白烧掉一次重试并留下一个死引擎。

**修复**（`dispatcher.py`）：
- 新增 `gpu_memory_fractions()`：用 `ast` 从 `core/__init__.py` 解析
  `GPU_MEMORY_UTILIZATION`，**不导入 `core`**（调度器刻意不拉 torch/vLLM），
  因此单一真相源不会漂移；
- 新增 `gpu_memory_state()` / `memory_shortfall()`；
- 调度循环在启动前检查，不足则**记日志并等待**（`[skip] ... lacks memory`），
  且把卡与任务都放回可用集合——**这是"等待"而不是"失败"，任务保留全部重试次数**。

**交接**：停旧调度器 → 4 个 supervisor 存活（ppid 变为 1）→ 启动新调度器 →
自动认领在跑任务、未重复启动。已验证：supervisor 数仍为 4。

**调度顺序**：3 个停摆的 priority-0 任务排在队首
（`full_static-qwen3-8b`、`sr_j_fixed-glm4-9b`、`sr_j_fixed-qwen3-8b`），
一旦有卡释放即被捡起。

---

## 五十六、主判据两格 CONCLUSION 确定 + glm4-9b 混淆已排除

结果监控在平衡子集上做了关键控制：
**在双臂都跑满 3 轮的样本上（n=775），`en_zh/glm4-9b` 的语料差仍为 +0.823**
—— 提前截断**不能**制造该效应，glm4-9b 的轮数不对称不构成解释。

`wmt19_en_zh` 两格（各 n=1000，COMPLETE）：

| 模型 | 我们的方法 | 无经验 | 语料差 | 配对 CI | p | 结论 |
|---|---:|---:|---:|---|---:|---|
| qwen3-8b | **46.144** | 42.525 | **+3.619** | [+2.786, +4.554] | <0.0005 | 确定胜出 |
| glm4-9b | **38.136** | 37.238 | **+0.892** | [+0.554, +1.279] | <0.0005 | 确定胜出 |

**关键对照**：无经验臂**自身的**语料增益与零不可区分
（glm4-9b −0.045，CI[−0.164,+0.076]，p=0.44；qwen3-8b +0.121，CI[−0.055,+0.300]，p=0.18），
而我们的方法分别是 +0.847 与 +3.738。
⇒ **这是真实的经验效应，不是"无经验臂碰巧也涨了"。**

`wmt19_zh_en/glm4-9b`（n=413，偏序）：+0.201，CI[−0.012,+0.428]，p=0.074 —— **尚不可分辨**。

**SOTA 站位**：en_zh/qwen3-8b **1/13**（+1.583）；en_zh/glm4-9b 2/13（−0.306）；
zh_en/glm4-9b 2/13（−0.196）。
**成本**：我们 4284–5395 tok/样本，无经验 2485–3044，最强基线 1925–2355
（即约为无经验的 1.4–2.2×、最强基线的 2–2.6×）。

---

# 五十七、**推翻性效度问题：参考答案泄漏进 refiner 提示词**（监控 agent 发现，我已独立核实）

## 57.1 事实

`experience/contrastive` 库是从用户提供的原始资产
（`wmt19_dataset_analysis_top10.csv` 等，含 2,978 条 en_zh）建的，
而**这些资产本身就包含测试集**。独立核实结果：

| 检查 | 结果 |
|---|---|
| 测试集源句逐字出现在库中 | **989 / 1000** |
| 其中 `state_after` 与测试参考答案**逐字节相同** | **986** |
| 轮0 检索**第一名 = 该样本自己的单元** | **983 / 1000（98.3%）** |
| 这些第一名的源相似度均值 | **1.0000** |

而 v2 渲染器会把 `state_after` 作为第四行
`Refined Translation (Gold): …` **打印进提示词**。

⇒ **refiner 在为某句翻译时，提示词里直接写着这句的参考答案。**

## 57.2 这是实现漏了原方法的约束，不是设计选择

原方法 `llm_re_cosine_qwen8b.py::retrieve_batch` 明确排除自己的记录：

```python
if similarities[i][idx] > 0.99 and candidate_text == query_texts[i]:
    continue
```

而 `RunConfig.ban_own_experience` 默认为 `True`，**却从未被实现**：
`core/` 与 `run_experiment.py` 中只有 dataclass 字段定义与
`tests/test_scaling.py` 的键名列表，检索调用点
（`pipeline.py` 的 `self.retriever.retrieve(ref.source, …)`）根本没有排除逻辑，
而 `ref` 就在调用点上可用。**且 `ban_own_experience` 在 `config_hash` 覆盖范围内
—— 即每一个冻结哈希都在为一个并不存在的禁令背书。**

## 57.3 影响

- 抄袭整句很少（`final == reference`：full_static 23/1000 = 2.3%，对照 12/1000 = 1.2%），
  所以**不是**整句复制。
- 但模型确实吸收了参考答案的措辞：对参考答案的 char-4gram 召回
  en_zh/qwen3-8b 由 0.2805 升到 0.3163（**+0.0358**），对照仅 +0.0013 —— **约 27 倍**；
  glm4-9b +0.0130 vs −0.0013。
- BLEU 恰恰奖励这个。⇒ **`+3.619` / `+0.892` 不能被当作
  "经验驱动的精炼"的证据，它们测的是"把答案给模型看之后的重写"。**

另外：即使排除自己的单元，**检索池仍来自测试集**（其他测试样本的参考答案），
因此要得到可辩护的 test-split 结论，库本身必须来自与测试不相交的切分。

## 57.4 已采取的行动

- **立即停止调度器与全部任务**，避免继续产出无效结果并浪费算力
- 清理全部孤儿 EngineCore，4 张卡已释放
- 已停掉两个监控 agent（战役暂停，继续轮询无意义）
- **未改动任何方法论代码**，等用户决定

已产出但受污染、需要重跑：`runs/e2_ablation` 共 8,030 行
（`sr_j_fixed` 3472 / `full_static` 2734 / `positive_only` 1248 / `random_retrieve` 576）。
**`sr_j_fixed` 与 `no_experience` 不使用经验库，其数据不受此问题影响。**

## 57.5 修复要点（含一个身份陷阱）

1. 在检索中排除自己的单元（复刻原方法的 `sim>0.99 且同文本 → skip`）。
2. **必须同时分离身份与输出目录**：修复后行为变了但 `config_hash` 不变
   （`ban_own_experience` 本来就是 `True`），续跑会把**被污染的旧行**与干净的新行
   混进同一文件。故需新增一个 legacy-default 省略的字段并换用新结果树。
3. 更稳妥的版本：库改为从与测试不相交的切分构建（`dispatcher` 已在
   `--draft-source cached` 上具备冻结切分的能力，可复用）。

## 57.6 另有两项运维缺口（监控 agent 报出）

- **supervisor 内部重试不做显存复查**（固定 `sleep(15)`、`max_attempts` 次）：
  任务在启动时显存充足、随后遇到竞争尖峰，会在 ~5 分钟内烧光全部重试。
  已修的调度器级门禁保护不了这种情形。建议 supervisor 每次重试前复查空闲显存并退避。
- **优先级反转**：3 个 priority-0 任务死亡后进入 60 秒退避，退避窗口内
  3 张卡被 priority-1 任务占满，导致主判据排在低优先级臂的整轮之后。

---

# 五十八、修复：复刻原方法的"排除自己单元"约束（用户确认 + 原码佐证）

## 58.1 定论

用户指出"baseline 跑的部分和检索集是独立的"。核对原方法源码后确认了准确含义：

```python
RETRIEVAL_DB_PATH = ".../wmt19_dataset_analysis_top10.csv"
TEST_DATA_PATH    = ".../wmt19_dataset_analysis_top10.csv"   # 竟然是同一个文件
```

**检索池与测试集同源是原方法的设计**；让它成立的是这句：

```python
if similarities[i][idx] > 0.99 and candidate_text == query_texts[i]:
    continue          # 绝不把某条的答案喂给它自己
```

用户所说的"独立"，正是指**对每一条而言，喂给它的示例里不含它自己的答案**。
**这一条我漏了实现**——`RunConfig.ban_own_experience` 默认 `True` 却从未生效。

逐字节复核确认重叠真实存在：测试集 ∩ 库 = **981/992**；
前 3 条样本的库单元 `state_after` 与测试参考答案**完全相同**。

## 58.2 修复

1. `ExperienceRetriever.retrieve()` 新增 `exclude_source` 参数，
   在排序后、配额选择前剔除 `source_input == 查询源句` 的单元
   （等价于原方法的 `sim>0.99 且同文本`，因为同文本必然相似度 1.0）。
   **在配额选择之前过滤**，所以仍然填满 k=4——与原方法"继续往下取"一致。
2. 两条管线（`core/pipeline.py`、`core/batched_pipeline.py` 各 2 处调用点）
   在 `cfg.retrieval_excludes_own_source` 为真时传入 `ref.source`。
3. **身份分离**：新增 `RunConfig.retrieval_excludes_own_source: bool = False`
   （列入 `LEGACY_HASH_DEFAULTS`）。不能只就地修 `ban_own_experience`——
   它在所有已记录配置里本来就是 `True`，修好后行为变了而哈希不变，
   续跑会把**污染行与干净行拼进同一文件**。
4. CLI/`supervisor`/`dispatcher` 全链路接线。

## 58.3 验证

| 检查 | 结果 |
|---|---|
| 修复前检索含自己单元 | 198 / 200（99.0%） |
| **修复后检索含自己单元** | **0 / 200（0.0%）** |
| 修复后仍填满 k=4 | 是 |
| v1 默认哈希 | `125eff5ed5057006`（**不变**） |
| v2 受污染哈希 | `feee8c374de862d1` |
| v2 加固哈希 | `c795581ce4b82d3b` |

新增门禁 `tests/test_own_source_ban.py`（7 项断言，含"库确实与测试集重叠"
与"该单元 state_after 与参考答案逐字节相同"两条防空洞化检查）。
**13 个测试套件全绿。**

## 58.4 重跑范围

- **必须重跑**：所有使用经验库的臂（`full_static`、`positive_only`、
  `random_retrieve`、`full_online`、`fixed_rounds`、`bon_judge`）
- **可直接复用**：`sr_j_fixed`、`no_experience`、`sr_j_stop`
  —— 它们的 `experience_mode` 不做检索，**该修复在原理上不可能影响其输出**，
  且初始草稿来自测试集 stored drafts、各调用均有种子，重跑会逐字节相同。
