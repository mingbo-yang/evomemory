# 交接文档（2026-09-12 08:30 EDT）

本文件供后续会话无缝接手。当前**一切自动运行中，无需人工干预**。

---

## 一、当前状态一句话

4 张卡满载运行带闸门消融与基线；调度器零空转、零错误；队列已按科学价值排序并通过飞行前校验；
**剩余约 7 小时 GPU 墙钟时间**。

---

## 二、如何继续（操作者只需知道这些）

```bash
cd /mnt/huawei/ymb/aaai2027/exp

# 1. 看是否在跑（应见 4 个 supervisor + 1 个 dispatcher）
ps -eo args | grep -E "supervisor.py --gpu|dispatcher.py --interval" | grep -v grep

# 2. 看进度
python preflight_queue.py                    # 队列合法性（应 OK）
grep -a "hb\]" logs/dispatcher.log | tail -1 # 心跳 + 队列计数

# 3. 若调度器意外死亡（极少见），重启它：
setsid nohup /home/ymb/miniconda3/envs/qwen35/bin/python dispatcher.py --interval 30 \
    > logs/dispatcher_stdout.log 2>&1 < /dev/null &

# 4. 汇总结果（随时可跑，只读）
/home/ymb/miniconda3/envs/qwen35/bin/python consolidate_results.py
```

**关键机制**：调度器从 `configs/queue.json` 取任务，一旦有卡空闲即在 **8–64 秒内**接力；
`run_experiment.py` 支持断点续跑（重启不丢已完成样本，只有 `config_hash` 完全一致的记录才被复用）。

---

## 三、环境与硬性约束

| 项 | 值 |
|---|---|
| Python | `/home/ymb/miniconda3/envs/qwen35/bin/python`（vllm 0.17.0 / transformers 4.57.6 / torch 2.10.0+cu128） |
| 绘图 | `/home/ymb/miniconda3/envs/llama4/bin/python`（有 matplotlib、无 sacrebleu；先跑 `make_plots.py --refresh-corpus --no-figures`） |
| baseline | `../baseline/` **只读，永不修改** |
| 编码模型 | **禁止使用**（检索为自建 BM25） |
| 冻结配置 | `configs/frozen.json` — τ=1.02、α=0.5、GPU 授权历史、qwen3-32b 偏差、核心论断状态 |
| 方案 | `PLAN.md`（v5 正文逐字重建 + 执行期变更记录） |

**NFS 陷阱（务必记住）**：`/mnt/huawei` 的 **mtime 滞后墙钟约 15 分钟**。
**绝不能用 mtime / "日志不再增长" 判断停滞** —— 已因此产生两次假警报。
判断活性请用 `wc -l`（jsonl 行数）、文件字节数、内容不变量。

**孤儿引擎**：`run_experiment.py` 异常退出会留下 `ppid=1` 的 `VLLM::EngineCore`，
占约 37 GB 并使该卡此后所有加载失败。清理方法（**按精确 PID**，勿用 `pgrep -f`，会匹配到自身）：
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do
  PAR=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
  C=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$C" in *VLLM::EngineCore*) [ "$PAR" = "1" ] && kill -TERM $p;; esac
done
```

---

## 四、已完成的方案内容（Phase 0–4）

| Phase | 状态 | 关键产物 |
|---|---|---|
| 0 环境/门禁/动机表 + **oracle-stop 上界** | ✅ | `reports/phase0_env.json`、`reports/repro_gate.json`(77/80)、`scores/motivation_steps.csv`、`scores/oracle_stop_bound.csv` |
| 1 manifest/切分/不变量 | ✅ | `data/manifests/*`、`tests/test_invariants.py`(29 项) |
| 2 smoke / 3 initial experience | ✅ | `runs/smoke/*`、`experience/initial/*`(16 库) |
| **4 主实验** | ✅ | **`runs/main/…`(无闸门 16/16)+ `runs/gated_main/…`(带闸门 16/16)** |
| 9 重评分/bootstrap/绘图 | ✅ | `scores/consolidated_results.csv`(89 行/81 格)、`scores/gated_bootstrap.csv`、`plots/fig1–4_{png,pdf,csv}` |
| 10 人工盲评 | ✅ | `human_eval/human_eval_pairs.csv`(200 对)+ `answer_key.DO_NOT_OPEN.json` |

---

## 五、未完成的内容与优先级

| 优先 | 项 | 队列 id 前缀 | 说明 |
|---|---|---|---|
| P2 | 128 条辅助切分上的 gold 库高功效验证 | `AUX-*` | 决定是否采纳"真实收益标签"；`--split` 修复已验证 |
| P3 | 核心消融：outcome_hidden / random_retrieve / positive_only | `G2-*` / `G3-*` | |
| P4 | Phase 7 反事实、Phase 8 累积、full_online、Phase 6 BoN-J | `P7-*` / `P8-*` / `G3-full_online-*` | Phase 7/8 **已在队列中**(见下方命令原文);Phase 6 需按 `--only k\|n` 分片入队 |
| P5 | 2×2 的 seed 43/44 重复（16 个任务） | `T5-*` | **最低价值，可放弃以省 3.5 小时** |
| P6 | §1 dev 证据重跑 | `T6-dev-*` | 32 条 |

**Phase 6/7/8 的命令原文**（Phase 7/8 已在队列中，此处保留供核对或重新入队；调度器支持任意命令，见 `cmd` + `expect_files`）：
```python
# Phase 7 反事实
{"id":"P7-counterfactual","gpu":None,"priority":4,"status":"pending","attempts_left":2,
 "cmd":[PY, f"{ROOT}/run_stopping_diagnostics.py","--counterfactual","--max-stop-states","100",
        "--runs",f"{ROOT}/runs/gated_main","--gpu","{gpu}","--out-dir",f"{ROOT}/scores"],
 "expect_files":["scores/stopping_diagnostics_counterfactual.csv"]}

# Phase 8 累积
{"id":"P8-accumulation","gpu":None,"priority":4,"status":"pending","attempts_left":2,
 "cmd":[PY, f"{ROOT}/run_accumulation.py","--gpu","{gpu}","--models","glm4-9b,qwen3-8b",
        "--tasks","wmt19_en_zh,coedit_gec","--orders","3",
        "--conditions","dynamic,frozen,no_negative","--chunk-size","32",
        "--eval-limit","200","--batch-size","32"],
 "expect_files":["scores/accumulation_curve.csv"]}

# Phase 6 scaling（先跑只读计划）
#   python run_scaling.py --dry-run
#   python run_scaling.py --reuse-only
#   然后按 --only k|n、--models、--tasks 分片入队

# !! Phase 6 入队前必须先补一处透传 !!
# run_experiment.py 已有 --n-candidates，但 supervisor.py 与 dispatcher.py 尚未转发它，
# 因此 BoN-J 目前无法经调度器运行。参照已实现的 --accept-max-len-ratio / --experience-root
# 补三处即可（supervisor 的 run_model() 参数 + cmd 构造 + argparse；dispatcher 的 build_cmd）。
# 注意 n_candidates 必须保持为「管线参数而非 RunConfig 字段」，否则会改掉全部 config_hash。
```

---

## 六、已定型的科学结论（勿重复论证，直接引用）

见 `reports/FINAL_RESULTS.md` 与 `reports/judge_bottleneck_diagnosis.md`（十部分）。要点：

1. **闸门修复系统性有效**：主表 16 格严格配对 **−13.716 → −0.738**（净 **+12.978**），13/16 改善
2. **判决性：经验无质量增量价值**：`fixed_rounds` vs `sr_j_fixed` 差 **+0.006 / −0.138**（p=0.920 / 0.734），成本 **1.9×**
3. **与最强基线（SR-J-Stop）统计持平**：6 格中 5 格不可区分，0 格显著胜出
4. **判定器无便宜解法**：换模型 14.4%、核查式提问 3.8% vs 原始 21.5%
5. **经验标签与真实收益仅 68.6% 一致**（31% 错误 + 39% `uncertain`）
6. **oracle 上界 ≈ +0.5/格**；最优策略是"尽量少改"

---

## 七、测试与校验（改动代码后必跑）

```bash
for t in test_invariants test_accept_gate test_judge_identical_pair \
         test_resume_equivalence test_scaling test_accumulation_gaps \
         test_accumulation_loop test_human_eval_export; do
  CUDA_VISIBLE_DEVICES="" python tests/$t.py | tail -1
done
python preflight_queue.py      # 队列飞行前校验
```

**特别提醒**：`test_accept_gate.py` 会断言**全部生产 run 的 config_hash 未变** ——
`--accept-max-len-ratio`、`--experience-root`、`--n-candidates` 都是**管线参数而非 `RunConfig` 字段**，
新增参数时必须保持这一点，否则在跑任务重启会丢弃已完成前缀。

---

## 八、本会话修复的真实缺陷（供追溯）

1. 输出文件 `open("w")` → **重启丢弃全部已完成样本** → 实现断点续跑（等价性已证明）
2. 批处理分支**从不落盘在线经验库** → `full_online` 会作废
3. 判定器把逐字节相同的两稿判成 `uncertain` → 且**修复最初只落在非批处理路径上**（已修正为单一来源）
4. 调度器**声称按优先级调度但从不读 `priority`** → 已补
5. **崩溃留下孤儿 vLLM 引擎**（4 次）→ 参数校验提前到模型加载之前
6. supervisor **日志名不含 split** → dev 与生产互相覆盖
7. 调度器**用测试集规模判断 dev 完成度** → 成功的 dev 任务被反复重试
8. `--split` 不接受 `accumulation` → AUX 任务秒退（已修 + 已端到端验证）
