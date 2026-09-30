# DeepSeek V4.1 / ALFWorld 小规模真实集成测试

时间：2026-09-24T10:43:45.676674+00:00。模型请求和网关返回的模型名称均为 `deepseek-v4.1-flash`，入口为用户指定的 `http://172.25.76.237:3000/v1`。本文记录接口和程序验证，不作为方法效果实验。

## 结论

初始化、真实任务执行、成对探测、编辑接口、独立门控、冻结测试和策略保存核对已逐阶段完成。不是一次无中断运行：发现兼容问题后保留失败记录，复用自然生成的 M0 和已完成证据继续测试，没有为获得正效果换任务或改策略。

最终探测为 neutral，编辑器返回 noop。门控采用同条件的 identity control 且 `dry_run=True`，两次执行均成功，平均差值为 0，未继承更新。真实 API 的“自然候选被接受并提交”分支未在本次触发；其程序行为由离线测试覆盖。

## 数据及环境

- ALFWorld 0.4.2、TextWorld 1.7.0、OpenAI 客户端 3.13.0，文本环境。
- 从官方包按名称顺序预先选 5 个不同任务族：2 个 bootstrap，1 个 learn，1 个 gate，1 个 test。
- 选择未使用 walkthrough 或执行成绩。来源、包摘要及成员路径见 `examples/alfworld-smoke/source.json`。
- 模型只收到公开目标、观测和 admissible actions，未收到文件中的隐藏状态或标准动作答案。
- 5 个游戏均已单独检查重置及一个动作的回放一致性。本次成对探测位于初始 checkpoint，因此实际 probe 的前缀回放步数为 0。

## 各阶段结果

| 阶段 | 实际结果 |
| --- | --- |
| 连接测试 | 真实 JSON 请求成功 |
| Bootstrap | 两次任务尝试中，一次遭遇输出截断；另一次 4 步成功，抽取 1 条自然策略 |
| 修复后的 Learn | 4 步成功 |
| Probe | 2 对 expose/mask，共 4 条真实分支，步数分别为 4、14、4、4，均成功，回报差均为 0 |
| Editor | 首次及诊断请求遭遇 HTTP 400；明确 JSON 输出要求后合法返回 noop |
| Gate | 独立游戏的两次正常部署执行均 5 步成功；只读 identity control，差值 0、无提交 |
| Frozen test | 独立游戏 5 步成功 |
| 持久化 | 最终 bank 与初始 M0 摘要相同 |
| 回归测试 | 47 项全部通过 |

旧轮 Learn 在 12 步预算内未完成；修复后运行的上限为 16 步，但实际只用 4 步。模型服务不支持本测试中强制确定的输出，不能把跨运行差异解释为方法改进。

## 发现并处理的问题

1. 共享文件系统上的临时动态库清理失败：真实环境命令使用 `TMPDIR=/tmp`。
2. 2048 个输出 token 被用满，JSON 不完整：smoke 使用 8192 上限；客户端识别 `finish_reason=length` 并报 `ModelOutputTruncated`，保留成本，不静默重试。
3. 编辑提示未明确 JSON 输出，网关拒绝请求：提示改为明确返回 JSON 对象，相同证据随后成功返回 noop。

## 调用与成本

包括连通性、失败尝试、错误诊断及最终继续执行的全部成本：

- 实际 API 调用：68 次，其中 4 次失败。
- 已知输入 token：61,959；已知输出 token：41,532。
- 2 次 HTTP 400 未返回 token 用量，明确计为 unknown。
- 真实轨迹日志中的环境步数：61；另有不调用模型的环境恢复审计，不混入模型效果统计。
- 查询后剩余额度：46,119,069；当前累计已用额度：3,880,931。网关未声明额度单位，不换算为 tokens 或金额。

限额设置分别为单次脚本最多 96 次调用，以及最终续测最多 44 次；所有阶段实际合计 68 次，已经停止调用。未启动规模实验。

## 审计路径

- `results/v41-connectivity/`：连通性与测试前后额度。
- `results/v41-alfworld-smoke/`：初始真实尝试、自然 M0、截断失败。
- `results/v41-alfworld-smoke-verified/`：修复后成功的 learn/probe，首次编辑 HTTP 400。
- `results/v41-editor-diagnosis/`：编辑错误诊断。
- `results/v41-alfworld-smoke-complete/`：最终合法编辑、只读门控及冻结测试。
- 最终目录内 `source_stages.json` 链接前序证据，`aggregate_usage.json` 汇总全程成本，`reproduce_continuation.py` 保存此次受限续测脚本。

本报告是历史记录。旧 `live_smoke.py` 已移至 `/mnt/huawei/ymb/backups/evoscope-code-cleanup-20260926/`；历史结果中的 `reproduce_continuation.py` 如需重放，应先按归档清单恢复其依赖。当前本地测试使用 [protocol_validation](PROTOCOL_V3_VALIDATION.md)，不要将这份旧清单当作未见过的正式测试集。
