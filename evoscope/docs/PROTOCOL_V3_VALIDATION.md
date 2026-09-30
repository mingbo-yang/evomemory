# 三项协议修复的本地验证

入口：`python -m evoscope.protocol_validation --source <固定输入目录> --output <新输出目录>`。
该入口限定物理 GPU 1、已有本地 `Qwen3.8-27B` 目录及数字回环地址，
不会读取远程 API 配置。目录名沿用历史命名；本地 config 声明的架构是
`Qwen3_5ForConditionalGeneration`，不能仅凭目录名断言官方模型版本。

两个数据集各取输入清单中前 12 个 learn 任务，保留全部 gate 任务；不用 test。
每个数据集复制同一来源的自然 M₀，银行分别更新。批大小为 2，每轮 1 个
checkpoint，总 probe 预算为 6、每条策略最多 3 个；probe/Gate 均为 3 次配对。
Gate 固定为 2 个 local 和 2 个 global 任务，每条策略预留最多 2 个题组。
这些是小规模验证参数，不修改正式实验默认值。

检查三个问题：

1. `gate/plan.json` 在学习及编辑前固定任务，记录相关任务不足；实际 Gate
   记录 old/new 两侧目标记忆的检索与曝光，曝光不足不得作为有效拒绝证据。
2. `probe/sampling.jsonl` 记录每个 policy/version 的跨轮累计配额；当两层均有
   可用 checkpoint，优先补累计较少的层。观察不到两层同时可选时不声称已
   验证该分支。轨迹成败与 expose/mask effect 分开记录。
3. `edit/eligibility.jsonl` 按独立状态、任务组和上下文累计证据，保留 neutral；
   至少 3 个状态且有稳定非零作用证据才允许自然编辑。

如果自然演化未调用编辑器或 Gate，最多各做一次单独诊断：使用真实采集的
至少 3 个兼容状态调用编辑器，或者在尚未消耗的预定题组执行 identity candidate
的 dry-run Gate。诊断不部署候选、不改变自然银行、不补造正负标签，也不
计为成功更新；没有合格状态或题组就明确跳过。所有诊断保存到独立目录。

`*-summary.json` 是结果已完整保存的完成标记。监控程序检测到标记后关闭
对应 worker，即使 native 环境卡在解释器退出阶段也不会一直占用模型。
两组测试完成或发生异常后，关闭自己创建的服务器进程组并检查其 GPU PID
已消失，记录 `cleanup.json`。不停止无关任务；不设置总运行时间上限。

该测试能验证实现及流程，但没有 held-out 对照评估，不能证明任务性能提升。
