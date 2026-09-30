# 临时干预诊断：scope-diagnostic-v1

这是默认关闭的试验。普通 CLI、Engine 的 Probe 和 `protocol_validation` 默认仍使用
原来的持续 expose/mask；奖励、编辑规则、Gate 门槛和预算不变。此前稳定源码包
`/mnt/huawei/ymb/exports/evoscope-source-20260926.zip` 不会覆盖。

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' memevolve-venv/bin/python -m evoscope.protocol_validation \
  --trial probe-scope \
  --source evoscope/results/protocol-v3-qwen38-gpu1-20260926 \
  --output evoscope/results/scope-trial-qwen38-gpu1-20260926
```

仅使用卡 1 本地 Qwen 权重。复用原监控器、独立数据集 worker 和完成后显存清理，
无总运行时间上限。源测试须已结束；每个数据集使用全部有效 learn Probe 状态
（小规模入口限 1–6 个，超过则拒绝，不按效果挑选）。本轮为两组各 6 个 checkpoint，
不生成新 M₀，也不编辑或运行 Gate。其结果不得混入正式 evidence buffer。

每个 checkpoint 重放五个分支，各 3 次，合计 180 条轨迹：

| 分支 | checkpoint 当步 | 后续步骤 |
| --- | --- | --- |
| normal | 正常检索，保留原 when | 正常检索，保留原 when |
| persistent-expose | 目标裸 do，保留位置 | 持续强制注入目标裸 do，沿用旧规则 |
| persistent-mask | 删除目标，不补位 | 持续屏蔽目标，沿用旧规则 |
| single-step-expose | 目标裸 do，保留位置 | 恢复正常检索和原 when |
| single-step-mask | 删除目标，不补位 | 恢复正常检索和原 when |

同一 repeat 的五个分支使用同一 seed，按预定轮转顺序执行；全部重新生成，
不复用旧 expose/mask 的得分当作本轮配对。恢复检查仍要求初始观测、回放、任务、
库摘要、模型配置、检索次序一致；不放宽 seen 限制。单步相对于 checkpoint，
不是整个任务的 step 0。非目标记忆在干预当步一致，后续自然轨迹允许分叉。

normal 对比持续干预时，差异可能还包含后续检索和上下文容量变化，不能单独归因
于 when。日志记录实际干预步数、目标是否掉出自然检索以及强制曝光次数。
“normal 与 mask 相同且 expose 更差”是诊断现象，不自动触发或禁止任何编辑。

neutral 保持奖励差为零的原定义。另存动作序列是否相同、首次分歧位置、步数差、
分支成本和重复中的动作序列种数；这些指标不进入标签或 Gate。统计里的行为计数
单位是配对 repeat，状态计数另列。错误不当成 neutral，不自动重试或换 checkpoint。
这不是 held-out 对照实验，不能用少量重放宣称成功率提升或总体效应分布。

## 回退

不传 `--trial probe-scope` 就使用原方案，无需先恢复代码。
若要完全撤销试验代码，原文件和摘要位于：

`/mnt/huawei/ymb/backups/evoscope-before-scope-trial-20260926/`

```bash
python /mnt/huawei/ymb/backups/evoscope-before-scope-trial-20260926/restore.py --check
python /mnt/huawei/ymb/backups/evoscope-before-scope-trial-20260926/restore.py --apply
```

恢复工具先核对所有文件摘要；若试验仍运行或文件后来另有修改，会拒绝覆盖。
先结束本次试验监控并等待 `cleanup.json` 确认显存释放，再完整恢复。
恢复只涉及本次改动的代码和文档，保留所有实验结果及备份。
