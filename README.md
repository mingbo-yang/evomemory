# EvoScope

本项目只维护 **EvoScope：通过成对执行证据修正策略记忆的适用条件**。

- [代码与运行说明](evoscope/README.md)
- [当前实验协议 v3](evoscope/docs/PROTOCOL_V3.md)
- [方法与实验文档索引](evoscope/docs/README.md)
- [实验结果](evoscope/results/)

```text
evomemory/
├── evoscope/          # 方法代码、测试、脚本、文档、实验结果
├── downloads/         # ALFWorld / WebShop 数据和模拟器
├── memevolve-venv/     # EvoScope 正在使用的 Python 环境（保留已有路径）
├── AGENTS.md          # 本地实验与资源使用约定
└── README.md
```

`memevolve-venv` 是历史命名的运行环境，不是旧方法代码。数据和环境保留原路径，确保现有任务、解释器入口和实验清单继续有效。

```bash
cd /mnt/huawei/ymb/evomemory
TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' memevolve-venv/bin/python -m evoscope.run_tests
```

MemEvolve、早期复现工具、旧研究资料和旧源码压缩包已移至
`/mnt/huawei/ymb/backups/evomemory-legacy-20260926/`，不属于当前项目。
归档内的 `ORGANIZATION_MANIFEST.json` 记录原路径与新路径，可用于恢复。
