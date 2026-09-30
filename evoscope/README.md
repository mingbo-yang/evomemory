# EvoScope

EvoScope 根据同一状态下 expose/mask 的成对执行证据，学习策略记忆的适用条件。
固定 `key` 和 `do`，只修改 `when`；候选经过独立 Gate 验证后才写入记忆库。
当前实验协议见 [PROTOCOL_V3.md](docs/PROTOCOL_V3.md)，方法背景见 [METHOD.md](docs/METHOD.md)。

## 代码结构

| 文件 / 目录 | 职责 |
| --- | --- |
| `core.py` | 策略与记忆库、固定 key+do 检索、任务划分、产物与成本记录 |
| `engine.py` | Probe、跨轮分层采样、多状态证据、编辑、Gate 与继承 |
| `gate_plan.py` | 候选生成前固定 local/global 验证题组 |
| `runner.py` | 任务执行、目标保留、checkpoint 回放、检索与曝光记录 |
| `environments.py` | ALFWorld、WebShop 与离线测试环境适配 |
| `models.py`、`editor_budget.py` | Actor/Editor、本地与网关客户端、证据压缩及输入预算 |
| `cli.py`、`__main__.py` | 统一命令入口：初始化、探测、演化、冻结评测 |
| `provider.py` | EvolveLab 集成；任务边界刷新记忆库 |
| `protocol_validation.py` | 卡 1 本地 Qwen 的三项协议修复验证及自动清理 |
| `local_watchdog.py`、`local_expanded.py` | 多模型/数据集实验监控及独立 worker |
| `package_source.py`、`run_tests.py` | 源码打包与离线回归入口 |
| `tests/`、`examples/` | 回归测试、格式示例与历史小规模环境样本 |
| `scripts/` | 本机 vLLM 部署脚本 |
| `docs/` | 当前方法、协议、诊断与历史报告 |
| `results/`、`.local/` | 实验产物、本机配置与运行依赖，不随源码发布 |

## 离线检查

在 `/mnt/huawei/ymb/evomemory` 下执行：

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' memevolve-venv/bin/python -m evoscope.run_tests
memevolve-venv/bin/python -m evoscope doctor
memevolve-venv/bin/python -m evoscope demo --output /tmp/evoscope-demo-new
```

`demo` 使用人工夹具验证流程，不调用模型，不代表方法效果。所有 output 必须是新目录；
目前不支持直接续跑中断目录。`requirements-real.txt` 是真实环境依赖，
`requirements-test.txt` 是测试依赖；本机现有运行环境路径保持不变。

## 本地协议验证

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' memevolve-venv/bin/python -m evoscope.protocol_validation   --source evoscope/results/reliability-v2-inputs-20260925   --output evoscope/results/protocol-validation-new
```

该入口在物理卡 1 启动已有 Qwen 权重，两个数据集各取固定的 12 个学习任务；
使用各自预生成的 M₀，每条记忆最多探测 3 个状态。它验证 Gate 预选与曝光记录、
跨轮采样和多状态聚合。自然演化与只读诊断分开记录，不将诊断计为有效更新。
两个数据集在独立 worker 中运行；结果完整保存后清理 worker 和本次服务器，
没有总时间上限。完整说明见 [PROTOCOL_V3_VALIDATION.md](docs/PROTOCOL_V3_VALIDATION.md)。

本机模型部署映射：

| 脚本 | 物理 GPU | 本地端口 | 服务名 |
| --- | ---: | ---: | --- |
| `scripts/serve_qwen9b_local.sh` | 0 | 8125 | `qwen3.5-9b-local` |
| `scripts/serve_qwen_local.sh` | 1 | 8124 | `qwen3.8-27b-local` |
| `scripts/serve_glm_local.sh` | 2 | 8123 | `glm-4.7-flash-local` |

Qwen3.8 是现有目录/服务命名，本地 config 声明 `Qwen3_5ForConditionalGeneration`。
启动脚本依赖本机 `.local/` 运行环境，不是通用安装脚本。完整对照实验使用
`python -m evoscope.local_watchdog --models qwen --hours 0 --source <固定输入目录> --output <新目录>`。
监控器拥有并清理自己启动的进程，避免重复占用端口或终止无关任务。

## 统一实验命令

已有本地模型服务时，可直接使用 CLI：

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' memevolve-venv/bin/python -m evoscope run   --local --base-url http://127.0.0.1:8124/v1 --model qwen3.8-27b-local   --environment alfworld --manifest /path/to/manifest.json   --policies /path/to/shared-m0.json --output /path/to/new-run   --send-seed --seed 42 --max-steps 30 --max-tokens 8192
```

- `bootstrap`：仅从 bootstrap 成功轨迹生成自然 M₀；所有比较方法共用同一初始库。
- `probe-only`：固定 M₀，只收集 expose/mask 证据，不编辑或更新。
- `run`：学习任务内持续更新，候选通过 Gate 后才继承。
- `evaluate`：冻结指定记忆库，只运行 test；空数组 `[]` 可用于无记忆对照。
- `check-manifest <文件>`：检查任务 ID、组和游戏文件的划分约束。

`run`、`probe-only`、`evaluate` 必须提供 `--policies`。可用
`--expected-bank-hash` 在模型调用前校验库摘要。WebShop 另加
`--environment webshop --webshop-root /mnt/huawei/ymb/evomemory/downloads/WebShop-10k`，
manifest 的 payload 使用 `goal_index`；ALFWorld 使用 `gamefile`。
不同数据集、不同模型分别维护演化库。示例策略是手写格式示例，不能作为自然 M₀ 的证据。

真实本地测试显式使用 `--local` 和数字回环地址，不读取网关凭据、不回退远程服务。
统一 CLI 仍保留用户指定网关的远程模式，仅在另行要求远程实验时使用；
凭据由环境变量或本地配置保存，不写入源码和结果。直接 CLI 使用已有服务器，
服务器的启动与退出由调用方负责；需要自动管理时使用上述监控入口。

## 当前协议与审计

`run` 默认至少积累 3 个独立状态且含 1 个稳定非零作用状态才允许编辑，
保留 neutral；证据按策略版本、完整库摘要和执行配置隔离。
全局 Probe 预算为 24、每策略 6；`local_expanded` 为 12/6。
`probe-only` 为 24/3，小规模 `protocol_validation` 为 6/3。
每个 Probe 包含 3 次成对执行，失败尝试也占预算，不等于 3 次模型调用。

Gate 默认 2 个 local 加 2 个 global 独立任务组，每任务 old/new 配对 3 次，
每策略最多 2 个预定候选槽位。题组在学习/候选生成前固定；
没有足够相关题时记录不足，整个 local Gate 都未曝光目标记忆时标记
`coverage_insufficient`，不把 0–0 当作候选无效的证据。验收条件及边界见当前协议。

同一对照组固定任务顺序、M₀、提示、模型与输出预算；默认 `max_tokens=8192`。
ALFWorld 通过重置与动作回放恢复 checkpoint，并核对公开观测。
WebShop 使用原始搜索与奖励，目标商品组不可跨 split。恢复或模型错误单独记录，
不算 neutral；`temperature=0` 和 seed 也不保证严格确定。

| 产物 | 检查内容 |
| --- | --- |
| `initial_bank.json`、`bank.json` | 初始及最终继承的记忆库 |
| `learn/episodes.jsonl` | 学习轨迹、分数、错误及 checkpoint |
| `probe/attempts.jsonl`、`probe/evidence.jsonl` | 探测预算、成对结果与上下文 |
| `probe/sampling.jsonl`、`probe/coverage.jsonl` | 累计采样配额与 clean checkpoint 覆盖 |
| `edit/eligibility.jsonl`、`edit/inputs.jsonl` | 编辑资格与实际多状态输入 |
| `gate/plan.json`、`gate/decisions.jsonl` | 预定题组、配对分数与目标曝光 |
| `costs.jsonl`、`summary.json` | 调用、环境步数、错误与阶段汇总 |
| 监控根目录的 `status.json`、`cleanup.json` | 运行进度和本次 GPU 清理结果 |

阶段执行结果不等于 held-out 对照效果。历史协议结果应分别报告，
不得混合不同版本的分数或把只读诊断记为方法提升。

## 源码打包与历史归档

```bash
memevolve-venv/bin/python -m evoscope.package_source --output /tmp/evoscope-source-new.zip
```

白名单包含代码、测试、文档和格式示例，排除 `.local/`、凭据、实验结果、
模型、环境、下载数据、归档及符号链接。已有压缩包不会覆盖。

早期付费冒烟测试、旧 pilot、一次性扩容/回放、排队与接管脚本已移至
`/mnt/huawei/ymb/backups/evoscope-code-cleanup-20260926/`。
`CLEANUP_MANIFEST.json` 记录原路径与文件摘要，旧 README 也保存在归档内。
原始实验结果及其源码快照保留原位置；历史脚本需要已移出的模块时，可按清单恢复。
当前方法和在运行的实验代码没有因本次整理改变。
