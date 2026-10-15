# 固定条件五模型循环赛

在 Mac 的仓库根目录、已激活 Python 环境中，连接已有空闲引擎运行：

```bash
bash round_robin.sh --existing-engine 127.0.0.1:28000
```

默认只有一个引擎、16 槽、10 个 PyTorch 推理线程。顺序运行全部十个模型对，每对一次加载双方模型，跑完全部分配的组号，再换下一对。外部引擎贯穿全部模型对；程序不启动、不停止或重装外部模拟器。

默认模型：

| 名字 | 文件 |
| --- | --- |
| General 460 | `checkpoints/General/checkpoint-step-00000460.pt` |
| Update 48 | `checkpoints/my/update-00000048.pt` |
| PPO 104 | `checkpoints/my/update-00000104.pt` |
| PPO 1208 | `checkpoints/my/update-00001208.pt` |
| BC 200 | `checkpoints/my/update-00000200.pt` |

卡组池读取 `.env` 的 `CR_PPO_DECK_POOL`，未设置时使用 `runs/deck-pool.json`。也可直接加 `--deck-pool /实际路径/pool.json`。缺少模型或池文件会停止，不使用固定标准卡组补位。

需要调整时复制 `round_robin.example.json` 为 `round_robin.local.json`，编辑其中的路径或设置。该本机配置不会提交 Git。支持 `--config` 选择另一份配置。

示例对局参数是依据已有训练记录给出的候选值：16 级、每次决策随机抽取 18–24 tick 执行延迟、连续动作间隔 4–8 tick、确定性出牌。它们不代表已实际读取服务器 `trainer-options.txt`；正式运行前请对照服务器保存的阶段参数核实。采样模式必须显式提供对所有模型一致的 `gate_temperature`。设备默认 `cpu`。

## 规模与分工

默认 `--blocks 0-99`，上下界都包含。每个组号在每对模型中各跑四局正常开局：双方交换位置 × 交换完整卡组。每对 400 局，五模型十对共 4000 局。16 槽同时推进四个完整组。

只检查完整流程可用 `--blocks 0-3`，十对共 160 局；该结果仅适合检查流程。

静态分布运行：

```bash
# 已准备 Linux KVM 引擎环境的服务器，启动一次引擎跑完所有模型对
bash round_robin.sh --blocks 0-79

# Mac：使用通过 ADB 转发的外部引擎
bash round_robin.sh --existing-engine 127.0.0.1:28000 --blocks 80-99
```

服务器支持 `--engines N`，按完整组静态划分；Mac 以上命令始终只有一个引擎。两台机器需使用相同代码、模型、卡组池、设备类型和对局设置。CPU 线程数、文件绝对路径、端口及负责的组号不属于研究身份。

## 续跑与合并

每次创建 `evaluations/rr-*/`，逐局写入日志。重复同一命令会扫描同身份历史，按“模型对＋全局组号”跳过已完整正常结束的四局组。不完整组重新跑；截断不计平局。不要删除历史目录。

复制 Mac 的整个 `rr-*` 原始目录到服务器，例如 `mac-results/`，再运行：

```bash
python -m native_runner.round_robin report evaluations/rr-*/ mac-results/rr-*/
```

合并只读原始数据，不需要加载模型或连接引擎。同一完整组的重复副本只计一次，同一对局出现冲突结果则报错。模型 SHA-256、卡组池 SHA-256、对局设置、代码身份或引擎/资源身份不同会拒绝合并。

每次运行自动生成自己的 `report/report.md` 和 `report/summary.json`，包含本机同身份完整历史。显式合并默认输出到 `evaluations/round-robin-report/`。

## 统计与数据

报告提供每对胜负平、得分率及完整组 bootstrap 的 95% 区间。等级分采用 Bradley–Terry 拟合、Elo 尺度，General 460 固定为 0；不是训练过程中已有的等级分。拟合每对每方向加入 0.5 个伪胜，使全胜/全负仍得到有限值，并在报告中明确标注。

等级分只使用所有模型对共同完整的组号，每次 bootstrap 同时重采样这些组号，再重新拟合，保留共享卡组/种子的相关性。只有部分模型对完成时先报告每对得分率。少于两组时不计算区间；样本极少或结果完全一致时区间可能过窄。

运行目录包含：

- `study.json`：全部模型 SHA-256、对局设置、池和代码身份。
- `settings.json`：本机实际使用的模型路径、池路径、设备和线程数。
- `shard-*/engine.json`：引擎、probe 和内容资源的实际身份。
- `shard-*/pair-*/plan.json`、`identity.json`、`matches.jsonl`：赛程、实际加载模型身份、逐局原始结果。
- `shard-*/pair-*/timing.json`：引擎重置、张量化、模型前向、动作选择和引擎步进墙钟耗时。步进墙钟包含传输和观测处理，不等同于纯 native CPU 时间。
- `report/matches.jsonl`：去重后的完整组对局，带模型名字，方便后续分析。

`duel.sh` 保持原用途，用于正常/强制开局对照。
