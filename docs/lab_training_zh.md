# 单机 PPO 训练记录与计划

[首页](../README.zh-CN.md) · [训练指南](training.md) · [实战模式](live_mode_zh.md)

这份文档记录在一台单机 Linux 服务器上搭训练环境的过程、测到的数据、已经定下的决定，以及接下来要做的事。目标：从 General 出发，用一周时间训练一个**更强的通用对练模型**。

## 1. 运行条件

- Linux x86_64，有 `/dev/kvm`；不需要 Slurm。
- NVIDIA GPU，一块即可；引擎多时可以用两块。
- 内存要够同时跑多台模拟器（每个引擎约 4 GB），`/dev/shm` 放模拟器的运行时文件。
- 硬盘留出几十 GB，放母盘、检查点和日志。
- 如果机器和别人共用，启动前看一下端口和 CPU 占用。

## 2. 环境搭建（已完成）

### Android SDK 与模拟器

- SDK 在 `~/android-sdk`；`~/.bashrc` 里设置了 `ANDROID_HOME` 和 `PATH`，包含 emulator、platform-tools、cmdline-tools。
- 母盘 AVD 名为 `cr-base`：android-31 google_apis x86_64，自带 ARM64 转译（libndk_translation），需要 `adb root`。
- 原版游戏在这台模拟器上可能卡在启动画面，疑似因为连不上 Google 服务，一直在等。**训练不需要原版能启动**。

### 游戏资源（从 MuMu 复制）

1. 在 MuMu 上把 `/data/user/0/nullsroyale.rel.free/update` 打包成 `nr-update.tar`。
2. 推进 `cr-base`，解压到同一位置；`chown` 改成应用自己的 uid；运行 `restorecon`。
3. 用 `check_prepared_device` 验证，结果为 `resources ok`。

### Python 环境

用 uv 建 Python 3.12 虚拟环境。仓库要求 3.11 以上，服务器系统自带的 Python 版本太低。

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

### 离线 APK 与母盘

1. 在 Windows 上构建 6 引擎版 APK：`.\build_offline_engine.ps1 -EngineCount 6 -OutputApk .\build\cr-ai-train6.apk`。
2. 把 APK、同名的 `.build.json` 和 `libcrprobe.so` 拷到服务器。
3. 在 `cr-base` 上运行 `tools/prepare_training_guest.py`，再用 `configure_emulator_engine_guests --engines 1` 检查，结果 `ok: true`。
4. 关掉 `cr-base`，**从 `.qcow2` 叠加层导出**数据盘和加密密钥，两样都要导：
   ```bash
   QI=~/android-sdk/emulator/qemu-img; cd ~/.android/avd/cr-base.avd
   $QI convert -p -O raw userdata-qemu.img.qcow2   ~/cr-train/prepared-userdata.img
   $QI convert -p -O raw encryptionkey.img.qcow2   ~/cr-train/prepared-encryptionkey.img
   ```
   底层的 `userdata-qemu.img` 是开机前的空盘，从它导出就是一张空盘。

### 踩过的坑（都已修好并合并）

| 问题 | 原因 | 修复 |
| --- | --- | --- |
| 克隆机的引擎 `production_ready: false`，`content_assets_mapped: false` | Android 12 用 `encryptionkey.img` 加密数据盘，克隆机拿到的是出厂密钥；而且 37 版模拟器不理会 `-initdata`，直接用空盘 | 新增 `CR_EMULATOR_ENCRYPTIONKEY`；启动脚本自己把母盘复制成克隆机的 `-data` |
| 训练器启动即退出 | 启动脚本把种子默认设成 0，训练器要求所有参数为正数 | 默认值改为 20260827 |
| 训练器只接受 7,695 套牌组 | 公开数据集按同样规则只统计出 7,089 套 | 新增 `CR_PPO_DECK_POOL_SIZE`，默认仍是 7695，填 0 表示不检查 |
| 采样评测太慢（约 1 倍实时） | 只有确定性解码用了 CUDA Graph | 采样也走 CUDA Graph |
| `--devices cuda:1` 的评测进程崩溃 | 终端里残留 `CUDA_VISIBLE_DEVICES=0`，子进程只看得到一块卡 | 每个子进程只看得到分给它的那一块卡（编号按 PCI 总线顺序，和 nvidia-smi 一致） |

### `.env` 里和训练有关的设置

`<user>` 换成服务器上自己的用户名。

```
CR_AI_PYTHON=/home/<user>/FirstLight_CR/.venv/bin/python
CR_TRAINING_ADB=/home/<user>/android-sdk/platform-tools/adb
CR_TRAINING_SDK_ROOT=/home/<user>/android-sdk
CR_EMULATOR=/home/<user>/android-sdk/emulator/emulator
CR_SOURCE_AVD=/home/<user>/.android/avd/cr-base.avd
CR_SOURCE_AVD_INI=/home/<user>/.android/avd/cr-base.ini
CR_EMULATOR_INITDATA=/home/<user>/cr-train/prepared-userdata.img
CR_EMULATOR_ENCRYPTIONKEY=/home/<user>/cr-train/prepared-encryptionkey.img
CR_TRAINING_APK=/home/<user>/FirstLight_CR/build/cr-ai-train6.apk
CR_EMULATOR_COMPAT=/home/<user>/cr-train/compat
CR_PPO_CHECKPOINT=/home/<user>/FirstLight_CR/checkpoints/General/checkpoint-step-00000460.pt
CR_PPO_DECK_POOL=/home/<user>/FirstLight_CR/runs/deck-pool.json
CR_PPO_DECK_POOL_SIZE=7089
```

### 牌组池

只下载了 IL_Replay 的 `replays/` 部分，走 hf-mirror 镜像，存到 `datasets/IL_Replay/`。然后生成牌组池：

```bash
python -m native_runner.training.v4.build_ppo_deck_pool --dataset-root datasets \
  --output runs/deck-pool.json --minimum-uses 5 --focus-decks native_runner/training/v4/focus_decks_20.json
```

结果是 7,089 套牌组，其中 20 套是重点卡组。

## 3. 启动训练

### 一键启动（推荐）

`lab_ppo.sh` 在后台依次拉起引擎、worker、拓扑和训练器，关掉终端也不影响。日志写到 `runs/名字/logs/`；训练器只肯往空目录里写，所以每次启动或续训，指标和检查点都写进一个新的 `runs/名字/train-时间戳/`：

```bash
cd ~/FirstLight_CR && git pull
bash native_runner/training/v4/lab_ppo.sh start spar-week1          # 新开一个训练，默认跑到第 30000 次更新
bash native_runner/training/v4/lab_ppo.sh status spar-week1         # 进度、最近对局的开局时间、检查点、剩余硬盘
bash native_runner/training/v4/lab_ppo.sh load spar-week1           # CPU 各组和 GPU 的实时占用，以及每次更新的时间花在哪
tail -f runs/spar-week1/logs/supervisor.log runs/spar-week1/logs/trainer.log
bash native_runner/training/v4/lab_ppo.sh stop                      # 停止并清理
bash native_runner/training/v4/lab_ppo.sh resume spar-week1         # 中断后从最新的检查点续训
```

- **启动前**会先清理上次残留的模拟器、worker、训练器和 `/dev/shm/firstlight-ppo-local-*`；**退出时**（训练结束、`stop`、出错）也会自动清理。只动自己账号下、名字是 `crppo_*` 的模拟器，不碰别人的。
- **默认单卡**（GPU 0），CPU 按 `CR_LAB_ENGINE_CPUS` / `CR_LAB_WORKER_CPUS` / `CR_LAB_CONTROL_CPUS` 分配（默认值见脚本开头，按自己的核数调整）。有第二块卡时，可以在训练中途拿来评测。改用双卡：`CR_LAB_TRAIN_GPUS=0,1`；双卡之间通信出问题时再加 `NCCL_P2P_DISABLE=1`。
- **训练参数**默认是第 6 节的对练配置，实际用的参数存在 `runs/名字/trainer-options.txt`，续训时会自动沿用。要换参数用 `options`（见下面「中途改参数」），不用换名字。
- **检查点**：每打完一批对局存一个完整检查点（约 48 MB，大约每 8 次更新一个）。脚本只保留最新 3 个，再加每 1000 次更新的第一个，一周大约 1–2 GB。对手池快照每 200 次更新一个。
- 剩余硬盘不足 5 GB 时，`supervisor.log` 里会出现警告。
- **`status` 怎么看进步**：
  - 最后几行是学习者对三个固定对手的胜率：「first」是本次训练最早的约 300 局，「latest」是最近的约 300 局。latest 比 first 高，说明在进步。
  - 自我对战的胜率永远在 50% 左右，看不出进步，所以不显示。
  - 同时看开局时间（首次出牌的中位数）是否在往前移、每局出牌次数是否正常。
- **`load` 怎么看**：
  - 「GPU inference」是 GPU 推理的时间；「trainer batching」是训练器拼批（训练器的核）；「waiting for engines」是在等引擎推进和 worker 解析画面（引擎和 worker 的核）；「optimization」是 GPU 训练。
  - 等引擎的时间占比最大，并且 engines 或 workers 那组核接近 100%：说明瓶颈在 CPU。
  - GPU inference 占比最大，并且 GPU 利用率高：说明瓶颈在 GPU 推理。

### 每天定时运行

服务器只在固定时段可用时，用 `schedule` 每天自动续训、自动停止：

```bash
bash native_runner/training/v4/lab_ppo.sh schedule spar16-g3 09:00 21:00   # 示例：每天 9:00 续训，21:00 前停（换成自己可用的时段）
bash native_runner/training/v4/lab_ppo.sh unschedule                        # 取消定时
crontab -l                                                                  # 查看定时任务
tail runs/schedule.log                                                      # 定时任务的日志
```

- 停止时间前 12 分钟开始等下一个检查点，一存好就停，所以几乎不丢进度；最晚到停止时间强制停。
- **全天自动重试**：从开始时间到停止流程开始前，每 15 分钟检查一次（可以用第 4 个参数只检查开始后的若干分钟，比如 `schedule spar16-g3 09:00 21:00 90`）。如果训练不在运行，并且不是被主动停止的，就自动续训。会触发重试的情况包括：交接时显存不够导致训练器崩溃；引擎的 probe 偶尔拒绝一条战斗事件（`rich.combatEvents capture is incomplete or rejected a record`，这是数据校验故意报错，会让整次采集作废）。
- 训练器或引擎非正常退出时，会把每台模拟器最近的系统日志（logcat）存到 `runs/名字/logs/guest-logcat-时间/`，probe 拒绝数据的具体原因就记在里面。
- 「主动停止」指手动运行 `stop`，或者每天的定时停止，它们会留下记号 `runs/lab-stopped-on-purpose`，重试看到这个记号就不会把训练拉起来。下一次 `start` 或 `resume` 会清掉这个记号。
- 续训会自动用上第一次 `start` 时的引擎数、CPU 分配和 GPU 设置，这些都存在 `runs/名字/launch-env` 里，不用再手动带变量。
- 定时任务用的是 `.env` 里的设置，不受当前终端里残留变量的影响。

### 中途改参数

- **续训时可以直接改**：总更新次数 `UPDATES`、`CR_PPO_INFERENCE_BATCH_WAIT_MS`。
- **训练参数**（奖励、对手、延迟、出牌间隔、gate 温度等）用 `options` 改，训练名不变，定时任务也不用动：

```bash
bash native_runner/training/v4/lab_ppo.sh options spar16-gw                                   # 查看当前参数，一行一个
bash native_runner/training/v4/lab_ppo.sh options spar16-gw --min-play-gap-ticks 4-8          # 新增或替换一项
bash native_runner/training/v4/lab_ppo.sh options spar16-gw --unset --game-elixir-overflow-grace   # 删除一项
```

  - 可以一次写多项。可重复的参数（如 `--fixed-opponent-checkpoint`）会整体替换：要保留的那几个也要一起写上。
  - 改完后的下一次续训（手动 `resume`，或者每天的定时启动）会发现参数和最新存档训练时用的不一样，自动**热启动**：用最新存档的模型权重和优化器状态开一个新的 `train-*` 阶段，只有更新计数重置。之后的续训就是普通续训。
  - 正在跑的训练不受影响，要马上生效就先 `stop`，再 `resume`。
  - 热启动后 gate 温度的退火会从头再来一遍（0.5 → 0.2，3000 次更新）。不想要的话，顺手改成固定值：`options 名字 --gate-temperature-start 0.2`。
- 续训时带 `CR_PPO_EXTRA_ARGS="..."`，等于把参数整体换成这一串，效果同上。
- **引擎数、GPU 数**存在 `runs/名字/launch-env`，改它们要在 `resume` 命令前带上新值，比如 `CR_PPO_ENGINE_COUNT=12 bash ... resume 名字`。

### 手动启动（排查问题时用）

单机没有 Slurm，设 `SLURM_JOB_ID=local` 就能通过脚本的检查。

| CPU 核（示例，按自己的核数调整） | 用途 | 端口 |
| --- | --- | --- |
| 0–7 | 2 台模拟器，各 4 个引擎 | 引擎 28000–28007 |
| 8–15 | worker | 29900 |
| 16–19 | 训练器 | |

```bash
# 终端 1：引擎节点，看到 ENGINE_CLUSTER_READY 才算启动成功
export SLURM_JOB_ID=local TMPDIR=/dev/shm CR_PPO_ENGINES_PER_GUEST=4
taskset -c 0-7 bash native_runner/training/v4/launch_ppo_engine_node.sh runs/node0 0 8
# 终端 2：worker 服务
taskset -c 8-15 python -m native_runner.training.v4.remote_ppo_worker_server \
  --bind-host $(hostname -s) --port 29900 --global-start 0 --engine-count 8 --engine-base-port 28000
# 终端 3：生成拓扑，再启动训练（这里是单卡写法；双卡用 --gpus 2）
python -m native_runner.training.v4.build_ppo_multinode_topology --node-ready runs/node0/engine-node-ready.json \
  --output runs/topology.json --trainer-job-id local --trainer-host $(hostname -s) --gpus 1 \
  --worker-cpus 8-15 --control-cpus 16-19
bash native_runner/training/v4/launch_ppo_multinode_training.sh runs/ppo-smoke8 runs/topology.json 3 40
```

**收尾**（启动脚本自己不会清理）：

```bash
touch runs/node0/STOP_ENGINES
pkill -u $USER -f "remote_ppo_worker_server|tcp_endpoint_proxy"
rm -rf /dev/shm/firstlight-ppo-local-* runs/node0
```

## 4. 性能观察

- **瓶颈在训练器的推理循环**，不在 CPU 或 GPU 的算力：CPU 和 GPU 都远没跑满。训练器用一个线程给所有引擎做推理，每批要给当前模型和每个固定对手各跑一次前向，每次的固定开销都差不多，这部分占了每次更新的大半；GPU 训练本身占比很小。
- **加大凑批等待时间没有帮助**（`CR_PPO_INFERENCE_BATCH_WAIT_MS`）：批数是少了，但等引擎的时间同样变长，总吞吐不变。它不影响训练参数，可以停下来换一个值再续训对比。
- **有效的是增加引擎数**：引擎和 worker 可以共用核心（`CR_PPO_ENGINE_CPU_SHARING=1`，CPU 不够分时脚本会自动打开）。引擎数写在续训合同里，所以只能新开一个训练。
- **引擎多时用双卡**：引擎多了以后，单卡在优化阶段可能显存不足，这时设 `CR_LAB_TRAIN_GPUS=0,1`。引擎少时单卡反而更快（双卡要把引擎分给两个进程，每批更小）。
- **entropy**：起点 General 的 entropy 很低，训练中要盯着看。
- **采样评测**：每个引擎大约 1 倍实时，CPU 主要花在 Python 解析观测上。
- 用 `lab_ppo.sh load` 看每次更新的时间花在哪，再决定往哪个方向调。

## 5. 延迟实验（21 tick ≈ 实战点击延迟）

设置：General 对 General，B 方每次出牌和放技能都推迟 21 tick 生效；采样解码，两方用相同的默认卡组，100 局。

```
{"completed": 100, "a_wins": 80, "b_wins": 20, "truncated": 0, "rejected_actions": 18,
 "a_actions_per_game": 38.7, "b_actions_per_game": 38.1}
```

- **被延迟的一方只赢了 20%**，95% 置信区间约 13–29%。两边下牌次数一样，说明它输在时机上，不是出牌少。
- 这个结论推翻了“延迟影响不大”的猜测：General 的打法依赖“出牌马上生效”。
- 之前跑的 10 局（B 赢 6 局）只是噪声。

复现命令：

```bash
python -m native_runner.training.v4.evaluate_parallel --ports 28000-28007 --games 100 \
  --output runs/delay-100 --devices cuda:0,cuda:1 --checkpoint-a $G --checkpoint-b $G --delay-b-ticks 21 --sample
```

## 6. 已经定下的决定

- **做通用对练模型，不做专精。** 对练要什么卡组都能打、打法正常。
- **训练时加入执行延迟。** 学习者和对手都加，延迟在 18–24 tick 之间随机。
- **两张牌不能同时落地。** `--min-play-gap-ticks 4-8`：同一方连续两次出牌或技能，落地至少相隔随机 4–8 tick（1 tick = 50 ms，即 200–400 ms），和人手点两张牌的间隔一致。原来一次决策最多能出 2 张牌，而且同一 tick 落地；相邻两次决策各自抽延迟，也可能几乎同时落地。新开的训练默认带上；已有的训练用 `options 名字 --min-play-gap-ticks 4-8` 加上，下一次续训自动热启动。评测器也有同名参数，对双方都生效。
- **对手**：自我对战 40%；固定对手 60%，从冻结的 General、Active IL、速猪专精 1 里均匀抽。原版 IL 占 0%（只作为锚点登记，不上场）。原版 IL 很被动，打它学到的是等对方犯错。
- **开局要主动**（见下面的测试）：
  - 圣水溢出惩罚 `--personal-elixir-overflow-penalty 0.01`，满格后多浪费 1 点就开始扣（`--personal-elixir-overflow-grace 1`）；
  - gate 温度在前 3000 次更新里从 0.5 降到 0.2，让模型前期多尝试早出牌，最后回到默认值。
- **整局圣水浪费惩罚（可选）**：`--game-elixir-overflow-penalty C --game-elixir-overflow-grace G`。从学习者**第一张牌真正下出之后**开始，把整局满格浪费的圣水累加，超过 G 点后每多浪费 1 点扣 C。开局为了等对手而满格等待不算在内。上面那条按段计算的惩罚，每次圣水降到 10 以下就清零，所以一局里可以反复白白浪费；这一项补上了整局总量。参照：人类录像里胜方整局浪费的中位数约 2.3 点（`elixir_leaked`）。`status` 会显示「after the first card」这一行。已有的训练用 `options` 加上，下一次续训自动热启动。
- **过早先手惩罚（可选）**：`--early-first-play-penalty C --early-first-play-free-seconds T`。学习者在对手还没有牌落地时抢先下出全场第一张牌，按出牌时间扣分：第一个决策点（4.5 秒）扣 C，线性降到 T 秒时为 0。对手已经先下牌时应对不扣。建议 C = 0.1、T = 14（开局 5 点圣水涨满的时间）。和满圣水惩罚一起，给开局留出 14–22 秒的自由区间。`status` 会显示先手被罚的比例。已有的训练用 `options` 加上，下一次续训自动热启动。
- **固定对手开局（可选）**：`--fixed-opponent-opening-force-after-seconds 8-20`。固定对手（General、Active IL、速猪专精等）开局都要等很久才出牌，学习者能靠「对手发呆」占便宜，开局越练越早。这个参数让固定对手如果到了每局随机抽的时间（8–20 秒之间）还没出牌，就强制它出第一张牌（出什么、放哪仍由它自己的模型决定）。学习者于是常常面对主动开局的对手，可以选择等对手先出再应对。改的是对手行为，不是奖励。`status` 会显示被强制开局的比例和中位时间。也可以写单个数，如 `20`。
- **GPU 数量看引擎数。** 引擎少时单卡更快；引擎多到单卡显存不够时用双卡。
- **对手池快照间隔**改成 200 次更新一次。默认每 8 次一次，几天就会把硬盘写满。
- **评测要看两件事**：一是双方都带延迟时，训练出的模型对 General 的胜率；二是每局出牌次数，以及第一张牌出得早不早。不只看胜率。

### 开局等待问题（10 次更新的测试）

General 习惯等到双倍圣水（约 120 秒）才出第一张牌，原作者在 `training_history.md` 里也记录过这个现象。下表是学习者第一次出牌时间的中位数：

| 配置 | 自我对战 | 打固定对手 | 每局出牌 |
| --- | --- | --- | --- |
| 默认（100% 打原版 IL） | — | 172 s（打原版 IL） | 34 |
| 换对手 + 溢出惩罚（`spar-test`） | 155 s | 121 s | 33–35 |
| 再把 gate 温度提到 0.5（`spar-test2`） | 47 s | 29 s | 39–41 |

- 默认的 gate 温度是 0.2，把「出牌还是等」的 logit 放大了 5 倍，开局几乎总是等。所以开局阶段的 entropy 是 0，模型从来没有尝试过早出牌。
- 10 次更新几乎没让模型学到东西，上表的变化主要来自采样方式变了。真正要练到人类水平（开局 10–20 秒出牌），靠的是一周里溢出惩罚慢慢起作用。训练中用 `lab_ppo.sh status` 看开局时间是否在往前移；如果第 1–2 天没有进展，再考虑专门给「第一张牌」加奖励，或者用人类录像约束（`--expert-bc-manifest`）。

## 7. 待办（完整版）

1. **训练加延迟**（已完成）
   - 训练器参数 `--execution-delay-ticks 18-24`（也可以写单个数，如 `21`）。每次决策随机抽一个延迟，双方的出牌和技能都在决策后这么多 tick 才生效，模型自己选的延迟偏移被忽略（和实战一致）。
   - 开了延迟后，被拒绝的出牌只计数（W&B 里的 `safety/rejected_actions`），不再让这一轮采集作废；不开延迟时仍然要求为 0。
   - 延迟写进续训合同：带延迟训练出的检查点，续训时也必须带同样的参数。
   - 启动脚本新增 `CR_PPO_EXTRA_ARGS`，原样追加到训练器命令后面。
   - 评测器新增 `--delay-a-ticks`，配合 `--delay-b-ticks` 在双方都有延迟的条件下评测。
   - 冒烟测试（按第 3 节启动引擎和 worker 后，终端 3 改成）：

     ```bash
     CR_PPO_EXTRA_ARGS="--execution-delay-ticks 18-24" \
       bash native_runner/training/v4/launch_ppo_multinode_training.sh runs/ppo-delay-smoke runs/topology.json 3 40
     ```

     看 3 次更新都能跑完，再看 `safety/rejected_actions` 有多少、每局出牌次数是否正常。
2. **一键启动脚本**（已完成）：`lab_ppo.sh`，用法见第 3 节。
3. **对练专用的调整**
   - 对手配置和开局主动性（已完成，见第 6 节）；
   - 训练中盯开局时间。如果不往前移，就给第一张牌加奖励，或者研究 `--expert-bc-manifest`（用人类录像约束打法，需要先建录像数据）；
   - 按常见流派调整重点卡组或它的占比，让卡组覆盖面更广。
4. **可选优化**：让引擎数多于核数、共用核心。引擎在等推理时 CPU 是空闲的，这样可能提高吞吐。

流程：用 `lab_ppo.sh start` 正式跑一周；第 3 天和结束时各评测一次（双方都带延迟，对 General 的胜率，以及开局时间和每局出牌次数）。
