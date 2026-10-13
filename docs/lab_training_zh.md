# 实验室单机 PPO 训练记录与计划

[首页](../README.zh-CN.md) · [训练指南](training.md) · [实战模式](live_mode_zh.md)

这份文档记录在实验室 Linux 服务器上搭训练环境的过程、测到的数据、已经定下的决定，以及接下来要做的事。目标：从 General 出发，用一周时间训练一个**更强的通用对练模型**。

## 1. 服务器

| 项目 | 情况 |
| --- | --- |
| CPU | 20 核，一个 NUMA 节点（0–19） |
| 内存 | 250 GB；`/dev/shm` 放模拟器运行时文件 |
| GPU | 2 × RTX A5500。没有 NVLink，两块卡之间的拓扑是 `NODE`（同一颗 CPU、不同 PCIe 控制器）。对双卡训练影响可以忽略 |
| 硬盘 | 可用约 50 GB |
| 系统 | Ubuntu 22.04，有 `/dev/kvm`，没有 Slurm |
| 共享 | 服务器上还有别人的模拟器在跑：启动前看一下端口和 CPU 占用 |

## 2. 环境搭建（已完成）

### Android SDK 与模拟器

- SDK 在 `~/android-sdk`；`~/.bashrc` 里设置了 `ANDROID_HOME` 和 `PATH`，包含 emulator、platform-tools、cmdline-tools。
- 母盘 AVD 名为 `cr-base`：android-31 google_apis x86_64，自带 ARM64 转译（libndk_translation），需要 `adb root`。
- 原版游戏在这台模拟器上会卡在启动画面。疑似因为国内连不上 Google 服务，一直在等。**训练不需要原版能启动**。

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

## 3. 手动启动（一键脚本完成前）

单机没有 Slurm，设 `SLURM_JOB_ID=local` 就能通过脚本的检查。

| CPU 核 | 用途 | 端口 |
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

## 4. 实测数据

| 测试 | 结果 |
| --- | --- |
| 单引擎 PPO 冒烟（1 次更新） | 1,280 帧，采样 15.8 秒（约 20 倍实时），优化 0.9 秒 |
| 8 引擎 PPO（3 次更新） | 每次更新 10,240 帧，约 25 秒；吞吐约 **405 帧/秒**，是单引擎的 5.5 倍。瓶颈是推理（采样 20.5 秒里推理占 14.3 秒），引擎推进约 10 秒 |
| entropy | 起点 General 约 1e-5；3 次更新后到 0.023，训练中要盯着看 |
| 采样评测速度 | 每个引擎约 1 倍实时。CPU 被 Python 解析观测占满；8 个引擎同时跑，100 局约 56 分钟 |

**一周的粗略估算**：按 400 帧/秒算，一天约 3,500 万帧，一周约 2.4 亿帧，大约是 General 当初 PPO 总帧数（约 1.1 亿）的两倍。不过每次更新的批量只有原作者的 1/24，梯度噪声更大。

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
- **对手池**：自我对战 50%；固定对手（冻结的 General、速猪专精 2）30%；锚定对手 Active IL 20%。不用被动的 IL 当主要对手，避免模型靠欺负被动对手刷胜率。
- **双卡训练**：`--gpus 2`，CPU 分配见第 3 节。
- **对手池快照间隔**改成约 200 次更新一次。默认的每 8 次一次，几天就会把硬盘写满。
- **评测要看两件事**：一是双方都带延迟时，训练出的模型对 General 的胜率；二是每局出牌次数，以及第一张牌出得早不早。不只看胜率。

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
2. **一键启动脚本**
   - 一条命令在后台拉起引擎、worker、拓扑和训练器，日志写到文件；
   - 使用双卡，并按第 3 节分配 CPU；
   - `CR_PPO_EXTRA_ARGS` 透传训练器参数（对手池、快照间隔、延迟）；
   - 启动前检查并清理上次残留的进程和 `/dev/shm` 目录，退出时（正常、Ctrl+C 或出错）自动清理；
   - 预留 `NCCL_P2P_DISABLE` 开关；
   - 中断后用 `CR_PPO_RESUME_CHECKPOINT` 续训。
3. **对练专用的调整**
   - 研究 `--expert-bc-manifest`：用人类录像约束打法，让它更像人；
   - 按常见流派调整重点卡组或它的占比，让卡组覆盖面更广。
4. **可选优化**：让 12 个引擎共用 8 个核，引擎在等推理时 CPU 是空闲的，这样可能提高吞吐。

流程：先在服务器上用 3 次更新做带延迟的冒烟测试，再正式跑一周；第 3 天和结束时各评测一次。
