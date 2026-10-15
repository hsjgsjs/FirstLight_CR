# 从当前 PPO 模型尝试人类录像训练

通用人类录像作为额外的行为模仿监督（BC），PPO 继续根据实际对局训练。人类录像不会混入 PPO 的在线采样缓冲，也不会用录像的胜负训练价值头。共享编码器仍会随模仿更新，因而不能保证模型实力不下降，需要保留基准并做对照评测。

以下命令从仓库根目录执行。示例从 `update-00001208.pt` 出发，训练 200 次新的 PPO 更新。先试一小段，不覆盖 `spar16-from104`。

## 1. 移入精选数据和重建缓存

需要两部分：

- 精选目录中的 `selection.jsonl`：由 `select_expert_replays` 生成，逐场记录要学习的原生 owner、胜方或接近的败方及其权重。
- 完整 V4 缓存目录：含 `index.json`、分片 JSON 和 `.cril.zst` 文件。分区缓存也可以提供含 `part-*/index.json` 的父目录。

只有原始录像 Parquet 时，需要先重建。使用空闲引擎的实际端口，例如：

```bash
bash native_runner/training/v4/launch_cache_partition.sh \
  /path/to/selected-dataset /path/to/expert-cache \
  --ports 26789 --source-count 0 --resident-slots 8 --verify-roundtrip
```

端口及 resident slots 必须符合该引擎的实际配置。已经重建的缓存无需再重建。迁移时保留目录结构；在训练机器上重新生成下面的清单，清单引用的是该机器上的实际路径。

## 2. 审计缓存并划分训练与验证

替换两处 `/path/to/...`：

```bash
.venv/bin/python -m native_runner.training.v4.build_expert_manifest \
  --selection /path/to/selected-dataset/selection.jsonl \
  --cache-root /path/to/expert-cache \
  --output runs/human-experts-v1 \
  --validation-percent 10 \
  --workers 4

cat runs/human-experts-v1/summary.json
```

支持重复 `--cache-root`，以及直接传入 `index.json`。审计会校验缓存哈希、重建结果和实际监督标签，只保留有可信动作与 gate 标签、自然时间和循环序列起点的选定视角。镜像或重复缓存的同一视角会报错，避免不明确地选中一份缓存。

`--workers 4` 使用四个进程并发处理不同分片，每个进程的 PyTorch CPU 线程数为 1，避免线程争抢。默认 `--workers 1` 为串行。先试 4，内存和磁盘有余量时可试 8；同时展开的分片数最多为 worker 数，因此进程增多也会增加 CPU 内存占用。解压、哈希和张量审计都在 CPU 上，不使用 GPU。进度日志会显示已完成分片数、耗时与 `shards_per_second`；并发与串行生成的训练/验证划分和清单内容一致。

输出 `train.json`、`validation.json`、`summary.json` 和 `excluded.json`。默认只接纳完整重建成功的录像；失败、无有效动作和缺失缓存的视角会被排除并报告。默认不使用失败录像的前半段。如确有需要，可显式加入 `--allow-trusted-prefix`，仅接纳特定终局不一致/提前终局失败的已有可信前缀掩码，不接纳动作拒绝或其他错误。

验证集按整场录像划分，同一场双方永远在同一集合。训练器拒绝将 `validation.json` 用作通用训练清单。验证清单留作独立检查，不会自动进行验证、选优或回滚。

## 3. 固定基准与两份实验参数

```bash
.venv/bin/python -m native_runner.training.v4.prepare_expert_experiment \
  --checkpoint runs/spar16-from104/train-20261015-055801/checkpoints/update-00001208.pt \
  --trainer-options runs/spar16-from104/trainer-options.txt \
  --expert-manifest runs/human-experts-v1/train.json \
  --output runs/human-trial-v1

cat runs/human-trial-v1/plan.json
```

该命令不启动训练。它把源检查点完整复制到 `baseline.pt`，要求检查点包含优化器状态，生成 `ppo-options.txt` 与 `human-options.txt`。两份参数都复用优化器，并从检查点保存的当前温度继续剩余退火。例如原退火 3000 次、源更新 1208，则新实验从保存的下一次温度开始，剩余 1792 次退火；不会重新回到 0.5。

除去旧 BC 选项及接续退火需要调整的计数，原对手、奖励、延迟等参数全部保留。模型、卡组池和机器资源仍使用当前训练机器的配置。输出目录必须为空，避免覆盖基准。旧实验若已有 BC，会从对照参数中去掉，保证对照为纯 PPO。

默认人类参数：

```text
--expert-bc-coefficient 0.1
--expert-bc-every-updates 4
--expert-bc-records-per-rank 2
--expert-bc-window-steps 128
```

每 4 次 PPO 更新，额外做一次独立的模仿优化器更新。每个 GPU 选两个专家视角，胜方/败方权重控制抽样频率；监督开局约 32 游戏秒和一个均匀抽取的后续约 32 秒窗口。如果两者都没有动作，则补充一个包含可信动作的上下文窗口。之前的循环状态上下文完整回放但不反向传播，监督保留可信 WAIT，不把出牌标签额外加权。野猪专用额外监督不用于通用入口。

`0.1` 是损失系数，不能解释为“10% 人类数据”或“参数仅改变十分之一”。Adam 和梯度裁剪会影响实际幅度；更新间隔以及对照评测同样重要。可在准备命令中调整 `--coefficient`、`--every-updates`。

## 4. 启动短实验

有训练仍在运行时先正常停止。实验期间取消旧任务的每日计划，防止它晚间自动启动；需要时再为新实验设置计划。

```bash
bash native_runner/training/v4/lab_ppo.sh stop
bash native_runner/training/v4/lab_ppo.sh unschedule

CR_PPO_CHECKPOINT="$PWD/runs/human-trial-v1/baseline.pt" \
CR_PPO_RESUME_CHECKPOINT="" \
CR_PPO_EXTRA_ARGS="$(cat runs/human-trial-v1/human-options.txt)" \
CR_LAB_KEEP_CHECKPOINTS=100 \
bash native_runner/training/v4/lab_ppo.sh start spar16-human-v1 200
```

这是 warm start：加载 1208 的权重和优化器，新阶段更新计数从零开始。`start ... 200` 表示跑新阶段的 200 次更新。提高保留数量用于留住短实验中的中间检查点；没有自动最佳模型筛选。

```bash
bash native_runner/training/v4/lab_ppo.sh status spar16-human-v1
tail -f runs/spar16-human-v1/logs/trainer.log
```

日志中每 4 次更新应出现 `ppo_expert_bc_complete`，并记录 `bc_gate_act_weight=1.0`、损失、监督数量、耗时与 `bc_step_probe_kl`。这个 KL 只测本次模仿更新在一小段当前在线数据上的变化，不是完整评测，也不会自动限制更新。

中断后用 `CR_LAB_KEEP_CHECKPOINTS=100 bash native_runner/training/v4/lab_ppo.sh resume spar16-human-v1 200`，必须保留清单和缓存位置。第一份严格检查点前中断时，恢复会重新从保存的 baseline 启动。

随后运行纯 PPO 对照。两者分开运行，因为本地 launcher 同时只支持一个训练任务：

```bash
CR_PPO_CHECKPOINT="$PWD/runs/human-trial-v1/baseline.pt" \
CR_PPO_RESUME_CHECKPOINT="" \
CR_PPO_EXTRA_ARGS="$(cat runs/human-trial-v1/ppo-options.txt)" \
CR_LAB_KEEP_CHECKPOINTS=100 \
bash native_runner/training/v4/lab_ppo.sh start spar16-ppo-control-v1 200
```

## 5. 比较后再扩训

比较 `baseline.pt`、纯 PPO 对照、PPO 加人类数据模型。在相同卡组、种子、双方位置、推理方式和延迟下，对同一组固定对手评测。训练最后 256 场统计和模仿损失都不能代替该比较。现有 `duel.sh` 是开局干预配对评测，不能直接把它的 forced/natural 差值当作这次模型提升。

同时亲手对战观察错误落点、防守、圣水管理和何时等待。首牌时间无需强行接近一个数值。若混合路线没有收益或出现退化，保留 baseline/对照，调整模仿频率、数据构成或监督窗口后再试；有收益才延长，或者关闭 BC 再继续 PPO 验证收益能否保留。

## 连续出牌间隔与提前量

新训练的默认 `--min-play-gap-ticks 8-12` 把同一方的连续出牌/技能间隔设为 0.4–0.6 游戏秒。它独立于 `--execution-delay-ticks 18-24`：首次出牌仍使用原有执行延迟，只有排得过近的后续动作会继续推迟。离线人机/双模型对战及 `live_play run` 默认采用同样的间隔范围，可通过 `--min-play-gap-ticks` 覆盖。

已经运行的试验保存了旧参数，更新代码不会改变它。为保留当前温度和剩余退火进度，先让当前 200 次试验结束或正常停止，再从最新检查点准备新阶段：

```bash
source_checkpoint="$(rg --files --no-ignore runs/spar16-human-v1 -g 'update-*.pt' | sort | tail -n 1)"
.venv/bin/python -m native_runner.training.v4.prepare_expert_experiment \
  --checkpoint "$source_checkpoint" \
  --trainer-options runs/spar16-human-v1/trainer-options.txt \
  --expert-manifest runs/human-experts-v1/train.json \
  --min-play-gap-ticks 8-12 \
  --output runs/human-trial-gap-v2

CR_PPO_CHECKPOINT="$PWD/runs/human-trial-gap-v2/baseline.pt" \
CR_PPO_RESUME_CHECKPOINT="" \
CR_PPO_EXTRA_ARGS="$(cat runs/human-trial-gap-v2/human-options.txt)" \
CR_LAB_KEEP_CHECKPOINTS=100 \
bash native_runner/training/v4/lab_ppo.sh start spar16-human-gap-v2 200
```

两份生成参数都使用 `8-12`；优化器、当前温度和剩余退火进度从检查点接续，其他原对手、奖励和延迟配置保留。新目录保留旧试验作比较；`200` 是新阶段的更新上限。更改间隔后的结果不适合直接当作仅改变 BC 的对照。

也可用 `options 名字 --min-play-gap-ticks 8-12` 改已有运行，再 `resume`；但这种改参数后的热启动会按保存的起始温度重新计数退火，需要保留退火进度时采用上面的准备入口。

火球命中移动目标和下兵拉扯都依赖提前量，但涉及的延迟不同：火球还包含飞行时间，下兵还包含部署时间和敌方转移目标的时机。稳定且一致的时序有利于 PPO 学习更早行动；只增加训练轮次不能保证改善。人类 BC 保留原录像动作时机，没有自动改写成更早的动作。后续专项应固定移动目标和速度，分别记录命中率、拉扯成功率与正常对战表现，并保留该等待的样本，避免只学会过早出牌。
