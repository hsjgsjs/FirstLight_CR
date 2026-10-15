# 出牌间隔、火球专项与实战录制

下面的训练命令在 Linux 训练服务器的仓库根目录执行；实战命令在 MuMu 所在 Windows 机器的仓库根目录执行。本次都是 Python 修改，不需要重装 APK。两台机器均更新原仓库代码。

通用在途动作、剩余时间输入、短期预测辅助任务、检查点与缓存迁移的详细设计见[时序架构计划](temporal_reasoning_plan_zh.md)，这些架构改动尚未实现。人类 BC 已完成的 200 次更新及判断边界见[实验记录](human_bc_trial_200_zh.md)。

当前优先处理的是通用时序判断：法术已在途中却重复防守、敌军会被塔击杀却额外补防。这些现象可能与短期结果估计、在途动作表示和实战延迟有关，不能只凭现象确定原因。先调好出牌间隔并录制实例，再核对模型当时真正看到的输入。下文火球专项是保留的可选入口，不会自动启用，暂不作为优先路线。

代码检查显示，当前输入已有单位速度、投射物位置/伤害/目标、塔攻击阶段和目标关系；尚无独立的、持续表示已提交动作及其剩余生效时间的输入队列。`expected_impact_tick` 目前用于实体保留优先级，没有作为明确的预计命中时间特征输入。网络有循环记忆，因此这些缺口并不意味着它完全无法推断未来，也尚未证明就是重复防守的根因。

通用改进候选是补齐可信的在途动作与剩余时间表示，再研究 0.5/1/2 秒短期位置、存活和血量变化的辅助预测。辅助标签只能用于训练，不能把真实未来输入实战策略。预测结果还要以这段时间内实际执行的动作作为条件；要判断“不再补防是否也安全”，需要无额外干预片段或受控反事实复现，不能直接拿已经补防后的幸存/伤害结果当标签。这套输入/辅助目标修改尚未实现。

## 1. 从最新检查点按 0.4–0.6 秒间隔继续训练

`--min-play-gap-ticks 8-12` 表示同一方连续出牌或技能至少间隔随机 8–12 个游戏 tick，每 tick 50ms。它不是每隔 0.4–0.6 秒强制出牌，也不会给独立的第一张牌再增加这段延迟。训练的 `--execution-delay-ticks 18-24` 继续保留。已有运行的 `4-8` 不会随代码更新自动变成 `8-12`。

为保留优化器、当前温度及剩余退火进度，使用准备入口生成新实验，而不是直接改旧运行参数后热启动。先让当前短实验结束；需要立即停止时运行 `bash native_runner/training/v4/lab_ppo.sh stop`，确认停止并写出检查点后继续。

### 只改变出牌间隔

```bash
source_checkpoint="$(rg --files --no-ignore runs/spar16-human-v1 -g 'update-*.pt' | sort | tail -n 1)"
test -f "$source_checkpoint" || exit 1
.venv/bin/python -m native_runner.training.v4.prepare_expert_experiment \
  --checkpoint "$source_checkpoint" \
  --trainer-options runs/spar16-human-v1/trainer-options.txt \
  --expert-manifest runs/human-experts-v1/train.json \
  --min-play-gap-ticks 8-12 \
  --output runs/human-gap-trial-v1

cat runs/human-gap-trial-v1/plan.json

CR_PPO_CHECKPOINT="$PWD/runs/human-gap-trial-v1/baseline.pt" \
CR_PPO_RESUME_CHECKPOINT="" \
CR_PPO_EXTRA_ARGS="$(cat runs/human-gap-trial-v1/human-options.txt)" \
CR_LAB_KEEP_CHECKPOINTS=100 \
bash native_runner/training/v4/lab_ppo.sh start spar16-human-gap-v1 200
```

新阶段训练 200 次更新，BC 仍为系数 0.1、每 4 次 PPO 做一次、每次两个专家视角；源对手、奖励和执行延迟保留。新目录保留源基准和旧试验。如果运行名已变更，替换命令中所有对应目录；输出目录需全新。

## 2. 可选：火球打可见移动目标的上下文采样

当前实现增加现有人类缓存中相关场景的采样频率，PPO 仍在完整正常对局中训练。没有硬编码火球落点，没有把所有人类动作统一提前，也不引入“必须预判”的奖励。

先生成一份带专项注释的训练清单，不需要重新跑模拟器或重建缓存：

```bash
.venv/bin/python -m native_runner.training.v4.fireball_focus \
  --manifest runs/human-experts-v1/train.json \
  --output runs/human-experts-v1/train-fireball-v1.json \
  --focus-percent 50 --workers 4
```

输出 `focused_sides` 和 `focus_frames`，表示有相关上下文的专家视角数与动作帧数。原清单的所有训练视角仍保留；不会读取验证集。没有找到符合条件的样本时会明确报错，不生成空训练清单。可对 `validation.json` 单独生成注释供后续检查；训练器仍拒绝验证集。

筛选规则使用缓存中的可信火球动作、可见敌方兵种的位置和速度：移动速度至少 0.25 格/秒，火球落点距敌军当前位置或未来 3 秒直线运动路径不超过 3 格。只包含兵种/英雄/角色，不包含建筑、区域和投射物。这个规则用于发现相关上下文，**不能证明该火球实际命中、落点最优或完成了成功预判**。

每个 BC 专家视角有 50% 概率从这些相关视角中抽取，其余从全体原训练视角抽取。专项样本额外监督一个围绕真实火球动作的约 32 秒窗口，同时保留正常开局、后续上下文、可信 WAIT 和完整循环状态预热。50% 不是全部训练数据或梯度的一半，也不是每个双样本小批次必然一专一通用。

建议先完成“只改间隔”的短实验，再从选定检查点试专项。下面的例子从原实验最新检查点建立单独专项分支，包含新间隔：

```bash
source_checkpoint="$(rg --files --no-ignore runs/spar16-human-v1 -g 'update-*.pt' | sort | tail -n 1)"
test -f "$source_checkpoint" || exit 1
.venv/bin/python -m native_runner.training.v4.prepare_expert_experiment \
  --checkpoint "$source_checkpoint" \
  --trainer-options runs/spar16-human-v1/trainer-options.txt \
  --expert-manifest runs/human-experts-v1/train-fireball-v1.json \
  --min-play-gap-ticks 8-12 --coefficient 0.05 --every-updates 4 \
  --output runs/fireball-trial-v1

CR_PPO_CHECKPOINT="$PWD/runs/fireball-trial-v1/baseline.pt" \
CR_PPO_RESUME_CHECKPOINT="" \
CR_PPO_EXTRA_ARGS="$(cat runs/fireball-trial-v1/human-options.txt)" \
CR_LAB_KEEP_CHECKPOINTS=100 \
bash native_runner/training/v4/lab_ppo.sh start spar16-fireball-v1 100
```

此处系数 0.05 是短实验起点，不保证更新影响只有一半。日志 `ppo_expert_bc_complete` 新增 `bc_fireball_focus_records`、`bc_fireball_moving_frames`，能确认专项样本实际进入更新。100 次更新后，比较源基准、只改间隔、专项模型在相同延迟下的表现；观察落空、正常防守和拉扯，不用 BC 损失或原始训练胜率代替独立比较。

这是专项采样入口，尚未实现受控局面的命中率基准和加农炮拉扯专项。人类录像的原始动作时机与训练的 18–24 tick 执行延迟仍可能存在差异；通过实战录制先测真实延迟。加农炮对骷髅气球还涉及建筑吸引范围、目标锁定和落点，不能用统一“提前半秒”替代。

## 3. MuMu 1440×2560 录制与调试

1440×2560 使用现成测量布局。第一次先核对布局；在手动友谊战里执行（不要同时跑第二个接管程序）：

```powershell
.\.venv\Scripts\python.exe -m native_runner.live_play shot
```

检查 `runs\live\shot-*-layout.png`：网格、四张手牌中心、左右塔血量与画面对应。英雄在场时再核对技能按钮。自行校准的 `runs\live\layout.json` 带 `calibrated: true` 时优先使用。分辨率或窗口比例改变后重新核对；世界坐标和模型输入独立于屏幕分辨率，点击映射依赖布局。

等正在打的这一局结束，Ctrl+C 退出旧接管进程，再用原来的模型路径启动：

```powershell
.\.venv\Scripts\python.exe -u -m native_runner.live_play run --checkpoint "你的当前检查点.pt" --gate-temperature 0.2 --min-play-gap-ticks 8-12 --record
```

`--account-id` 沿用已保存设置；需要时显式提供。第一次调试可增加 `--games 1 --record-screenshots`，保存开局和结束的截图，便于核对分辨率；ADB 截图会增加启动时的耗时，通常不需要长期打开。`--dry-run --record` 可先验证采集和拟点击坐标，档案会明确标记没有真实点击。

原有启动脚本也能转发这些参数：` .\native_runner\launch_live_my.ps1 --record --min-play-gap-ticks 8-12`。该脚本会重启游戏，并使用脚本里指定的权重，因此先核对权重路径、等当前对局结束后再用。

每局保存到终端打印的 `runs\live\recordings\match-...`：

```text
manifest.json          # 模型哈希、代码哈希、座位、分辨率、布局、温度、间隔、结束状态
events.jsonl           # 决策、实际点击坐标、跳过/丢失、环境重建、执行事件、终局
index.jsonl            # 原始观测/策略局面文件索引、tick、时钟、大小、SHA256
captures/*.json.gz     # 每一次真实请求得到的原始 atomic 响应，包括校验失败的响应
policy_frames/*.json.gz # 每次决策前实际使用的 FAIR 局面及终局，关联原始观测与循环状态段
start.png / end.png    # 可选调试截图；与 atomic 观测不是同步拍摄的训练帧
```

所有实际采集到的局面、输赢、错误、WAIT 都保留，不因胜率或训练筛选删除。采集从进程挂上本局开始；频率受推理、probe 和 ADB 开销影响，无法补回启动前或两次观测之间未采集的状态。它是结构化数据录制，不是屏幕视频。压缩写盘在有界后台队列中完成，数据提交耗时逐条记录；磁盘过慢会反压而非丢弃，写入错误会停止并明确报错。

点击回执里的执行时间是估计值，不能当作已生效证据。录制会另外关联真实 native card_play/spawn 事件计算决策到部署延迟；关联不唯一或缺失时标记未解决，不伪造成功。动作拒绝、跳过、疑似丢失、长观测间隔和循环状态重建都需在后续训练转换时审核。

录完运行检查命令，替换具体目录：

```powershell
.\.venv\Scripts\python.exe -m native_runner.live_recording "runs\live\recordings\match-实际目录" --verify
```

检查文件哈希、保存数量、最大观测间隔、跳过/疑似丢失次数、环境重建次数和实际部署延迟。若发现点歪或漏点，保留整局目录和 `shot` 的原图/叠加图/采集 JSON，先修校准或时序，再决定该局哪些片段适合学习。训练素材的使用标记与原始文件分开，失败片段同样有诊断和专项价值。

实战档案目前标记 `training_ready: false`：结构化数据已保存，尚未实现将它们自动审计并转换成 `.cril.zst` 的入口。不能把档案路径直接传给 `--expert-bc-manifest`，也不能把 AI 自己的所有动作直接标为专家。后续要根据实际成功动作、人工纠正与结果选择用途；原始档案始终保留。
