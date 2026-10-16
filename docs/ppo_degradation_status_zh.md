# PPO“一学就退化”排查现状（2026-10-16 夜）

本文是排查的交接记录：已确认什么、正在跑什么、明早看什么、下一步怎么决定。详细数据与推理见[实力提升计划](strength_plan_zh.md)第 8 节，各选项用法见[实验室训练](lab_training_zh.md)第 4、6 节。

## 1. 问题

- 旧设置（学习率 1e-5、每批 1 轮）实际几乎不学，KL 显示为 0；200 次更新里对固定对手的胜率一直平在约 40%。
- 14:38 换成学习率 3e-5、每批 4 轮，真正开始学之后，胜率单调下滑，KL、clip、EV 却都显示正常。
- 这个胜率正是 PPO 自己的训练目标：60% 的训练对局就是打这几个固定对手。所以问题不在“锚不够”，而在每次更新本身没有带来改进。
- 实验室之后练出的模型（104、1208、BC 200）在循环赛里都没有超过 General 460。也就是说，PPO 在本实验室的规模下从未证明过能变强。

## 2. 已排除

| 怀疑 | 结论 | 依据 |
|---|---|---|
| 更新方向写反 | 不是 | `test_training_v4_ppo_direction.py`：门控、卡牌都朝正优势方向移动 |
| 延迟头噪声 | 不是主因 | `policy_drift.py`：延迟头 KL 很小，漂移集中在落点头 |
| 自博弈 | 不是主因 | 关掉自博弈（100% 固定对手）照样下滑 |
| 新输入 | 不是主因 | 冻结新输入的对照组也下滑 |
| 动作被拒或被改 | 不是 | 被拒出牌全程为 0；以“牌离开手牌”判定执行，落点在决策时已被限制为合法格 |
| 优势值链路接错 | 没有发现 bug | 逐项核对并用记录协议测试复现：GAE 的终局与截断、分段处的价值自举（与下一段首步逐位相同）、局中结束的线终值为 0、不出手期间奖励累加到上一步、自博弈两座位各用自己的奖励、分段循环状态（第一轮比率恰为 1）、强制开局只作用于冻结对手 |
| 锚太松 | 只是放大因素 | 人类 BC 每次更新、系数 0.2 只把下滑放慢：41/45/40/39/33/31%，101–104 次降到 23% |

## 3. 剩下的嫌疑（按可能性排序）

1. **价值裁剪过紧，价值头跟不上**
   - 机制：价值裁剪等于策略裁剪 0.1，每次更新每个状态的价值最多移动约 0.1，而回报在 ±1 左右。一个分段只有 160 次决策，(γλ)^160 ≈ 0.04，大部分分段的优势值几乎全由价值头决定。
   - 放大因素：“同相位”使收尾段每 8 次更新才出现一次，价值头更难跟上。
   - 后果：价值头落后时，优势值会一致地偏，策略追着价值头的错误走。
   - 吻合的迹象：后期收尾段 EV 跌到 −0.41、−0.12。
   - 对策：已新增 `--value-clip-range`（`0` 关闭）。
2. **噪声：每次更新的数据太少**
   - 规模：原项目 1,528 局并发，每次更新约 340 局；本实验室每次更新约 8 局结束。
   - 收尾段：只剩少数加时局，仍然走满 4 轮完整步长。
   - 对策：用 `--gradient-noise-every` 测量，再决定是否加长分段、攒段，或按有效帧数缩小步长。
3. **没有东西稳住落点分布**
   - 熵系数固定为 0，落点温度 1.0。
   - 吻合的迹象：漂移分析中落点头漂移最大。

## 4. 正在跑的实验

三组都从 recent u200 出发，设置与 `spar16-anchor-a` 相同（人类 BC 每次更新、系数 0.2，`running` 归一化），只改列出的一项：

| 运行 | 改动 | 回答的问题 |
|---|---|---|
| `spar16-snr-a` | `--gradient-noise-every 2` | 基准组的梯度信噪比是多少 |
| `spar16-shape-a`（已停，被野猪测试替换） | `--shaping-beta 0.2` | 塔血奖励加密有没有用（早期读数：snr 约 0，胜率 26%，样本太少） |
| `spar16-hog-sanity` | `--hog-deploy-reward 0.3 --hog-deploy-reward-episode-cap 1.5` | 流水线能否学会一个即时、密集的奖励 |

早期读数（13 次更新）：snr-a 的 `grad cos` 在 0.22 和 0.01 之间，`snr` 在 0.54 和 0.04 之间。

## 5. 明早看什么

```bash
bash native_runner/training/v4/lab_ppo.sh trend spar16-snr-a 16
bash native_runner/training/v4/lab_ppo.sh trend spar16-hog-sanity 16
```

**野猪测试**：每 8 次更新统计一次每局下了几次野猪骑士。脚本按 `hog_deploy_reward_total / 0.3` 换算，每局上限 5 次：

```bash
python - runs/spar16-hog-sanity <<'PY'
import json, sys
from pathlib import Path
stage = sorted(Path(sys.argv[1]).glob("train-*"))[-1]
rows = [json.loads(l) for l in (stage / "matches.jsonl").read_text().splitlines()]
for low in range(1, max(r["update"] for r in rows) + 1, 8):
    seats = [(r, o) for r in rows if low <= r["update"] < low + 8 for o in r["current_owners"]]
    if seats:
        hogs = sum(r.get("hog_deploy_reward_total", [0, 0])[o] for r, o in seats) / 0.3 / len(seats)
        print(f"u{low:3d}-{low+7:<3d}  每局野猪 {hogs:4.2f}  ({len(seats)} 个座位)")
PY
```

**梯度信噪比**：列出每次测量的原始读数，并按熵区分开局段和收尾段（熵高于 0.3 视为收尾段）：

```bash
python - <<'PY'
import json
from pathlib import Path
for name in ("spar16-snr-a",):
    stage = sorted(Path("runs", name).glob("train-*"))[-1]
    print("==", name, stage.name)
    for line in (stage / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "gradient_snr" in row:
            phase = "收尾" if row.get("entropy", 0) > 0.3 else "开局"
            print(f"  u{row['update']:3d} {phase}  cos {row['gradient_half_cosine']:6.3f}  snr {row['gradient_snr']:6.2f}  EV {row['value_explained_variance']:5.2f}")
PY
```

## 6. 判断与下一步

| 观察 | 含义 | 下一步 |
|---|---|---|
| 野猪次数在 20–30 次更新内明显上升 | 流水线能把奖励正确归到动作上 | 问题在价值头或信号规模 |
| 野猪次数不涨或下降 | 流水线有 bug | 回到代码查 |
| `grad cos` 明显为正，胜率仍下滑 | 方向一致地偏 | 价值裁剪是首要嫌疑，跑 vclip0 |
| `grad cos` 接近 0，`snr` 远小于 1 | 噪声为主 | 加长分段（`--rollout-segment-seconds` 120–160）、攒段或缩小步长，同时也试 vclip0 |

不论哪种情况，下一组实验都是 `spar16-vclip0`：与 snr-a 相同，只加 `--value-clip-range 0`。命令里的 `<当前空出来的运行>` 换成刚停掉的那组，例如 `spar16-hog-sanity`；新运行要用它的 `launch-env`（实例），起点仍取 snr-a 的 u200：

```bash
git pull
bash native_runner/training/v4/lab_ppo.sh stop <当前空出来的运行>
mkdir -p runs/spar16-vclip0/logs
cp runs/spar16-snr-a/trainer-options.txt runs/spar16-vclip0/
grep -v '^CR_PPO_CHECKPOINT=' runs/<当前空出来的运行>/launch-env > runs/spar16-vclip0/launch-env
grep '^CR_PPO_CHECKPOINT=' runs/spar16-snr-a/launch-env >> runs/spar16-vclip0/launch-env
bash native_runner/training/v4/lab_ppo.sh options spar16-vclip0 --value-clip-range 0
CR_LAB_WARM_START_OPTIMIZER=1 bash native_runner/training/v4/lab_ppo.sh resume spar16-vclip0 400
bash native_runner/training/v4/lab_ppo.sh watch spar16-vclip0
```

有效的标准：收尾段 EV 不再崩，胜率不再一路下滑。

## 7. 算力的看法（备查）

- **瓶颈在 CPU，不在 GPU**：模拟器需要 CPU 和 KVM，GPU 只做推理和更新。只加 GPU 不会增加每次更新的对局数。先用 `lab_ppo.sh load 运行名` 看“等引擎”的占比。
- **训练器和引擎节点可以分开部署**：它们之间走 TCP（`remote_ppo_worker_server`），但必须在同一地域，否则每步通信会拖慢整体速度。
- **支持 KVM 的机器从哪找**：公有云普通虚拟机开启嵌套虚拟化（以各家文档为准，可用抢占式实例）、按小时计费的弹性裸金属，或纯 CPU 独立服务器。
- 先用今晚的结果确定是不是规模问题，确认之后再花钱。

## 8. 本轮新增的工具

| 工具 | 用途 |
|---|---|
| `lab_ppo.sh watch / unwatch / trend` | 训练意外退出后自动拉起；按阶段、按若干次更新查看胜率和学习指标；`trend` 含 `grad cos`、`snr` 列 |
| `eval_curve.py` | 跟随一个运行，定期用固定条件对 General 460 评测，生成训练曲线 |
| `policy_drift.py` | 在人类录像状态上，按头逐项测量策略相对基准检查点的漂移 |
| `--gradient-noise-every N` | 测梯度信噪比；只测量，不改变训练结果 |
| `--value-clip-range X` | 调整价值裁剪；`0` 关闭 |
| `--no-score-model-delay-offset` | 不把延迟头计入 PPO 比率（备用） |
