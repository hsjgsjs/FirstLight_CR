# PPO“一学就退化”排查现状（2026-10-17 更新）

本文是排查的交接记录：已确认什么、跑过什么、结果说明什么、下一步怎么决定。详细数据与推理见[实力提升计划](strength_plan_zh.md)第 8 节，各选项用法见[实验室训练](lab_training_zh.md)第 4、6 节。

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

## 3. 剩下的嫌疑（按可能性排序，10-17 按梯度信噪比的结果重排）

1. **噪声：每次更新的数据太少**（首要）
   - 规模：原项目 1,528 局并发，每次更新约 340 局；本实验室每次更新约 8 局结束。
   - 实测：`snr` 约 0.05，`grad cos` 接近 0，见第 4 节。
   - 对策：加长分段、攒段，或缩小步长，见第 5 节。
2. **没有东西稳住落点分布**
   - 熵系数固定为 0，落点温度 1.0。
   - 吻合的迹象：漂移分析中落点头漂移最大；噪声可能主要集中在落点头。
3. **价值裁剪过紧，价值头跟不上**（降为次要）
   - 机制：价值裁剪等于策略裁剪 0.1，回报在 ±1 左右；一个分段 (γλ)^160 ≈ 0.04，优势值主要由价值头决定。
   - 降级原因：价值头一致地偏会让 `grad cos` 明显为正，实测接近 0。
   - 对策：`--value-clip-range 0` 已可用。

## 4. 10-16 夜的实验与结果

三组都从 recent u200 出发，设置与 `spar16-anchor-a` 相同（人类 BC 每次更新、系数 0.2，`running` 归一化），只改列出的一项：

| 运行 | 改动 | 结果 |
|---|---|---|
| `spar16-snr-a` | `--gradient-noise-every 2` | 400 次更新，胜率 32% → 约 8%；`grad cos` 全程在 ±0.09 内（均值约 0.01），`snr` 约 0.0–0.16 |
| `spar16-shape-a`（早停） | `--shaping-beta 0.2` | 样本太少，不下结论 |
| `spar16-hog-sanity` | `--hog-deploy-reward 0.3 --hog-deploy-reward-episode-cap 1.5` | 按全部座位平均，每局野猪 0.2–0.8 次，400 次更新没有上升趋势；待按卡组复核（见下） |

**梯度信噪比的含义**

- `snr` 估计的是“真实梯度的平方范数 / 一次更新里噪声的平方范数”。0.05 意味着更新方向与真实梯度的夹角余弦只有约 √0.05 ≈ 0.22，更新的绝大部分是噪声。
- 系统偏差（例如价值头一致地偏）会让两半数据的梯度指向同一错误方向，`grad cos` 应明显为正。实测接近 0，所以不是偏差主导，价值裁剪降为次要嫌疑。
- 结论：**每次更新的数据太少，步子又是整步，参数在做随机游走，离开起点的好位置**。这与旧设置（学习率 1e-5、每批 1 轮，步子约小 12 倍）胜率不动、新设置胜率下滑都吻合，也与原项目（每次更新的数据约多 40 倍，信噪比约 2）能练出来吻合。

**野猪测试要按卡组复核**：学习方的卡组从整个卡组池抽取，大多数卡组没有野猪，按全部座位平均会把信号稀释掉。下面的脚本只统计带野猪的卡组，用每局实际下野猪次数（不受奖励上限截断），并以冻结对手的带野猪座位作对照。两个运行都跑，`spar16-snr-a` 没有野猪奖励，作为对照组：

```bash
python - spar16-hog-sanity spar16-snr-a <<'PY'
import json, shlex, sys
from pathlib import Path
HOG = 26000021
for name in sys.argv[1:]:
    run = Path("runs", name)
    options = shlex.split((run / "trainer-options.txt").read_text())
    pool = json.loads(Path(options[options.index("--deck-pool") + 1]).read_text())
    hog_decks = {d["deck_id"] for d in pool["decks"] if HOG in d["card_ids"]}
    stage = sorted(run.glob("train-*"))[-1]
    rows = [json.loads(l) for l in (stage / "matches.jsonl").read_text().splitlines() if l.strip()]
    print(f"== {name} {stage.name}  卡组池 {len(pool['decks'])} 个，含野猪 {len(hog_decks)} 个")
    print("   updates   学习方带野猪座位  每局野猪  零野猪比例   对手带野猪座位  对手每局野猪")
    for low in range(1, max(r["update"] for r in rows) + 1, 50):
        mine, theirs = [], []
        for r in rows:
            if not low <= r["update"] < low + 50:
                continue
            decks = (r["deck0_id"], r["deck1_id"])
            for owner in (0, 1):
                if decks[owner] in hog_decks:
                    (mine if owner in r["current_owners"] else theirs).append(r["hog_deployments"][owner])
        def show(xs):
            return f"{len(xs):5d}  {sum(xs)/len(xs):8.2f}" if xs else f"{0:5d}  {'-':>8}"
        zero = f"{sum(x == 0 for x in mine)/len(mine):8.0%}" if mine else f"{'-':>8}"
        print(f"   {low:3d}-{low+49:<3d}   {show(mine)}  {zero}          {show(theirs)}")
PY
```

| 结果 | 含义 |
|---|---|
| hog-sanity 学习方每局野猪明显上升，snr-a 不升 | 流水线能学会密集奖励；胜负信号只是太稀、太吵 |
| 两组都不升 | 连密集奖励也被噪声淹没，或者奖励没正确归到动作上，需要再查 |

## 5. 下一步：加大每次更新的数据量

不用改代码的办法是加长分段：`--rollout-segment-seconds 160`，每次更新的决策数是现在的 4 倍，大部分对局在两段内结束，同相位问题也随之变轻。代价是每次更新的墙钟时间约为 4 倍。另一组只把学习率降到 1e-5，看下滑是否随步长平方减慢（噪声）还是线性减慢（偏差）。

两组都与 snr-a 相同，起点都是 snr-a 的 u200。`spar16-seg160` 沿用 snr-a 的实例，`spar16-lr1e5` 沿用 hog-sanity 的实例：

```bash
git pull
bash native_runner/training/v4/lab_ppo.sh stop spar16-snr-a
bash native_runner/training/v4/lab_ppo.sh stop spar16-hog-sanity
for name in spar16-seg160 spar16-lr1e5; do
  mkdir -p runs/$name/logs
  cp runs/spar16-snr-a/trainer-options.txt runs/$name/
done
cp runs/spar16-snr-a/launch-env runs/spar16-seg160/
grep -v '^CR_PPO_CHECKPOINT=' runs/spar16-hog-sanity/launch-env > runs/spar16-lr1e5/launch-env
grep '^CR_PPO_CHECKPOINT=' runs/spar16-snr-a/launch-env >> runs/spar16-lr1e5/launch-env
bash native_runner/training/v4/lab_ppo.sh options spar16-seg160 --rollout-segment-seconds 160
bash native_runner/training/v4/lab_ppo.sh options spar16-lr1e5 --learning-rate 1e-5
for name in spar16-seg160 spar16-lr1e5; do
  CR_LAB_WARM_START_OPTIMIZER=1 bash native_runner/training/v4/lab_ppo.sh resume $name 400
  bash native_runner/training/v4/lab_ppo.sh watch $name
done
```

如果 seg160 显存不够，改成 120。

| 观察 | 含义 | 下一步 |
|---|---|---|
| seg160 的 `snr` 升到约 0.15–0.2，胜率按对局数算下滑明显变慢 | 噪声随数据量下降，方向对 | 加一个“攒几段再更新”的选项，继续加大每次更新的数据量 |
| seg160 的 `snr` 不升 | 噪声不随样本数下降，来源在别处（例如落点头） | 按头分别测信噪比 |
| lr1e5 的下滑约为 snr-a 的 1/9 | 噪声主导 | 小步长长期跑可以不退化，但学得慢 |
| lr1e5 的下滑约为 snr-a 的 1/3 | 有系统偏差 | 回到价值裁剪（`--value-clip-range 0`） |

## 6. 本轮新增的工具

| 工具 | 用途 |
|---|---|
| `lab_ppo.sh watch / unwatch / trend` | 训练意外退出后自动拉起；按阶段、按若干次更新查看胜率和学习指标；`trend` 含 `grad cos`、`snr` 列 |
| `eval_curve.py` | 跟随一个运行，定期用固定条件对 General 460 评测，生成训练曲线 |
| `policy_drift.py` | 在人类录像状态上，按头逐项测量策略相对基准检查点的漂移 |
| `--gradient-noise-every N` | 测梯度信噪比；只测量，不改变训练结果 |
| `--value-clip-range X` | 调整价值裁剪；`0` 关闭 |
| `--no-score-model-delay-offset` | 不把延迟头计入 PPO 比率（备用） |
