# PPO“一学就退化”排查现状（2026-10-23 更新）

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
| 自博弈 | 噪声主导时不是主因；攒批后要重测 | 关掉自博弈（100% 固定对手）照样下滑，但那时每次更新约 8 局，什么信号都被噪声盖住；见第 5 节「下一步」 |
| 新输入 | 不是主因 | 冻结新输入的对照组也下滑 |
| 动作被拒或被改 | 不是 | 被拒出牌全程为 0；以“牌离开手牌”判定执行，落点在决策时已被限制为合法格 |
| 优势值链路接错 | 没有发现 bug | 逐项核对并用记录协议测试复现：GAE 的终局与截断、分段处的价值自举（与下一段首步逐位相同）、局中结束的线终值为 0、不出手期间奖励累加到上一步、自博弈两座位各用自己的奖励、分段循环状态（第一轮比率恰为 1）、强制开局只作用于冻结对手 |
| 锚太松 | 只是放大因素 | 人类 BC 每次更新、系数 0.2 只把下滑放慢：41/45/40/39/33/31%，101–104 次降到 23% |

## 3. 剩下的嫌疑（按可能性排序，10-17 按梯度信噪比的结果重排；10-19 晚补充）

0. **10-19 晚的现状**：攒批（acc1，64 局/更新）已经消除了随机漂移，前 40 次更新在同样的对局上比起点高 10–15 个点；但策略随后稳定地走向「少出牌、多等待」并走过头，胜率回落。嫌疑是自博弈或钻固定对手强制先出牌的空子，见第 5 节末尾。
1. **噪声：每次更新的数据太少**（已确认，攒批可解决）
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

snr-a 从 `spar16-recent-v1` 第 200 次更新出发（已按 `config.json` 核对；其余组以各自 `config.json` 的 `source_checkpoint` 为准，见第 5 节），三组设置与 `spar16-anchor-a` 相同（人类 BC 每次更新、系数 0.2，`running` 归一化），只改列出的一项：

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
import json, sys
from pathlib import Path
HOG = 26000021
for name in sys.argv[1:]:
    stage = sorted(Path("runs", name).glob("train-*"))[-1]
    config = json.loads((stage / "config.json").read_text())
    pool = json.loads(Path(config["deck_pool"]).read_text())
    hog_decks = {d["deck_id"] for d in pool["decks"] if HOG in d["card_ids"]}
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

## 5. 10-19：seg160 / lr1e5 作废，改做攒批的剂量实验

### 起点弄错了

`spar16-seg160`（`--rollout-segment-seconds 160`）和 `spar16-lr1e5`（`--learning-rate 1e-5`）本想与 snr-a 同起点对照，实际从 General 460 出发，而 snr-a 等各组的起点是 `spar16-recent-v1` 第 200 次更新。两者不能比较，结论作废：

- General 460 在没人先出牌时会一直等到约 120 秒（双倍圣水）才开局，这是已知行为。新 run 的自博弈和不强制开局的固定对手局因此僵持到 120 秒，学习方第一张牌约 120 秒、每局溢出约 41 圣水；snr-a 同类对局约 10 秒、溢出约 0。
- 原因：`supervisor.log` 的 `starting from` 只打印启动时的 `CR_PPO_CHECKPOINT` 环境变量，launch-env 被后来的操作改写后就不再反映当初的起点。训练器实际加载的文件以阶段目录的 `config.json` 为准：

```bash
python - <<'PY2'
import json
from pathlib import Path
for config in sorted(Path("runs").glob("spar16-*/train-*/config.json")):
    c = json.loads(config.read_text())
    print(config.parts[1], config.parts[2], c.get("initial_update_step"), c.get("source_checkpoint"))
PY2
```

- seg160 也测不了原来的问题：加长分段不增加同时进行的独立对局线（每次更新仍约 90 条），只让每条线多走几步，所以 `snr` 不升是预料之中的。

### 为什么继续查数据量

不论起点、惩罚、自博弈怎么改都会下滑，人类 BC 只能放慢；所有运行的 `grad cos`、`snr` 都接近 0。与此吻合的解释是：每次更新的方向几乎全是噪声，Adam 又把每一步缩放到约等于学习率的固定大小，参数从起点的好位置随机游走，平均只会变差。每次更新的独立对局数是原仓库的约 1/40。

### 攒批：`--accumulate-batches N`

`--accumulate-batches N` 在策略不变的情况下连收 N 个完整批次（所有对局线各打完一局，默认 8 个引擎 × 8 条线 = 64 局），再用全部数据做一次更新：

- 每段数据仍按原样计算优势（段尾用价值自举），只是拼成更多对局线一起更新；批次之间换对手、换卡组照常进行。
- 每次更新都落在批次结束处，所以每次更新都存一个可严格续训的检查点。
- 同时要把 `--lane-minibatch-size` 调大（4096），每轮才是一步大步，而不是几步小步。
- GPU 显存由一次前向的对局线数决定，即 `min(更新的对局线数, --lane-microbatch-size)`。不攒批时一次更新只有约 90 条线，碰不到微批上限 256；攒批后线数远超 256，每次前向按 256 条算，显存约为原来的 3 倍，22 GB 的卡会爆。所以攒批时要把 `CR_PPO_LANE_MICROBATCH_SIZE` 降到 96（约等于原来一次前向的线数）。梯度在所有微批上累加后才走一步，所以这只影响速度，不影响结果。
- 代价是内存：一次更新的数据全部留在内存里，拼接时短暂占两份。N = 4 约为现在单次更新的 20 倍，第一次更新完成前看 `free -g`，不够就降到 2。
- 信噪比的两半各取每一段的一半对局线，所以两半混有同样的早段和晚段，`snr` 和之前可以直接比较。

### 实验：acc1 与 acc4 并行

两组起点都与 snr-a 相同（命令从 snr-a 的 `config.json` 读出），选项与 snr-a 相同，只加攒批。acc1 每次更新约 64 局、约 4 分钟；acc4 约 256 局、约 15 分钟，接近原仓库的规模。分别沿用 seg160、lr1e5 的实例：

```bash
git pull
for name in spar16-seg160 spar16-lr1e5; do
  bash native_runner/training/v4/lab_ppo.sh unwatch $name
  bash native_runner/training/v4/lab_ppo.sh stop $name
done
START=$(python -c "import json,glob;print(json.load(open(sorted(glob.glob('runs/spar16-snr-a/train-*/config.json'))[0]))['source_checkpoint'])")
echo "起点: $START"
for pair in "spar16-acc1 1 spar16-seg160" "spar16-acc4 4 spar16-lr1e5"; do
  set -- $pair
  mkdir -p runs/$1/logs
  cp runs/spar16-snr-a/trainer-options.txt runs/$1/
  grep -v '^CR_PPO_CHECKPOINT=' runs/$3/launch-env > runs/$1/launch-env
  echo "CR_PPO_CHECKPOINT=$START" >> runs/$1/launch-env
  bash native_runner/training/v4/lab_ppo.sh options $1 --accumulate-batches $2 --lane-minibatch-size 4096 --gradient-noise-every 1
  CR_LAB_WARM_START_OPTIMIZER=1 bash native_runner/training/v4/lab_ppo.sh resume $1 400
  bash native_runner/training/v4/lab_ppo.sh watch $1
done
```

#### 第一次启动失败的两个原因（已修复）

1. **显存不够**：两组都在第一次更新时报 `CUDA out of memory`，原因见上文微批一条。
2. **watch 重启换了起点**：`ensure` 读过 `.env` 后再调用 `resume`，`resume` 把 `.env` 里的 `CR_PPO_CHECKPOINT`（General 460）当成命令行显式给的值，盖过了 `launch-env`，还把它写回了 `launch-env`。只在还没有检查点时出现（有检查点时从检查点续训）。`ensure` 现在会先丢掉从 `.env` 读到的值。

重新启动（训练选项不变；命令行给的两个值优先，并会写进 `launch-env`，之后 watch 重启也用它们）：

```bash
git pull
START=$(python -c "import json,glob;print(json.load(open(sorted(glob.glob('runs/spar16-snr-a/train-*/config.json'))[0]))['source_checkpoint'])")
echo "起点: $START"
for name in spar16-acc1 spar16-acc4; do
  bash native_runner/training/v4/lab_ppo.sh unwatch $name
  bash native_runner/training/v4/lab_ppo.sh stop $name
  CR_PPO_CHECKPOINT=$START CR_PPO_LANE_MICROBATCH_SIZE=96 CR_LAB_WARM_START_OPTIMIZER=1 \
    bash native_runner/training/v4/lab_ppo.sh resume $name 400
  bash native_runner/training/v4/lab_ppo.sh watch $name
done
```

启动后确认 `config.json` 的 `source_checkpoint` 与 snr-a 相同。每次更新要等整批打完，`trend` 用小一点的块（例如 `trend spar16-acc4 4`）。参照：snr-a 前 100 次更新胜率 30% → 22%。

| 观察 | 含义 | 下一步 |
|---|---|---|
| `snr` 随 N 上升，下滑随之变慢，acc4 持平或上升 | 数据量不够就是原因 | 在速度与效果之间选 N |
| `snr` 上升，胜率仍按 snr-a 的速度下滑 | 方向确定但错误，问题在奖励或优势 | 对照：在对局之间打乱胜负再训练，看是否同样下滑 |
| `snr` 不随 N 上升 | 噪声来自对局数以外 | 按头分别测信噪比 |

#### 中期结果（acc1 第 43 次、acc4 第 10 次更新）

对局编号决定卡组和对手（`matchmaking.py` 按序号抽取），所以三组同一编号的对局是同一卡组、同一对手，可以配对比较。对固定对手的胜率，每格 256 个编号（约 150 局固定对手）：

| 对局编号 | snr-a（约 8 局/更新） | acc1（64 局/更新） | acc4（256 局/更新） |
|---|---|---|---|
| 0–255 | 31% | 40% | 33% |
| 512–767 | 20% | 46% | 21% |
| 1024–1279 | 12% | 42% | 32% |
| 1536–1791 | 9% | 40% | 30% |
| 2048–2303 | 6% | 49% | 34% |
| 2304–2559 | 6% | 46% | 38% |

- 同样的对局上，snr-a 一路降到 6%，acc1 和 acc4 都没有降。按对局数算，攒批的两组都已远远越过 snr-a 开始下滑的位置，所以下滑不是对局、对手或奖励本身造成的。
- 起点约 31%–33%（acc4 前 256 局全由起点打出）。acc1 稳定在 40%–49%，看起来在变强；需要训练结束后用固定条件评测确认。
- 按更新次数还没有结论：snr-a 在第 32–64 次更新时还是 28%，到第 70–96 次才降到 20%。acc1 要过了第 100 次更新仍持平，才能排除「每一步本身有问题」。
- acc4 每次更新走的步长与 acc1 相同，同样的对局只走了 1/4 的步数，所以在同样的对局上进步得比 acc1 慢。批量加大时学习率一般也要相应调大。

#### 10-19 晚：acc1 先变强、再学成被动打法

acc1 到第 44–51 次更新时，对固定对手的胜率从 46%–50% 回落到 40%、35%。`trend` 里同时出现：每局出牌减少、浪费圣水增多、首牌变晚，`snr` 升到 0.41–0.56（此前都在 0 附近）。用 `tools/ppo_paired_style.py` 在同一编号（同卡组、同对手）的对固定对手对局上比较打法：

| 对局编号 | acc1 胜率 | 出牌 | 浪费 | 首牌（秒） | 由第几次更新打的 | acc4 胜率 | 出牌 | 浪费 | 首牌（秒） |
|---|---|---|---|---|---|---|---|---|---|
| 0–255 | 40% | 34.4 | 0.3 | 9.5 | u1–4 | 33% | 35.6 | 0.2 | 9.2 |
| 768–1023 | 42% | 30.8 | 1.3 | 10.5 | u13–16 | 29% | 37.1 | 0.2 | 9.5 |
| 1280–1535 | 50% | 29.7 | 1.8 | 11.0 | u20–24 | 38% | 35.9 | 0.1 | 9.5 |
| 2048–2303 | 49% | 33.1 | 1.5 | 11.0 | u32–36 | 34% | 35.9 | 0.1 | 9.5 |
| 2816–3071 | 40% | 31.4 | 5.6 | 12.2 | u44–48 | — | — | — | — |
| 3072–3327 | 35% | 25.8 | 8.3 | 12.5 | u48–51 | — | — | — | — |

配对差值（acc4 − acc1，同一批对局）：acc1 每局少出 3–7 张牌、多浪费 1–3 圣水、首牌晚 1–3.5 秒，从第一格起就一致。

- **acc1 从一开始就朝「被动」走**：首牌 9.5 → 10.2 → 10.8 → 11.6 → 12.2 → 12.5 秒，浪费圣水一路上升。这是有固定方向的变化，不是噪声，与 `snr` 转正吻合。
- **被动起初是赢的**：同样的对局上 acc1 比 acc4（基本还是起点）高 10–15 个点。可能的原因：80% 的固定对手局里，对手被强制在 6–20 秒内先出牌，等对方先出再防守反击占圣水便宜。
- **之后走过头**：出牌降到约 26、浪费到 8 以上，胜率随之回落。
- **snr-a 不同**：它在同样的对局上也比 acc4 开局晚 2–8 秒、浪费更多，但忽高忽低，没有稳定趋势，`snr` 全程约 0。snr-a 的崩溃主要是噪声造成的随机漂移。
- **acc4 打法几乎没动**（出牌 35–39、浪费 0.1–0.4、首牌约 9.5 秒），但只更新了 11 次。acc1 在第 9–12 次更新时已经是浪费 1.3、首牌 10.2，所以 acc4 每次更新漂得比 acc1 少。

结论：每次更新的数据太少，会让参数随机漂移（snr-a），攒批能解决这一层。数据量够了以后，还露出第二个问题：PPO 稳定地把策略推向「少出牌、多等待」，先有收益，然后走过头。

#### 下一步（10-19 晚开始）

1. **acc1 换成 acc1-noself**：设置与 acc1 完全相同，只把自博弈关掉（`--self-play-percent 0 --fixed-opponent-percent 100`），起点相同。理由：两个相同的模型对打时，谁先出牌谁吃亏，自博弈会教出「不先出、只防守」的打法；文档上文也记过，自博弈局会僵持到 120 秒。第 2 节「关掉自博弈照样下滑」是在每次约 8 局、噪声主导时测的，不能排除自博弈在攒批条件下的作用。

   ```bash
   bash native_runner/training/v4/lab_ppo.sh unwatch spar16-acc1
   bash native_runner/training/v4/lab_ppo.sh stop spar16-acc1 15
   mkdir -p runs/spar16-acc1-noself/logs
   cp runs/spar16-acc1/trainer-options.txt runs/spar16-acc1/launch-env runs/spar16-acc1-noself/
   bash native_runner/training/v4/lab_ppo.sh options spar16-acc1-noself --self-play-percent 0 --fixed-opponent-percent 100
   CR_LAB_CHECKPOINT_BUCKET=10 CR_LAB_WARM_START_OPTIMIZER=1 \
     bash native_runner/training/v4/lab_ppo.sh resume spar16-acc1-noself 400
   bash native_runner/training/v4/lab_ppo.sh watch spar16-acc1-noself
   ```

   自博弈比例改变后，同一编号抽到的对局不同，不能逐局配对；按更新次数比较打法列（`tools/ppo_paired_style.py` 上半部分，或 `trend spar16-acc1-noself 4`）。

   | 结果 | 含义 | 下一步 |
   |---|---|---|
   | 第 20–40 次更新时，首牌仍约 9.5 秒、浪费低于 1，胜率高于起点 | 被动来自自博弈 | 自博弈加开局拖延惩罚，或用历史快照代替当前模型对打 |
   | 照样变被动 | 在钻固定对手强制先出牌的空子 | 提高不强制开局的固定对手比例 |

2. **acc4 继续跑**：看它多更新几十次后是否也变被动。如果不变，说明「被动」也和每次更新的局数有关。
3. **保住 acc1 最好的版本**：第 20–40 次更新左右（同样对局上 46%–50%）。旧版训练器每存一个新检查点就删掉旧的（PR #86 已修，见 `lab_training_zh.md`），只能从手动备份里找。引擎空出来后，用固定条件让它和起点对打，确认是真的变强。
4. **备选诊断**：`policy_drift.py --device cpu` 比较 acc1 最新检查点和起点，看是否主要是门控在动、出牌概率在下降。

## 6. 10-20 至 10-21：训练稳住了，但停在 52%

### noself：被动来自自博弈

acc1-noself 与 acc1 设置相同，只关掉自博弈（全部打固定对手）。到第 77 次更新仍不被动：出牌 38–43、浪费约 0、首牌约 10 秒，而 acc1 在第 44–51 次已经是出牌 26、浪费 8。

- 被动打法主要来自自博弈：两个相同的模型对打，谁先出牌谁吃亏。
- 固定对手被强制开局不是被钻的空子，否则 noself 也会变被动。
- 每次 64 局、没有自博弈时噪声主导（`snr` 约 0），胜率在 9%–48% 之间乱跳，看不出方向。

### acc4：256 局一次更新，先涨后平

| acc4 的更新 | 对固定对手胜率 | 浪费 | 首牌 |
|---|---|---|---|
| 1–8 | 28%–33% | 0.0 | 9–10 秒 |
| 25–37 | 46%–47% | 0.1–0.2 | 10–11 秒 |
| 45–52 | 51%–53% | 0.7–1.0 | 12 秒 |

### acc4-fast：学习率翻倍加更新与采集重叠，失败

从 acc4 热启动，`--learning-rate` 翻倍，加 `--overlap-collection`。第一次启动漏了 `CR_LAB_WARM_START_OPTIMIZER=1`，Adam 从零开始。从第 2 次更新起，每次更新开始时 KL 已超过 0.01，只走 1–2 步就停，`snr` 10–100（不是学习信号），胜率停在 41%，浪费升到 2.4。

续训会恢复优化器状态，跑过十几次更新后 Adam 早已不冷，现象依旧，所以主因不是冷启动，而是：重叠时数据晚一次更新，再加上人类数据那一步的移动，策略一开始就偏离打出数据的策略；学习率翻倍又把偏离放大。`--overlap-collection` 配原学习率还没有单独验证过。

### acc4-2g：双卡，与 acc4 等效、快约 2.5 倍

从 acc4 最新检查点热启动（`runs/keep/acc4-final.pt`，带优化器状态）：16 个引擎、两张卡，`--accumulate-batches 2`（每次更新仍是 256 局），`--expert-bc-records-per-rank 1`（两张卡合计仍是每次 2 条人类录像）。两张卡的梯度、优势值归一化、KL 提前停止都是全局合并的，所以与单卡 256 局等效。每次更新约 13 分钟（acc4 与另一组共用机器时约 33 分钟），引擎和 worker 的 CPU 约 87%，已接近 10 核 20 线程的上限；剩下的空闲是优化阶段（约 35%），双卡时 `--overlap-collection` 不可用。

第 1–38 次更新（相当于 acc4 的第 50–90 次）：

| 段 | 对固定对手 | 胜率 |
|---|---|---|
| 第 1–19 次 | 2,921 局 | 52.4% |
| 第 20–40 次 | 2,918 局 | 51.4% |

- **不劣化**：这是第一次跑约 90 次更新还没退化。
- **打法回来了**：浪费 0.4–0.6（首牌后的中位数 0.3），首牌 11 秒，出牌约 35，开局拖延惩罚从未触发。acc4 末尾的浪费上升只是一段波动。
- **也不进步**：`kl` 约 0.0002，`clip` 约 0.01，`snr` 0.02–0.3。

### 人类数据那一步和 PPO 拉动一样大

训练日志里 `bc_step_probe_kl` 是人类数据那一步在本次对局数据上让策略移动的 KL，可以直接和 PPO 的 `approx_kl` 比：

| 阶段 | BC/PPO | 胜率 |
|---|---|---|
| acc4 第 1–40 次 | 0.74–0.89 | 28% → 47% |
| acc4 第 41–48 次与 acc4-2g 全程 | 0.76–1.45，平均约 1.1 | 约 52% |

PPO 每次的移动里只有约 1/5 是稳定方向（`snr` 约 0.05），人类数据那一步每次都朝同一方向拉，多次累积后稳定的部分它占上风：不劣化，也不进步。

另外，人类数据那一步与 PPO 共用同一个 AdamW。Adam 按梯度大小缩放步长，所以系数 0.2 不等于「五分之一的力度」；而且它的方向会通过动量（0.9）带进下一次更新的前几步 PPO，`approx_kl` 里也含有一部分人类数据的拉力。比较两步的梯度大小：metrics.jsonl 的 `gradient_norm`（PPO 各步平均）与 `bc_gradient_norm`。

### 下一步：降低人类数据权重，加历史快照对手

新参数 `--history-opponent-percent P`（见 `lab_training_zh.md`）：P% 的对局对手是本运行自己过去的快照，固定对手保留自己的份额。以前配置了固定对手时，训练器只存快照、不让快照上场。

`spar16-bchist`：从 acc4-2g 最新检查点热启动，改四项：

- `--expert-bc-coefficient 0.05`：人类数据权重降到四分之一。
- `--self-play-percent 20 --history-opponent-percent 20`：一半自博弈换成历史快照（固定对手仍 60%）。
- `--snapshot-every 10`：约每 2 小时一个快照（每个约 48 MB，400 次更新约 2 GB）。起点检查点是第一个快照。

两项同时改，结果不能归因到单独一项，但两项都指向同一目标：PPO 推得动，同时不被自博弈带向被动。

| 观察（20–30 次更新） | 含义 |
|---|---|
| 对固定对手胜率升到 55% 以上，浪费、首牌不变 | 人类数据确实在拖后腿；快照对手挡住了被动 |
| 胜率还是约 52% | 不是这两项，下一步测熵奖励（目前写死为 0）或关掉自博弈 |
| 胜率涨，但浪费超过 2、首牌 13 秒以上 | 人类数据原本在挡被动，快照对手不够 |
| 胜率低于 45% | 停掉，回到 `runs/keep/acc4-2g-final.pt` |

`trend` 新增 `vs past (n)` 列，是对自己快照的胜率；它会随快照变强而停在 50% 上下，不代表强弱。`vs fixed` 已排除快照对局；`status` 把快照对局合成一行。

## 7. 10-23：bchist 不退化也不进步，停止

`spar16-bchist` 从 acc4-2g 热启动，跑到第 140 次更新后于 10-23 停止。

- **对固定对手胜率从第 1 次更新起就在 55% 左右**：第 1–4 次更新基本还是起点在打，已经是 54–55%；之后五段（每段约 4000 局）在 52%–56% 之间，彼此都在误差内。比 acc4-2g 的 52% 高，主要是对局组成变了（自博弈 40% → 20%、加了快照对手），不是练出来的。
- **对快照胜率约 48%，不代表变弱**：快照按 PFSP 加权抽取（`league.py`，学习者胜率越低的快照越常被抽中），新快照又和当前模型几乎一样强，这一列本来就会停在 50% 上下或略低。
- **学习信号和 acc4-2g 一样弱**：KL 约 0.0003，clip 约 0.013，`snr` 0–0.3，熵 0.21 不变。人类数据系数降到 0.05 后，每次更新的移动并没有变大。
- **打法**：浪费圣水在第 25–50 次升到 4–5，之后回落到 2–3；首牌从约 11 秒变成 12–14 秒；出牌数从 33 涨到 38–40。快照对手挡住了大部分被动倾向。
- **实测**：用户评测 u80 打 General 460 胜率 60% 以上，是目前最强的模型。这里面有多少来自 bchist 本身、有多少来自 acc4 阶段，要用同样条件评测 acc4-2g final 才能分开。
- **下一步**：循环赛（General 460、u80、u140、acc4-2g final）选起点；继续训练的收益不大，转向输入改造，见[架构增强计划](architecture_enhancement_plan_zh.md)。
- **10-23 晚的后续**：输入改造后的 `spar16-v42` 用 PPO 放大新层学习率（K=30）时，新层增长像随机游走，对固定对手胜率 57% → 54–55%，与本文第 4 节的结论一致：这里 PPO 的 `snr` 约 0.05，放大步长不能提高信噪比。改为先用人类录像离线训练新层、其余冻结，再接 PPO。当前进展和交接见[架构增强计划](architecture_enhancement_plan_zh.md)第 11 节「进展记录与交接」。

## 8. 本轮新增的工具

| 工具 | 用途 |
|---|---|
| `lab_ppo.sh watch / unwatch / trend` | 训练意外退出后自动拉起；按阶段、按若干次更新查看胜率和学习指标；`trend` 含 `grad cos`、`snr` 列 |
| `eval_curve.py` | 跟随一个运行，定期用固定条件对 General 460 评测，生成训练曲线 |
| `policy_drift.py` | 在人类录像状态上，按头逐项测量策略相对基准检查点的漂移 |
| `--gradient-noise-every N` | 测梯度信噪比；只测量，不改变训练结果 |
| `--value-clip-range X` | 调整价值裁剪；`0` 关闭 |
| `--accumulate-batches N` | 连收 N 个完整批次再更新一次，加大每次更新的数据量 |
| `tools/ppo_paired_winrate.py`、`tools/ppo_paired_style.py` | 按对局编号（同卡组、同对手）配对比较几个运行的胜率和打法（出牌、浪费圣水、首牌时间） |
| `--keep-strict-checkpoints` | 训练器保留旧检查点，交给 `lab_ppo.sh` 按 `CR_LAB_KEEP_CHECKPOINTS`、`CR_LAB_CHECKPOINT_BUCKET` 清理（脚本自动带上） |
| `--overlap-collection` | 攒批时，GPU 更新期间引擎继续打下一批，减少引擎空等；数据晚一次更新（见 `lab_training_zh.md`） |
| `--no-score-model-delay-offset` | 不把延迟头计入 PPO 比率（备用） |
| `--history-opponent-percent P` | 一部分对局对手换成本运行自己过去的快照，固定对手份额不变；`trend` 的 `vs past` 列和 `status` 的「own past snapshots」一行显示对它们的胜率 |
| `--new-parameter-lr-scale K`、`--new-parameter-lr-decay-updates N` | `zero_start` 新层的学习率放大到 K 倍，再按各层自己的训练年龄在 N 次更新内降回 1 倍，其他参数不变；`status` 末尾的新层表显示大小、年龄和倍数，见[架构增强计划](architecture_enhancement_plan_zh.md)第 11 节 |
| `pretrain_new_parameters.py` | 冻结其余参数，只用人类录像训练 `zero_start` 新层，按验证集保留最好的一步，输出可直接热启动 PPO；`--steps 0 --reset-new-parameters` 生成新层清零的对照副本，见[架构增强计划](architecture_enhancement_plan_zh.md)第 11 节 |
