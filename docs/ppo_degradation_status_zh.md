# PPO“一学就退化”排查现状（2026-10-19 更新）

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

## 6. 本轮新增的工具

| 工具 | 用途 |
|---|---|
| `lab_ppo.sh watch / unwatch / trend` | 训练意外退出后自动拉起；按阶段、按若干次更新查看胜率和学习指标；`trend` 含 `grad cos`、`snr` 列 |
| `eval_curve.py` | 跟随一个运行，定期用固定条件对 General 460 评测，生成训练曲线 |
| `policy_drift.py` | 在人类录像状态上，按头逐项测量策略相对基准检查点的漂移 |
| `--gradient-noise-every N` | 测梯度信噪比；只测量，不改变训练结果 |
| `--value-clip-range X` | 调整价值裁剪；`0` 关闭 |
| `--accumulate-batches N` | 连收 N 个完整批次再更新一次，加大每次更新的数据量 |
| `--overlap-collection` | 攒批时，GPU 更新期间引擎继续打下一批，减少引擎空等；数据晚一次更新（见 `lab_training_zh.md`） |
| `--no-score-model-delay-offset` | 不把延迟头计入 PPO 比率（备用） |
