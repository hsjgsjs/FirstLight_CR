# 从 daybreak-cr 借鉴的做法（2026-10-25）

[daybreak-cr](https://github.com/Jason-XII/daybreak-cr) 也基于上游 FirstLight CR，同一个游戏版本（Null's Royale 15.535.13），但用的是私有原生引擎和大算力（PPO 每小时 1.6–5 万局，我们约 1200 局）。他们的引擎和 77 万局数据都拿不到，能用的是他们跑出来的实验结论。这里只记对我们有用的部分，数字都来自他们的 `docs/TRAINING.md` 和 `docs/CONVERSION.md`。

## 1. 已经做进代码的

### 选牌和落点单独调低温度（不用训练）

- 他们的结果：选牌和落点温度 0.1、出不出牌仍按温度 1 采样，同一个模型对原来的自己胜率 68.5%；实战从 2 胜 3 负变成 14 胜 4 负 1 平。0.05 比 0.1 再好一点（55.4%），完全贪心反而差一点（48%）。
- 我们以前三者共用一个"动作温度"，训练、循环赛、实战都是 1.0；`--deterministic` 会把出不出牌也变成贪心，所以几乎不出牌。
- 现在加了一个只在出牌时生效的设置，只管选牌和落点；出不出牌（门控）、回合内延迟和是否继续出第二张都不受影响。训练从不设置它，也不存进检查点；PPO 和 BC 给已存动作打分时也不用它。
- 用法：
  - 实战：`python -m native_runner.live_play run --checkpoint CK --gate-temperature 0.2 --card-tile-temperature 0.1`
  - 循环赛：配置里加 `"card_tile_temperature": 0.1`（需要 `"sample": true`），所有模型共用。
  - 同一个模型比两种温度（建议先跑这个确认）：

    ```bash
    python -m native_runner.training.v4.evaluate_parallel --ports 28000-28007 --games 400 \
      --output runs/card-tile-0.1 --devices cuda:0 \
      --checkpoint-a CK --checkpoint-b CK --sample \
      --gate-temperature-a 0.2 --gate-temperature-b 0.2 --card-tile-temperature-a 0.1 \
      --deck-pool runs/deck-pool.json --battle-level 16 --delay-a-ticks 21 --delay-b-ticks 21
    ```

    A 方胜率明显高于 50% 再改实战默认值。0.05 和 0.25 也值得各跑一次。

### 人类录像的落地时间和塔血核对

- 他们的实测：RoyaleAPI 记录的 `data-t` 是玩家点下去的时刻，牌落地要晚 21 帧。实战友谊赛 71 次出牌都对上；300 局样本里完全复现的从 210 局升到 249 局（共 266 局）。
- 我们以前按晚 1 帧重建（`ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS = 1`），人类的兵大约早 1 秒落地，后面的局面就和真实对局不同了。标签一直在点击时刻，不受影响；受影响的是重建出来的观测。
- 我们以前只核对胜负和皇冠数。录像里本来有双方 6 座塔的终局血量，没用上。
- 现在缓存构建多了两个参数：
  - `--royaleapi-landing-ticks N`：牌在点击后第几帧落地，默认仍是 1，等下面的验证结果再决定改不改。
  - `--require-tower-fidelity`：终局塔血对不上的录像不算完整重建。不加时只记录不拒收。
- 每个分片的摘要里多了每场录像的核对结果（`replay_fidelity`），以及塔血可比和不一致的场数。
- 改 21 帧时顺带修了两处：引擎拒收一次出牌时，按点击时刻找要屏蔽的那一步标签（以前按落地时刻找，晚 21 帧就会错到后面第 4 步）；结束前最后 1 秒点下、落地前对局已经结束的出牌，不再把整场判为失败。

**验证步骤**（只用 CPU 和引擎，不用 GPU；和新层预训练可以同时跑）：

```bash
# 同一批 300 场，分别按晚 1 帧和晚 21 帧重建（端口和 resident slots 按实际引擎填）
for N in 1 21; do
  bash native_runner/training/v4/launch_cache_partition.sh /path/to/IL_Replay runs/fidelity-plus$N \
    --ports 26789 --source-count 300 --resident-slots 8 --royaleapi-landing-ticks $N
done
python -m native_runner.training.v4.compare_replay_fidelity \
  --run plus1=runs/fidelity-plus1 --run plus21=runs/fidelity-plus21 --json runs/fidelity-compare.json
```

输出每种设置下完整跑完、胜负一致、皇冠一致、塔血一致、三项全对的场数，再列出只在一边完全复现的场数。判断：

| 结果 | 下一步 |
|---|---|
| 21 帧的"三项全对"明显更多 | 用 `--royaleapi-landing-ticks 21 --require-tower-fidelity` 重建人类缓存，再跑新层预训练和 BC |
| 两者差不多，塔血都很少对上 | 时间对齐不是主要问题，先查别的差异（游戏数据版本、国王塔等级、随机种子），按 daybreak 的 `CONVERSION.md` 逐条对照 |
| 1 帧更好 | 我们的引擎接口语义和他们不同，保持 1 |

塔血不一致的原因不只是时间：录像对局发生在正式服的数据版本上，我们的引擎是固定的 15.535 数据，平衡改动也会让塔血对不上。所以塔血完全一致的比例本身可能不高，要看两种设置之间的差。

## 2. 他们试过、证明不划算的（不用再做）

| 方向 | 结果 |
|---|---|
| 把搜索的落点结果蒸馏进网络 | 搜索本身能多赢约 20 个百分点，但网络在新对局上只学到其中约 11%，标签从 200 局加到 915 局只多 1 个点 |
| 卡牌数值表（射程、血量、伤害等 28 项） | 3 万局时验证损失降 0.037，17.7 万局全量数据时没收益 |
| UPGO | 平或略差，熵下降 |
| 给落点单独用短期回报（λ 0.95） | 落点没变好 |
| 左右镜像平均 | 48% |
| 让模型多按英雄技能 | 52%，在误差内 |
| 只用 PPO 学落点 | 一个落点好坏只值约 4% 胜率，信号被噪声淹没；PPO 的提升主要在出牌时机：只换时机头 74%，只换选牌头 56%，只换落点头 51% |
| 换上拿人类对局胜负离线训练的价值网络 | 它对人类对局胜负的解释方差比 PPO 价值网络高（0.26 对 0.18），但换上后训练出的模型没有变强 |

对我们的含义：PPO 主要练出牌时机；落点主要靠人类数据的质量和输入设计。新输入（斩杀、时空关系）用人类录像离线预训练，不指望 PPO 把它们学出来。

## 3. 还没做、以后可以考虑的

- 落点头前加 1 个残差块，用 `zero_start` 从零开始：他们测到几乎不加计算量、验证损失明显下降。
- 晋级门槛：新模型对冠军打约 800 局、胜率 ≥53% 才晋级。55% 会把真实 3–5% 的进步挡掉。
- 吞吐：他们的大头来自让同一批推理请求只涉及尽量少的不同模型，加上异步服务。我们的训练器已经对每个模型用 CUDA Graph 和单独的流；真正确定的收益是我们自己计划里的"对局线打完立即开新局、相位错开"（[实力提升计划](strength_plan_zh.md)第 5 节，估计吞吐提高约三分之一）。这两项都要在服务器上实测后再改。
- 实战落地时间：他们实测从决策到落地 21–26 帧（58% 是 22 帧），我们训练用 18–24，基本一致。
