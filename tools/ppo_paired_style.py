"""对固定对手的对局，按对局编号分块比较几个运行的打法：胜率、每局出牌、浪费圣水、首牌时间。

同一编号 = 同卡组、同对手，所以下半部分的配对差值不受卡组难度影响。自博弈比例不同的运行抽签不同，只能看上半部分。
在仓库根目录运行：python3 tools/ppo_paired_style.py [运行名 ...]（默认 spar16-snr-a spar16-acc1 spar16-acc4）。
"""
import glob, json, statistics as st, sys
runs = sys.argv[1:] or ["spar16-snr-a", "spar16-acc1", "spar16-acc4"]
BLOCK = 256
data = {}
for run in runs:
    rows = {}
    for path in sorted(glob.glob(f"runs/{run}/train-*/matches.jsonl")):
        for line in open(path):
            g = json.loads(line)
            # Games against the run's own snapshots (--history-opponent-percent) are not fixed-opponent games.
            if (g.get("policy_matchup") != "current-history" or not g.get("opponent_checkpoint_id")
                    or g.get("opponent_kind") == "snapshot"):
                continue
            owner = g["current_owners"][0]
            key = (g["sequence"], g["deck0_id"], g["deck1_id"], g["opponent_checkpoint_id"], owner)
            wasted = g.get("cumulative_elixir_overflow_wasted")
            first = g["first_nonwait_action_tick"][owner]
            rows[key] = {
                "update": g["update"],
                "win": g["result"][owner] > 0,
                "plays": g["active_actions"][owner],
                "wasted": None if not wasted else wasted[owner],
                "first": None if first is None else first / 20,
            }
    data[run] = rows


def summary(part):
    if not part:
        return "-"
    wasted = [v["wasted"] for v in part if v["wasted"] is not None]
    first = [v["first"] for v in part if v["first"] is not None]
    ups = [v["update"] for v in part]
    return (f"{100 * sum(v['win'] for v in part) / len(part):3.0f}% {st.mean(v['plays'] for v in part):4.1f}"
            f" {st.mean(wasted) if wasted else float('nan'):4.1f} {st.median(first) if first else float('nan'):4.1f}"
            f" u{min(ups)}-{max(ups)}")


print("每格：胜率  每局出牌(均值)  浪费圣水(均值)  首牌秒数(中位)  由第几次更新打的")
print("编号块      " + "".join(f"{r:>34s}" for r in runs))
blocks = sorted({k[0] // BLOCK for rows in data.values() for k in rows})
for b in blocks:
    cells = [summary([v for k, v in data[r].items() if k[0] // BLOCK == b]) for r in runs]
    print(f"{b * BLOCK:5d}-{b * BLOCK + BLOCK - 1:<5d}" + "".join(f"{c:>34s}" for c in cells))

print()
print("配对差值（后者减前者），同一编号、同卡组、同对手：出牌、浪费圣水、首牌秒数")
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        a, b = runs[i], runs[j]
        common = sorted(set(data[a]) & set(data[b]))
        if not common:
            continue
        print(f"  {b} - {a}")
        for blk in sorted({k[0] // BLOCK for k in common}):
            keys = [k for k in common if k[0] // BLOCK == blk]
            diff = lambda f: [data[b][k][f] - data[a][k][f] for k in keys
                              if data[a][k][f] is not None and data[b][k][f] is not None]
            plays, wasted, first = diff("plays"), diff("wasted"), diff("first")
            mean = lambda xs: st.mean(xs) if xs else float("nan")
            print(f"    编号 {blk * BLOCK:5d}+  {len(keys):3d} 局  出牌 {mean(plays):+5.1f}  "
                  f"浪费 {mean(wasted):+5.1f}  首牌 {mean(first):+5.1f} 秒")
