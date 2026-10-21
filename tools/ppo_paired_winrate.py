"""对固定对手的胜率：按对局编号分块，并在同一编号、同卡组、同对手的对局上配对比较几个运行。

在仓库根目录运行：python3 tools/ppo_paired_winrate.py [运行名 ...]（默认 spar16-snr-a spar16-acc1 spar16-acc4）。
"""
import glob, json, sys
from collections import defaultdict
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
            rows[key] = (g["update"], g["result"][owner] > 0)
    data[run] = rows
print("对固定对手的胜率，按对局编号每 256 局一格（括号里是这些对局由第几次更新的策略打的）")
count = lambda b: sum(any(k[0] // BLOCK == b for k in rows) for rows in data.values())
blocks = [b for b in sorted({k[0] // BLOCK for rows in data.values() for k in rows}) if count(b) >= 2]
print("编号块      " + "".join(f"{r:>26s}" for r in runs))
for b in blocks:
    cells = []
    for r in runs:
        part = [v for k, v in data[r].items() if k[0] // BLOCK == b]
        if part:
            ups = [u for u, _ in part]
            cells.append(f"{100*sum(w for _, w in part)/len(part):4.0f}% n={len(part):3d} (u{min(ups)}-{max(ups)})")
        else:
            cells.append("-")
    print(f"{b*BLOCK:5d}-{b*BLOCK+BLOCK-1:<5d}" + "".join(f"{c:>26s}" for c in cells))
print()
print("配对比较：同一编号、同卡组、同对手的对局")
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        a, b = runs[i], runs[j]
        common = sorted(set(data[a]) & set(data[b]))
        if not common:
            print(f"  {a} vs {b}: 没有共同对局"); continue
        by = defaultdict(lambda: [0, 0, 0, 0])
        for k in common:
            row = by[k[0] // BLOCK]
            wa, wb = data[a][k][1], data[b][k][1]
            row[0] += 1; row[1] += wa; row[2] += wb; row[3] += (wa != wb)
        print(f"  {a} vs {b}")
        for blk, (n, wa, wb, disc) in sorted(by.items()):
            print(f"    编号 {blk*BLOCK:5d}+  共同 {n:3d} 局  {a} {100*wa/n:4.0f}%  {b} {100*wb/n:4.0f}%  结果不同 {disc:3d} 局")
