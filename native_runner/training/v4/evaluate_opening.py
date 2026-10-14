"""Paired CPU-only evaluation of a frozen opponent's opening timeout.

Each block runs eight games: natural/forced B x both sides x both deck assignments.
An already running, idle native engine is required; this command never starts training.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random

import torch

from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT
from ...match_factory import MatchConfig
from .async_cluster_self_play import opening_force_seconds_v4, parse_play_gap_ticks, parse_seconds_range
from .deck_pool import DeckPoolManifestV4
from .evaluate import evaluate, sample_deck_pair
from .evaluation_protocol import balanced_cases, paired_summary
from .policy_session import resolve_v4_checkpoint


def build_schedule(match, *, blocks, opening_seconds, deck_pool=None, block_start=0):
    if blocks < 1 or block_start < 0:
        raise ValueError("blocks must be positive and block_start nonnegative")
    rng = random.Random(match.seed)
    cases = []
    # Advance the same deck RNG for skipped blocks so changing engine count does
    # not change any block's seed, decks, ordering or opening deadline.
    for block in range(block_start + blocks):
        base = replace(match, seed=match.seed + block)
        ids = None
        if deck_pool is not None:
            base, ids = sample_deck_pair(base, deck_pool, rng)
        if block < block_start:
            continue
        cases.extend(balanced_cases(
            base, block=block,
            deadline_seconds=opening_force_seconds_v4(opening_seconds, base.seed),
            deck_ids=ids,
        ))
    return cases


def render_report(summary):
    lines = [
        "# 强制对手早出牌：CPU 配对评测", "",
        "A 为固定参考模型；B 为被测对手。仅 forced 条件对 B 启用开局限制。", "",
        "| 条件 | 正常结束 | B 胜 / 负 / 平 | B 胜率 | B 得分率 | 实际强制的局数 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]

    def percent(value):
        return "无数据" if value is None else f"{value:.2%}"

    for condition, row in summary["conditions"].items():
        lines.append(
            f"| {condition} | {row['finished']}/{row['games']} | "
            f"{row['b_wins']} / {row['b_losses']} / {row['draws']} | "
            f"{percent(row['b_win_rate'])} | {percent(row['b_score'])} | {row['forced_games']} |"
        )
    interval = summary["b_score_delta_95"]
    interval_text = "完整组不足 2 组" if interval is None else f"[{percent(interval[0])}, {percent(interval[1])}]"
    labels = {
        "insufficient_blocks": "完整组少于 20 组，仅供流程检查和初步观察。",
        "inconclusive": "区间跨越允许下降幅度，暂时不能判断是否明显变弱。",
        "drop_exceeds_margin": "区间整体低于允许下降幅度，被测对手在这组条件下明显变弱。",
        "drop_within_margin": "区间支持下降幅度小于设定阈值；结论仅适用于本次参考模型与卡组分布。",
    }
    lines.extend([
        "", f"完整配对组：{summary['complete_blocks']}；排除不完整组：{summary['excluded_blocks']}。",
        f"B 得分率变化（forced − natural）：{percent(summary['b_score_delta_forced_minus_natural'])}。",
        f"按完整组 bootstrap 的 95% 区间：{interval_text}。",
        f"允许下降幅度：{percent(summary['max_score_drop'])}。{labels[summary['conclusion']]}",
        "", "得分率 =（胜 + 0.5 × 平）/ 正常结束局数。负变化表示 B 变弱。",
        "表格包含各条件所有正常结束对局；配对变化仅使用 8 局均正常结束的完整组。",
        "未结束或截断不算平局。首张牌落地时间、动作拒绝数和强制次数见 JSON。",
        "少量样本或全部相同结果可能得到过窄的 bootstrap 区间；应增加种子并更换参考模型复核。",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint-a", required=True, help="fixed reference policy")
    parser.add_argument("--checkpoint-b", required=True, help="opponent tested with and without the timeout")
    parser.add_argument("--blocks", type=int, default=20, help="8 games per block (default: 160 games total)")
    parser.add_argument("--block-start", type=int, default=0, help="first global block index for parallel workers")
    parser.add_argument("--opening-seconds", type=parse_seconds_range, default=(8.0, 20.0))
    parser.add_argument("--match-config", type=Path)
    parser.add_argument("--deck-pool", type=Path)
    parser.add_argument("--battle-level", type=int, choices=(11, 16), default=11)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", help="argmax decisions; default samples as in training")
    parser.add_argument("--gate-temperature", type=float, help="same override for both models; default uses saved values")
    parser.add_argument("--delay-ticks", type=int, default=0, help="same execution delay for both models")
    parser.add_argument("--min-play-gap-ticks", type=parse_play_gap_ticks)
    parser.add_argument("--max-score-drop", type=float, default=0.05, help="absolute B score drop threshold; 0.05 = 5 pp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="write the schedule without loading models or connecting")
    args = parser.parse_args(argv)
    if args.blocks < 1 or args.cpu_threads < 1 or args.block_start < 0:
        parser.error("--blocks and --cpu-threads must be positive; --block-start must be nonnegative")
    if not 0 <= args.max_score_drop <= 1 or not 0 <= args.delay_ticks <= 65535:
        parser.error("invalid score-drop margin or execution delay")
    if args.gate_temperature is not None and (not math.isfinite(args.gate_temperature) or args.gate_temperature <= 0):
        parser.error("--gate-temperature must be finite and positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory, to keep evaluation records separate")
    config = json.loads(args.match_config.read_text(encoding="utf-8")) if args.match_config else {}
    for key in ("owner0_account_id", "owner1_account_id"):
        config.pop(key, None)
    config.update(
        seed=args.seed, level_cap=args.battle_level,
        minimum_card_level=args.battle_level, king_tower_level=args.battle_level,
    )
    match = MatchConfig(**config)
    deck_pool = None if args.deck_pool is None else DeckPoolManifestV4.load(args.deck_pool)
    schedule = build_schedule(
        match, blocks=args.blocks, opening_seconds=args.opening_seconds,
        deck_pool=deck_pool, block_start=args.block_start,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = {
        "schema": "opening-evaluation.v1", "device": device, "games": len(schedule),
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cases": [asdict(case) for case in schedule],
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.plan_only:
        print(f"Planned {len(schedule)} games in {args.output / 'plan.json'}")
        return 0
    # Resolve directories once, so a concurrently trained checkpoint cannot change
    # between conditions. evaluate loads both models once for the whole schedule.
    checkpoint_a = resolve_v4_checkpoint(args.checkpoint_a)
    checkpoint_b = resolve_v4_checkpoint(args.checkpoint_b)
    torch.set_num_threads(args.cpu_threads)
    evaluate(
        checkpoint_a, checkpoint_b, games=len(schedule), match=match, output=args.output,
        host=args.host, port=args.port, device=device, sample=not args.deterministic,
        delay_a_ticks=args.delay_ticks, delay_b_ticks=args.delay_ticks,
        min_play_gap_ticks=args.min_play_gap_ticks,
        gate_temperature_a=args.gate_temperature, gate_temperature_b=args.gate_temperature,
        schedule=schedule,
    )
    rows = [json.loads(line) for line in (args.output / "matches.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary = paired_summary(rows, max_score_drop=args.max_score_drop, seed=args.seed)
    summary["requested_blocks"] = args.blocks
    summary["requested_games"] = len(schedule)
    (args.output / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = render_report(summary)
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    return 0 if summary["complete_blocks"] == args.blocks else 1


if __name__ == "__main__":
    raise SystemExit(main())
