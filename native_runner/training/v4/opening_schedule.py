"""Reproducible per-game frozen-opponent opening schedules, without engine dependencies."""

from __future__ import annotations

import random


def opening_force_seconds_v4(bounds: tuple[float, float] | None, key: object) -> float | None:
    """The battle time, drawn uniformly from ``bounds`` and fixed per game by ``key``, to force an opening."""

    if bounds is None:
        return None
    return random.Random(f"opening-force-{key}").uniform(*bounds)


def parse_seconds_range(value: str) -> tuple[float, float]:
    low, _, high = value.strip().partition("-")
    try:
        bounds = (float(low), float(high or low))
    except ValueError:
        raise ValueError("expected N or MIN-MAX seconds, e.g. 8-20") from None
    if not 0.0 < bounds[0] <= bounds[1] <= 180.0:
        raise ValueError("seconds must satisfy 0 < MIN <= MAX <= 180")
    return bounds


def frozen_opponent_opening_plan_v4(
    bounds: tuple[float, float] | None,
    key: object,
    *,
    natural_percent: int = 0,
    il_after_seconds: int | None = None,
) -> tuple[bool, float | None]:
    """Return (exempt, deadline), keeping the selection fixed for a whole game.

    Draw the exemption separately from the deadline so existing deadline draws
    stay unchanged. The exemption also bypasses the optional legacy IL timeout.
    """
    if isinstance(natural_percent, bool) or not isinstance(natural_percent, int) or not 0 <= natural_percent <= 100:
        raise ValueError("natural opening percent must be an integer in [0, 100]")
    if bounds is None and il_after_seconds is None:
        return False, None
    exempt = random.Random(f"opening-natural-{key}").random() < natural_percent / 100.0
    if exempt:
        return True, None
    if il_after_seconds is not None:
        return False, float(il_after_seconds)
    return False, opening_force_seconds_v4(bounds, key)

