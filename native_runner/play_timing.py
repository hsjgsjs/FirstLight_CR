"""Shared consecutive-action timing, in game ticks (50 ms each)."""

import random


DEFAULT_PLAY_GAP_TICKS = (6, 10)


def parse_play_gap_ticks(value: str) -> tuple[int, int]:
    low, _, high = value.strip().partition("-")
    try:
        bounds = (int(low), int(high or low))
    except ValueError:
        raise ValueError("play gap must be N or MIN-MAX ticks, e.g. 6-10") from None
    if not 1 <= bounds[0] <= bounds[1] <= 200:
        raise ValueError("play gap must be 1 <= MIN <= MAX <= 200 ticks")
    return bounds


class PlayGapClock:
    """Space accepted commands independently for each owner; WAITs never enter the clock."""

    def __init__(self, bounds=DEFAULT_PLAY_GAP_TICKS, *, seed=0):
        self.bounds = bounds
        self.last: dict[int, int] = {}
        self.rng = random.Random(seed)

    def next_tick(self, owner: int, earliest: int) -> int:
        if self.bounds is None or owner not in self.last:
            return earliest
        return max(earliest, self.last[owner] + self.rng.randint(*self.bounds))

    def commit(self, owner: int, tick: int) -> None:
        self.last[owner] = tick
