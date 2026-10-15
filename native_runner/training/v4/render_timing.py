"""Enforce policy action gaps on a running renderer, using native execution receipts."""

from ...cr_native_env import NativeClashEnv
from ...play_timing import DEFAULT_PLAY_GAP_TICKS, PlayGapClock


class SpacedRenderNativeV4(NativeClashEnv):
    def __init__(self, *args, min_play_gap_ticks=DEFAULT_PLAY_GAP_TICKS, timing_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.play_gap = PlayGapClock(min_play_gap_ticks, seed=timing_seed)

    def _queue_spaced(self, owner, execute_in_ticks, queue):
        # Queue against one stopped native clock: inference/render time cannot
        # shorten the gap, and future plays remain pending in BattleEnvV1.
        with self._lock:
            self.pause()
            try:
                tick = int(self.observe()["tick"])
                target = self.play_gap.next_tick(owner, tick + max(1, execute_in_ticks))
                receipt = queue(target - tick)
                if not receipt.get("terminalCanceled"):
                    self.play_gap.commit(owner, int(receipt["executeTick"]))
                return receipt
            finally:
                self.resume()

    def queue_hand_action_next_tick(self, action):
        return self._queue_spaced(
            action.owner, 1,
            lambda delay: super(SpacedRenderNativeV4, self).queue_hand_action_at(action, execute_in_ticks=delay),
        )

    def queue_ability_action_at(self, action, *, execute_in_ticks=1):
        return self._queue_spaced(
            action.owner, execute_in_ticks,
            lambda delay: super(SpacedRenderNativeV4, self).queue_ability_action_at(action, execute_in_ticks=delay),
        )
