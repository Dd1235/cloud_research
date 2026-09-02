from collections import defaultdict, deque


class BlockRateTracker:
    """How often each block is referenced, seen from the router's own dispatches.

    A block's re-reference rate is what decides whether a stale view entry is
    still true: a hot block keeps refreshing its place in the worker's cache, a
    cold one does not. The router sees every request it routes, so it can
    estimate the rate per block over a sliding window without asking any worker.
    """

    def __init__(self, window: float):
        assert window > 0

        self.window = window
        self._reference_times = defaultdict(deque)

    def observe(self, blocks, now: float) -> None:
        for block in blocks:
            times = self._reference_times[block]
            times.append(now)
            self._forget_old(times, now)

    def rate(self, block, now: float) -> float:
        """References per second over the window, 0.0 for a block never seen."""
        times = self._reference_times.get(block)

        if not times:
            return 0.0

        self._forget_old(times, now)
        return len(times) / self.window

    def _forget_old(self, times, now: float) -> None:
        while times and times[0] < now - self.window:
            times.popleft()
