import heapq
import itertools

import numpy as np


class Engine:

    def __init__(self, seed: int = 0):
        self.now = 0.0
        self.rng = np.random.default_rng(seed)
        self._heap = []
        self._seq = itertools.count()

        # events that keep the run going. daemons (periodic samplers, cache view
        # refreshes) reschedule themselves forever, so they must not count or a
        # run would never end
        self._live = 0

    def schedule(self, delay: float, fn, *args, daemon: bool = False) -> None:

        assert delay >= 0, "i don't have time machine"

        event_time = self.now + delay
        heapq.heappush(
            self._heap,
            (event_time, next(self._seq), daemon, fn, args),
        )

        if not daemon:
            self._live += 1

    # generic event loop. run() stops once no live events remain, even if
    # daemons are still pending. run(until=T) executes every event at or
    # before T, daemon or live, so a periodic sampler can be tested on its own
    def run(self, until : float | None = None) -> None:
        while self._heap:
            if until is None and self._live == 0:
                return

            event_time, seq, daemon, fn, args = heapq.heappop(self._heap)

            if until is not None and event_time > until:
                heapq.heappush(
                    self._heap,
                    (event_time, seq, daemon, fn, args),
                )
                return

            if not daemon:
                self._live -= 1

            self.now = event_time
            fn(*args)


    @property
    def pending(self) -> int:
        return len(self._heap)
