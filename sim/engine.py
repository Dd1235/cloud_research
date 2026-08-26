import heapq
import itertools

import numpy as np


class Engine:

    def __init__(self, seed: int = 0):
        self.now = 0.0
        self.rng = np.random.default_rng(seed)
        self._heap = []
        self._seq = itertools.count()

    def schedule(self, delay: float, fn, *args) -> None:

        assert delay >= 0, "i don't have time machine"

        event_time = self.now + delay
        heapq.heappush(
            self._heap,
            (event_time, next(self._seq), fn, args),
        )

    # generic event loop
    def run(self, until : float | None = None) -> None:
        while self._heap:
            event_time, seq, fn, args = heapq.heappop(self._heap)

            if until is not None and event_time > until:
                heapq.heappush(
                    self._heap,
                    (event_time, seq, fn, args),
                )
                return

            self.now = event_time
            fn(*args)


    @property
    def pending(self) -> int:
        return len(self._heap)