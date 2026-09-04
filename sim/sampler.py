class OutstandingSampler:
    """Records how many requests each worker holds, at a fixed interval.

    busy_time tells how much work a worker did over the whole run; it says
    nothing about whether that work arrived in bursts that queued. Sampling
    the outstanding count over time is what load imbalance means for tail
    latency: a worker that is idle half the time and swamped the other half
    has the same busy_time as a steady one and a very different p99.

    Runs as a daemon event so it never keeps the simulation alive.
    """

    def __init__(self, engine, workers, interval: float = 0.1):
        assert interval > 0

        self._engine = engine
        self._workers = workers
        self._interval = interval

        # (time, {worker id: outstanding}) per tick. keyed by id, not by
        # position: the workers list can gain and lose members mid-run
        # (scale-out and scale-in), and a positional row would silently
        # misattribute counts the moment the composition changes
        self.samples = []

        engine.schedule(0.0, self._sample, daemon=True)

    def _sample(self) -> None:
        self.samples.append(
            (
                self._engine.now,
                {worker.id: worker.outstanding for worker in self._workers},
            )
        )

        self._engine.schedule(self._interval, self._sample, daemon=True)

    def mean_outstanding(self, since: float = 0.0) -> list[float]:
        """Per worker time average of the outstanding count, from `since` on.

        A worker absent from a tick (not yet added, or already removed) is
        counted as holding nothing then: its average is over the whole window,
        so a latecomer's idle prehistory shows as a low mean, which is what a
        fleet-imbalance number should say about a worker that was not there.
        """
        counted = [
            outstanding
            for sampled_at, outstanding in self.samples
            if sampled_at >= since
        ]

        if not counted:
            return [0.0 for _ in self._workers]

        worker_ids = sorted({worker_id for per_tick in counted for worker_id in per_tick})

        return [
            sum(per_tick.get(worker_id, 0) for per_tick in counted) / len(counted)
            for worker_id in worker_ids
        ]
