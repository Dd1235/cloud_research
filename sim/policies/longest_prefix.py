class LongestPrefix:
    """Rank workers by matched prefix, break ties by shortest queue.

    match_threshold is SGLang's production guard, as a fraction of the
    request's blocks: when even the best match is below it, the match is too
    small to be worth chasing and the request routes by load alone. At 0 the
    ranker is pure, which is deliberately dangerous: one recorded block on one
    worker decides every request that shares it (the lock-in on real traces).
    """

    name = "longest_prefix"

    def __init__(self, match_threshold: float = 0.0):
        assert 0.0 <= match_threshold <= 1.0

        self.match_threshold = match_threshold

    def choose(self, req, workers):
        best = max(
            workers,
            key=lambda worker: (
                worker.view.match(req.blocks),
                -worker.outstanding,
            ),
        )

        if best.view.match(req.blocks) < self.match_threshold * len(req.blocks):
            return min(workers, key=lambda worker: worker.outstanding)

        return best
