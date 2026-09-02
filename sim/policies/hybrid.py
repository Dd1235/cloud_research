class Hybrid:
    """score = alpha * overlap - beta * load, both normalised to [0, 1].

    This is the shape every production router uses: a cache term added to a
    load term with tuned weights. Adding is what makes it fragile to a stale
    view, because an inflated overlap outvotes the load term outright. The
    overlap_source switch lets the same scorer read the view's survival-
    weighted expectation instead of its raw promise, when the view offers one.
    """

    name = "hybrid"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, overlap_source: str = "raw"):
        assert overlap_source in ("raw", "expected")

        self.alpha = alpha
        self.beta = beta
        self.overlap_source = overlap_source

    def overlap(self, req, worker) -> float:
        if not req.blocks:
            return 0.0

        if self.overlap_source == "expected":
            match_expected = getattr(worker.view, "match_expected", None)
            if match_expected is not None:
                return match_expected(req.blocks) / len(req.blocks)

        return worker.view.match(req.blocks) / len(req.blocks)

    def score(self, req, worker, max_outstanding: int) -> float:
        load = (
            worker.outstanding / max_outstanding
            if max_outstanding
            else 0.0
        )

        return self.alpha * self.overlap(req, worker) - self.beta * load

    def choose(self, req, workers):
        max_outstanding = max(
            worker.outstanding
            for worker in workers
        )

        # ties break towards the lower worker id, so a run is reproducible
        return max(
            workers,
            key=lambda worker: (
                self.score(req, worker, max_outstanding),
                -worker.id,
            ),
        )

# alpha = 1, beta = 0 - locality only
# alpha = 0, beta = 1, load only
# alpha = 1, beta = 1 equal tradeoff
