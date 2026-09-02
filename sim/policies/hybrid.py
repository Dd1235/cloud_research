class Hybrid:
    name = "hybrid"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def score(self, req, worker, max_outstanding: int) -> float:
        overlap = (
            worker.cache.match(req.blocks) / len(req.blocks)
            if req.blocks
            else 0.0
        )

        load = (
            worker.outstanding / max_outstanding
            if max_outstanding
            else 0.0
        )

        return self.alpha * overlap - self.beta * load

    def choose(self, req, workers):
        # both terms are normalised to [0, 1] so alpha and beta are comparable
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
