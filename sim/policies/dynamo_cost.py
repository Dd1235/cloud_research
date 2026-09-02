class DynamoCost:
    """NVIDIA Dynamo's KV router cost, in block units.

        cost = overlap_weight * prefill_blocks_still_needed + active_blocks

    where the first term is what routing here would have to prefill and the
    second is the decode load the worker already carries. Dynamo counts the
    KV blocks of every active request; here that is approximated by the
    outstanding request count times this request's own block count, which
    keeps the two terms in the same unit. The lowest cost wins. Like every
    added-together scorer it reads the cache view cardinally.
    """

    name = "dynamo_cost"

    def __init__(self, overlap_weight: float = 1.0, overlap_source: str = "raw"):
        assert overlap_source in ("raw", "expected")

        self.overlap_weight = overlap_weight
        self.overlap_source = overlap_source

    def matched_blocks(self, req, worker) -> float:
        if self.overlap_source == "expected":
            match_expected = getattr(worker.view, "match_expected", None)
            if match_expected is not None:
                return match_expected(req.blocks)

        return worker.view.match(req.blocks)

    def cost(self, req, worker) -> float:
        prompt_blocks = len(req.blocks)
        prefill_blocks = prompt_blocks - self.matched_blocks(req, worker)
        active_blocks = worker.outstanding * prompt_blocks

        return self.overlap_weight * prefill_blocks + active_blocks

    def choose(self, req, workers):
        return min(
            workers,
            key=lambda worker: (self.cost(req, worker), worker.id),
        )
