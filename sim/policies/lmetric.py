class Lmetric:
    """LMETRIC (Zhang et al., OSDI'26): score = new prefill tokens x batch size.

    The product needs no tuned weight: a full cache hit makes the score zero
    however busy the worker is, and a cold worker's score grows with both its
    queue and the prompt. It is still a cardinal use of the cache view, so an
    inflated overlap shrinks the score of a worker that has in fact evicted
    the prefix. Batch size is approximated by the requests already on the
    worker plus this one.
    """

    name = "lmetric"

    def __init__(self, overlap_source: str = "raw"):
        assert overlap_source in ("raw", "expected")
        self.overlap_source = overlap_source

    def matched_blocks(self, req, worker) -> float:
        if self.overlap_source == "expected":
            match_expected = getattr(worker.view, "match_expected", None)
            if match_expected is not None:
                return match_expected(req.blocks)

        return worker.view.match(req.blocks)

    def score(self, req, worker) -> float:
        cached_tokens = min(
            self.matched_blocks(req, worker) * worker.block_size,
            req.prompt_tokens,
        )
        new_prefill_tokens = req.prompt_tokens - cached_tokens
        batch_size = worker.outstanding + 1

        return new_prefill_tokens * batch_size

    def choose(self, req, workers):
        return min(
            workers,
            key=lambda worker: (self.score(req, worker), worker.id),
        )
