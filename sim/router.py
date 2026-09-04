from .views import PerfectView


class Router:
    """The one place a request is matched to a worker.

    choose -> record what the view promised against what was true -> tell the
    view about the dispatch -> submit. Scripts never call policy.choose on their
    own, so every experiment gets the error accounting and the dispatch hook
    without knowing they exist.
    """

    def __init__(self, engine, policy, workers, view_factory=None, tracker=None,
                 record_block_samples: bool = False):
        self.engine = engine
        self.policy = policy
        self.workers = workers

        # opt-in, for the theory check: one (known block age, view age, was
        # the promise false, depth) sample per promised block. off by default
        # because it grows by every matched block of every request
        self.block_samples = [] if record_block_samples else None

        # per-block reference rates from the router's own dispatches; only the
        # survival view reads it, so it is optional
        self.tracker = tracker

        # per worker, so a later herd index can ask who got the burst
        self.dispatches = [0] * len(workers)

        for worker in workers:
            worker.view = (
                view_factory(worker)
                if view_factory is not None
                else PerfectView(worker.cache, engine)
            )

    def dispatch(self, req) -> None:
        chosen = self.policy.choose(req, self.workers)

        # estimates first: a view that records dispatches must not get to count
        # this request's own blocks as already present, and the rate tracker
        # must not count this request's own references either
        self._record_estimates(req, chosen)
        chosen.view.record_dispatch(req.blocks, self.engine.now)

        if self.tracker is not None:
            self.tracker.observe(req.blocks, self.engine.now, chosen.id)

        self.dispatches[chosen.id] += 1
        chosen.submit(req)

    def replay(self, requests) -> None:
        for req in requests:
            delay = req.arrival - self.engine.now
            self.engine.schedule(delay, self.dispatch, req)

    def _record_estimates(self, req, chosen) -> None:
        """Three numbers per decision, all in tokens like cached_tokens.

        The worker sets req.cached_tokens later, at admission, so it cannot be
        read here. These capture what was knowable at the instant of routing.
        """

        def cached_tokens_for(matched_blocks: int, worker) -> int:
            return min(matched_blocks * worker.block_size, req.prompt_tokens)

        # what the router's view promised on the worker it picked
        req.estimated_cached_tokens = cached_tokens_for(
            chosen.view.match(req.blocks),
            chosen,
        )

        # what that worker really held right now
        true_blocks = chosen.cache.match(req.blocks)
        req.true_cached_tokens_at_dispatch = cached_tokens_for(true_blocks, chosen)

        # a match is prefix-wise, so the promised blocks past the true match
        # are exactly the false promises. each sample carries how old the view
        # believed the block to be, so false positives can be binned by age
        if self.block_samples is not None:
            match_with_ages = getattr(chosen.view, "match_with_ages", None)
            if match_with_ages is not None:
                known_ages = match_with_ages(req.blocks)
                self.block_samples.extend(
                    (known_age, chosen.view.age, index >= true_blocks, index + 1)
                    for index, known_age in enumerate(known_ages)
                )

        # the most any worker really held right now: the best decision possible
        req.best_cached_tokens_at_dispatch = max(
            cached_tokens_for(worker.cache.match(req.blocks), worker)
            for worker in self.workers
        )

        # how old the picture behind the estimate was; 0 for a perfect view
        req.view_age_at_dispatch = chosen.view.age

        # a survival view also says how much of its promise it expects to hold
        match_expected = getattr(chosen.view, "match_expected", None)
        if match_expected is not None:
            req.expected_cached_tokens = min(
                match_expected(req.blocks) * chosen.block_size,
                float(req.prompt_tokens),
            )
