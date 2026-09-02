from .views import PerfectView


class Router:
    """The one place a request is matched to a worker.

    choose -> record what the view promised against what was true -> tell the
    view about the dispatch -> submit. Scripts never call policy.choose on their
    own, so every experiment gets the error accounting and the dispatch hook
    without knowing they exist.
    """

    def __init__(self, engine, policy, workers, view_factory=None):
        self.engine = engine
        self.policy = policy
        self.workers = workers

        # per worker, so a later herd index can ask who got the burst
        self.dispatches = [0] * len(workers)

        for worker in workers:
            worker.view = (
                view_factory(worker)
                if view_factory is not None
                else PerfectView(worker.cache)
            )

    def dispatch(self, req) -> None:
        chosen = self.policy.choose(req, self.workers)

        # estimates first: a view that records dispatches must not get to count
        # this request's own blocks as already present
        self._record_estimates(req, chosen)
        chosen.view.record_dispatch(req.blocks, self.engine.now)

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
        req.true_cached_tokens_at_dispatch = cached_tokens_for(
            chosen.cache.match(req.blocks),
            chosen,
        )

        # the most any worker really held right now: the best decision possible
        req.best_cached_tokens_at_dispatch = max(
            cached_tokens_for(worker.cache.match(req.blocks), worker)
            for worker in self.workers
        )

        # how old the picture behind the estimate was; 0 for a perfect view
        req.view_age_at_dispatch = chosen.view.age
