from .radix import PrefixCache


class CacheView:
    """What the router believes a worker's cache holds.

    Policies only ever call match(). The router additionally reports every
    dispatch through record_dispatch(), which only views that build their own
    picture from routing decisions care about; the rest ignore it.
    """

    def match(self, blocks) -> int:
        raise NotImplementedError

    def record_dispatch(self, blocks, now: float) -> None:
        pass

    # how old the information behind match() is, in seconds. the router records
    # it per decision so a sweep can report the age actually observed rather
    # than the knob that produced it
    @property
    def age(self) -> float:
        return 0.0


class PerfectView(CacheView):
    """The worker's true cache, read live at the moment of the decision.

    This is what every policy did before views existed. It is the zero-age
    point of the staleness sweep and the upper bound nothing real can reach.
    """

    def __init__(self, cache: PrefixCache):
        self._cache = cache

    def match(self, blocks) -> int:
        return self._cache.match(blocks)


class SnapshotView(CacheView):
    """View model A: a copy of the true cache taken every `period` seconds.

    Between refreshes the router routes on a picture that is anywhere from 0
    to one period old, mean period / 2. That is exactly what a metrics scrape
    gives a production router, and what the mac cluster will do for real.

    The refresh is a daemon event: it reschedules itself forever, so it must
    never be the thing keeping a run alive.
    """

    def __init__(self, engine, cache: PrefixCache, period: float, phase: float = 0.0):
        assert period > 0

        self._engine = engine
        self._cache = cache
        self._period = period

        # nothing scraped yet: match() reports an empty cache
        self._snapshot = None
        self.taken_at = None
        self.refreshes = 0

        engine.schedule(phase, self._refresh, daemon=True)

    def _refresh(self) -> None:
        self._snapshot = self._cache.copy()
        self.taken_at = self._engine.now
        self.refreshes += 1

        self._engine.schedule(self._period, self._refresh, daemon=True)

    @property
    def age(self) -> float:
        if self.taken_at is None:
            return float("inf")

        return self._engine.now - self.taken_at

    def match(self, blocks) -> int:
        if self._snapshot is None:
            return 0

        return self._snapshot.match(blocks)


class ShadowView(CacheView):
    """View model C: the router's own record of what it routed where.

    Every dispatch is inserted into a router-side prefix cache with its own
    capacity, so the view is exactly the router's routing history and nothing
    else: worker evictions are never seen, and a block the worker dropped
    long ago still looks present. This is llm-d's approximate mode and what
    Mitzenmacher called record-insert.
    """

    def __init__(self, capacity_blocks: int):
        self._index = PrefixCache(capacity_blocks)

    def match(self, blocks) -> int:
        return self._index.match(blocks)

    def record_dispatch(self, blocks, now: float) -> None:
        self._index.insert(blocks, now)


VIEW_KINDS = ("perfect", "snapshot", "shadow")


def make_view_factory(kind: str, engine, *, period=None, shadow_blocks=None):
    """Returns a function worker -> CacheView for the requested view model.

    None means the perfect view; the router then wraps each worker's own cache.
    """
    if kind == "perfect":
        return None

    if kind == "snapshot":
        assert period is not None, "a snapshot view needs a refresh period"
        return lambda worker: SnapshotView(engine, worker.cache, period)

    if kind == "shadow":
        assert shadow_blocks is not None, "a shadow view needs a capacity"
        return lambda worker: ShadowView(shadow_blocks)

    known_kinds = ", ".join(VIEW_KINDS)
    raise KeyError(f"unknown view {kind!r}; known views: {known_kinds}")
