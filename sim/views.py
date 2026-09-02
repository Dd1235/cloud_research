import math

from .radix import PrefixCache


class CacheView:
    """What the router believes a worker's cache holds.

    Policies only ever call match(). The router additionally reports every
    dispatch through record_dispatch(), which only views that build their own
    picture from routing decisions care about; the rest ignore it.
    """

    def match(self, blocks) -> int:
        raise NotImplementedError

    # the same leading blocks as match(), each with its last-access age as the
    # view knows it. this is what lets a view distrust a block that has sat
    # untouched for a whole cache lifetime
    def match_with_ages(self, blocks) -> list[float]:
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

    def __init__(self, cache: PrefixCache, engine=None):
        self._cache = cache
        self._engine = engine

    def match(self, blocks) -> int:
        return self._cache.match(blocks)

    def match_with_ages(self, blocks) -> list[float]:
        return self._cache.match_with_ages(blocks, self._engine.now)


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

    def match_with_ages(self, blocks) -> list[float]:
        if self._snapshot is None:
            return []

        # ages are measured against now, so a block's age includes the time
        # since the scrape; a re-reference since then is invisible to the view
        return self._snapshot.match_with_ages(blocks, self._engine.now)


class ShadowView(CacheView):
    """View model C: the router's own record of what it routed where.

    Every dispatch is inserted into a router-side prefix cache with its own
    capacity, so the view is exactly the router's routing history and nothing
    else: worker evictions are never seen, and a block the worker dropped
    long ago still looks present. This is llm-d's approximate mode and what
    Mitzenmacher called record-insert.
    """

    def __init__(self, capacity_blocks: int, engine=None):
        self._index = PrefixCache(capacity_blocks)
        self._engine = engine

    def match(self, blocks) -> int:
        return self._index.match(blocks)

    def match_with_ages(self, blocks) -> list[float]:
        return self._index.match_with_ages(blocks, self._engine.now)

    def record_dispatch(self, blocks, now: float) -> None:
        self._index.insert(blocks, now)


class TtlView(CacheView):
    """Trust a block only while its last-access age is below a time to live.

    Che's approximation says an LRU cache of C blocks behaves like a cache in
    which every block lives a characteristic time T_C after its last access.
    So a view entry older than T_C is, in expectation, describing a block that
    has already been evicted. Cutting the match at the first block older than
    the ttl turns any view (snapshot, shadow, event fed) into one whose false
    positives are bounded by the same rule, with no extra telemetry: the ages
    are already in the view. Dynamo's fixed 120 s expiry is this with a guess
    where the measured turnover should be.
    """

    def __init__(self, inner: CacheView, ttl: float):
        assert ttl > 0

        self._inner = inner
        self.ttl = ttl

    def _trusted_ages(self, blocks) -> list[float]:
        trusted = []

        for age in self._inner.match_with_ages(blocks):
            if age >= self.ttl:
                break   # a prefix match must be contiguous from the first block

            trusted.append(age)

        return trusted

    def match(self, blocks) -> int:
        return len(self._trusted_ages(blocks))

    def match_with_ages(self, blocks) -> list[float]:
        return self._trusted_ages(blocks)

    def record_dispatch(self, blocks, now: float) -> None:
        self._inner.record_dispatch(blocks, now)

    @property
    def age(self) -> float:
        return self._inner.age


class SurvivalView(CacheView):
    """Weight each block the view reports by the chance it is really still there.

    A block seen at last-access age a has survived if it was re-referenced since
    (probability 1 - exp(-rate * a)), or, if not, if its residual life outran a
    (probability 1 - a / T_C under Che's model). So

        S(a) = 1 - exp(-rate * a) * min(a / T_C, 1)

    and the expected surviving prefix match is the sum over depth of the product
    of the survivals along the path, because a prefix only survives to depth j
    if every block before it did. match() itself is left ordinal and unchanged;
    only scorers that add overlap to a load term need the expectation.
    """

    def __init__(self, inner: CacheView, engine, tracker, turnover: float):
        assert turnover > 0

        self._inner = inner
        self._engine = engine
        self._tracker = tracker
        self.turnover = turnover

    def survival(self, block, age: float) -> float:
        rate = self._tracker.rate(block, self._engine.now)
        gone_if_not_refreshed = min(age / self.turnover, 1.0)

        return 1.0 - math.exp(-rate * age) * gone_if_not_refreshed

    def match_expected(self, blocks) -> float:
        path_survival = 1.0
        expected_depth = 0.0

        for block, age in zip(blocks, self._inner.match_with_ages(blocks)):
            path_survival *= self.survival(block, age)
            expected_depth += path_survival

        return expected_depth

    def match(self, blocks) -> int:
        return self._inner.match(blocks)

    def match_with_ages(self, blocks) -> list[float]:
        return self._inner.match_with_ages(blocks)

    def record_dispatch(self, blocks, now: float) -> None:
        self._inner.record_dispatch(blocks, now)

    @property
    def age(self) -> float:
        return self._inner.age


VIEW_KINDS = ("perfect", "snapshot", "shadow")


def make_view_factory(kind: str, engine, *, period=None, shadow_blocks=None, ttl=None,
                      tracker=None, turnover=None):
    """Returns a function worker -> CacheView for the requested view model.

    ttl wraps whichever view is built so entries older than it are not trusted;
    tracker + turnover wrap it in a survival view instead. The perfect view is
    never wrapped: it has no stale entries to distrust.
    """
    assert ttl is None or tracker is None, "ttl and survival are alternatives"

    if kind == "perfect":
        return None

    if kind == "snapshot":
        assert period is not None, "a snapshot view needs a refresh period"
        base = lambda worker: SnapshotView(engine, worker.cache, period)
    elif kind == "shadow":
        assert shadow_blocks is not None, "a shadow view needs a capacity"
        base = lambda worker: ShadowView(shadow_blocks, engine)
    else:
        known_kinds = ", ".join(VIEW_KINDS)
        raise KeyError(f"unknown view {kind!r}; known views: {known_kinds}")

    if ttl is not None:
        return lambda worker: TtlView(base(worker), ttl)

    if tracker is not None:
        assert turnover is not None, "a survival view needs the cache turnover"
        return lambda worker: SurvivalView(base(worker), engine, tracker, turnover)

    return base
