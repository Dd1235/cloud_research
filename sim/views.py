import inspect
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


# an overlay only has to hold one refresh period of dispatches to one worker
OVERLAY_BLOCKS = 1 << 16


class SnapshotView(CacheView):
    """View model A: a copy of the true cache taken every `period` seconds.

    Between refreshes the router routes on a picture that is anywhere from 0
    to one period old, mean period / 2. That is exactly what a metrics scrape
    gives a production router, and what the mac cluster will do for real.

    The refresh is a daemon event: it reschedules itself forever, so it must
    never be the thing keeping a run alive.
    """

    def __init__(self, engine, cache: PrefixCache, period: float, phase: float = 0.0,
                 overlay: bool = False):
        assert period > 0

        self._engine = engine
        self._cache = cache
        self._period = period

        # nothing scraped yet: match() reports an empty cache
        self._snapshot = None
        self.taken_at = None
        self.refreshes = 0

        # the blind spot of a scrape is everything the router itself sent
        # since. an overlay remembers those dispatches until the next refresh
        # shows them in the copy, so the view is never blind to its own
        # decisions. llm-d calls these speculative entries
        self._overlay = PrefixCache(OVERLAY_BLOCKS) if overlay else None

        engine.schedule(phase, self._refresh, daemon=True)

    def _refresh(self) -> None:
        self._snapshot = self._cache.copy()
        self.taken_at = self._engine.now
        self.refreshes += 1

        # whatever the worker really kept of the overlay is in the copy now;
        # a dispatch it never admitted before the scrape is a small blind
        # spot that the next refresh closes
        if self._overlay is not None:
            self._overlay = PrefixCache(OVERLAY_BLOCKS)

        self._engine.schedule(self._period, self._refresh, daemon=True)

    def record_dispatch(self, blocks, now: float) -> None:
        if self._overlay is not None:
            self._overlay.insert(blocks, now=now)

    @property
    def age(self) -> float:
        if self.taken_at is None:
            return float("inf")

        return self._engine.now - self.taken_at

    def match(self, blocks) -> int:
        scraped = 0 if self._snapshot is None else self._snapshot.match(blocks)
        dispatched = 0 if self._overlay is None else self._overlay.match(blocks)

        return max(scraped, dispatched)

    def match_with_ages(self, blocks) -> list[float]:
        now = self._engine.now

        # ages are measured against now, so a block's age includes the time
        # since the scrape; a re-reference since then is invisible to the
        # copy, though not to the overlay, whose ages are times since dispatch
        scraped = [] if self._snapshot is None else self._snapshot.match_with_ages(blocks, now)
        dispatched = [] if self._overlay is None else self._overlay.match_with_ages(blocks, now)

        return scraped if len(scraped) >= len(dispatched) else dispatched


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


def _with_depth(residence_cdf):
    """Normalise a residence cdf to the two-argument form (idle_age, depth).

    A cdf measured on all evictions together ignores depth; one measured per
    generation wants the block's position along the match. Accepting both
    keeps every earlier caller and calibration file working.
    """
    if residence_cdf is None:
        return None

    try:
        wants_depth = len(inspect.signature(residence_cdf).parameters) >= 2
    except (TypeError, ValueError):
        wants_depth = False

    if wants_depth:
        return residence_cdf

    return lambda idle_age, depth: residence_cdf(idle_age)


class SurvivalView(CacheView):
    """Weight each block the view reports by the chance it is really still there.

    Che's model: a block lives a characteristic time T_C after its last access.
    The view knows each block's last access as of its own scrape, so a block
    whose known age has reached T_C is gone unless something the view could not
    see happened: a re-reference during the scrape's age a_s, probability
    1 - exp(-rate * a_s). Hence

        S(block) = 1 - exp(-rate * a_s) * [known age >= T_C]

    A block younger than T_C is trusted outright. The expected surviving prefix
    match is the sum over depth of the product of the survivals along the path,
    because a prefix only survives to depth j if every block before it did.
    match() itself stays ordinal and unchanged; only scorers that add overlap to
    a load term need the expectation.

    The earlier draft used a linear ramp age / T_C for the eviction chance; that
    over-predicted false positives by 3x against measurement, because the ramp
    is the right answer only when the block's access time is unknown.

    The step at T_C is Che's idealisation. A radix cache that evicts leaves
    first spreads its residence times well below T_C (measured p10 5.6 s to p90
    13 s against a 14 s turnover), so the step under-predicts. residence_cdf,
    when given, replaces the step with P[evicted by this idle age]; the worker
    can report that curve as a small histogram, or a calibration run can fit it.
    """

    def __init__(self, inner: CacheView, engine, tracker, turnover: float, residence_cdf=None,
                 worker_id=None):
        assert turnover > 0

        self._inner = inner
        self._engine = engine
        self._tracker = tracker
        self.turnover = turnover
        self._residence_cdf = _with_depth(residence_cdf)
        # the rescue asks whether *this* worker saw the block again, so rates are
        # read per worker when the tracker was fed per worker
        self._worker_id = worker_id

    def gone_if_not_refreshed(self, known_age: float, scrape_age: float, depth: int = 1) -> float:
        """P[evicted by now | the view saw it present scrape_age ago, idle since].

        With a residence cdf F this is the hazard over the scrape interval,
        (F(x) - F(x - a)) / (1 - F(x - a)) for known idle age x and scrape age a:
        the block had already survived to x - a when the view saw it, so only
        evictions inside the interval count. Using F(x) outright over-predicted
        false positives 10-40x for fresh scrapes. A view with no scrape age (a
        record-insert index) has not observed survival, so F(x) is right there.
        Che's deterministic lifetime is the same rule with a step for F.
        """
        if self._residence_cdf is None:
            return 1.0 if known_age >= self.turnover else 0.0

        gone_by_now = self._residence_cdf(known_age, depth)

        if scrape_age <= 0.0 or scrape_age == float("inf"):
            return gone_by_now

        survived_to_scrape = 1.0 - self._residence_cdf(known_age - scrape_age, depth)
        if survived_to_scrape <= 0.0:
            return 1.0

        return min(max((gone_by_now - (1.0 - survived_to_scrape)) / survived_to_scrape, 0.0), 1.0)

    def survival(self, block, known_age: float, depth: int = 1) -> float:
        scrape_age = self._inner.age
        gone = self.gone_if_not_refreshed(known_age, scrape_age, depth)
        if gone == 0.0:
            return 1.0

        if scrape_age == float("inf"):
            return 1.0 - gone

        rate = self._tracker.rate(block, self._engine.now, self._worker_id)
        return 1.0 - math.exp(-rate * scrape_age) * gone

    def match_expected(self, blocks) -> float:
        """Expected surviving match depth, sum over j of P[block j still present].

        Losses along a matched path are nested, not independent: a radix cache
        evicts leaves first, so a block is only ever evicted after every block
        below it, and "block j is present" already implies its ancestors are.
        The expected depth is therefore the sum of the marginal survivals,
        clamped so no block is trusted more than the one above it. Multiplying
        the survivals instead, as if each block could vanish on its own, turns
        a 3% risk per block into a 27% loss over a 20-block path, which is the
        10x over-prediction the theory check found.
        """
        expected_depth = 0.0
        path_survival = 1.0

        for position, (block, known_age) in enumerate(zip(blocks, self._inner.match_with_ages(blocks))):
            # the first matched block sits one edge from the root, so its
            # depth, the generation the residence cdf may condition on, is
            # position + 1
            path_survival = min(path_survival, self.survival(block, known_age, position + 1))
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
                      tracker=None, turnover=None, residence_cdf=None, overlay=False):
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
        base = lambda worker: SnapshotView(engine, worker.cache, period, overlay=overlay)
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
        return lambda worker: SurvivalView(base(worker), engine, tracker, turnover, residence_cdf,
                                           worker_id=worker.id)

    return base
