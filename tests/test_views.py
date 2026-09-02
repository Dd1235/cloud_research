import pytest

from sim.engine import Engine
from sim.metrics import summarize
from sim.policies.longest_prefix import LongestPrefix
from sim.request import Request
from sim.router import Router
from sim.views import SnapshotView, make_view_factory
from sim.workers import Worker


def build_worker(engine, worker_id: int = 0, output_cost: float = 0.01):
    return Worker(
        engine,
        wid=worker_id,
        c_prefill=0.01,
        c_decode=output_cost,
        cache_blocks=16,
        block_size=16,
    )


def build_request(request_id: int, arrival: float, output_tokens: int = 1):
    return Request(
        id=request_id,
        arrival=arrival,
        prompt_tokens=32,
        output_tokens=output_tokens,
        blocks=("a", "b"),
    )


def test_snapshot_view_does_not_see_an_insert_until_the_next_refresh():
    engine = Engine(seed=0)
    worker = build_worker(engine)
    view = SnapshotView(engine, worker.cache, period=1.0)

    # the worker learns the prefix at 0.2, between the refreshes at 0 and 1
    engine.schedule(0.2, worker.submit, build_request(1, arrival=0.2))

    engine.run(until=0.5)
    assert worker.cache.match(("a", "b")) == 2
    assert view.match(("a", "b")) == 0
    assert view.refreshes == 1

    engine.run(until=1.0)
    assert view.match(("a", "b")) == 2
    assert view.refreshes == 2
    assert view.age == 0.0


def test_snapshot_view_is_empty_before_its_first_refresh():
    engine = Engine(seed=0)
    worker = build_worker(engine)
    worker.cache.insert(("a", "b"), now=0.0)

    view = SnapshotView(engine, worker.cache, period=1.0, phase=2.0)

    assert view.match(("a", "b")) == 0
    assert view.age == float("inf")

    engine.run(until=2.0)
    assert view.match(("a", "b")) == 2


def test_stale_view_sends_a_warm_prefix_to_a_cold_worker():
    def reuse_with(view_kind):
        engine = Engine(seed=0)
        workers = [build_worker(engine, worker_id) for worker_id in range(2)]
        router = Router(
            engine,
            LongestPrefix(),
            workers,
            make_view_factory(view_kind, engine, period=30.0),
        )

        # the first request keeps worker 0 busy for ~1 s; the second arrives
        # while it runs, so the choice is "warm but busy" vs "cold and idle"
        router.replay(
            [
                build_request(1, arrival=0.0, output_tokens=100),
                build_request(2, arrival=0.5),
            ]
        )
        engine.run()

        return sum(worker.tokens_reused for worker in workers)

    fresh_reuse = reuse_with("perfect")
    stale_reuse = reuse_with("snapshot")

    # with the truth, longest prefix waits for the warm worker and reuses the
    # prefix; with a 30 s old (empty) snapshot both workers look cold and the
    # tie goes to the idle one, which has to prefill from scratch
    assert fresh_reuse == 32
    assert stale_reuse == 0


def test_shadow_view_reports_a_block_the_worker_already_evicted():
    engine = Engine(seed=0)
    worker = Worker(
        engine,
        wid=0,
        c_prefill=0.01,
        c_decode=0.01,
        cache_blocks=2,
        block_size=16,
    )
    router = Router(
        engine,
        LongestPrefix(),
        [worker],
        make_view_factory("shadow", engine, shadow_blocks=100),
    )

    def request_for(request_id: int, arrival: float, blocks):
        return Request(
            id=request_id,
            arrival=arrival,
            prompt_tokens=32,
            output_tokens=1,
            blocks=blocks,
        )

    # a two-block cache: the second request evicts the first request's blocks
    # from the worker, but the router's own index still remembers routing them
    requests = [
        request_for(1, 0.0, ("a", "b")),
        request_for(2, 1.0, ("c", "d")),
        request_for(3, 2.0, ("a", "b")),
    ]
    router.replay(requests)
    engine.run()

    third = requests[2]

    # (a,b) evicted by (c,d), then (c,d) evicted again when (a,b) comes back
    assert worker.cache.evictions == 4
    assert third.estimated_cached_tokens == 32   # the shadow promised the whole prompt
    assert third.true_cached_tokens_at_dispatch == 0
    assert third.cached_tokens == 0              # the worker had evicted it

    metrics = summarize(requests, [worker], warmup_frac=0.0)

    # 32 falsely promised tokens out of 96 prompt tokens, and since nothing
    # changed between dispatch and admission the execution error is the same
    assert metrics["view_fp_rate"] == pytest.approx(32 / 96)
    assert metrics["execution_fp_rate"] == pytest.approx(32 / 96)
    assert metrics["view_fn_rate"] == 0.0


def test_same_instant_arrivals_are_routed_before_the_iteration_they_trigger():
    """Two requests at the same timestamp: what does the second one see?

    Both dispatches are scheduled before the run starts, so they carry lower
    sequence numbers than the worker iteration the first one kicks off. The
    router therefore sees both before the worker admits either. With the
    perfect view the second request finds nothing cached yet, because the
    first has not been admitted; the shadow view already recorded the first
    dispatch and promises the prefix. At admission both land in the same
    iteration and the second really does reuse the first's prefill.
    """
    from sim.batched_worker import BatchedWorker

    def second_request_with(view_kind):
        engine = Engine(seed=0)
        worker = BatchedWorker(
            engine,
            wid=0,
            c_prefill=0.01,
            c_decode=0.002,
            c_iter=0.018,
            cache_blocks=16,
            block_size=16,
        )
        router = Router(
            engine,
            LongestPrefix(),
            [worker],
            make_view_factory(view_kind, engine, shadow_blocks=16),
        )

        first = build_request(1, arrival=0.0)
        second = build_request(2, arrival=0.0)
        router.replay([first, second])
        engine.run()

        assert first.first_token == second.first_token   # same iteration
        return second

    with_perfect = second_request_with("perfect")
    assert with_perfect.estimated_cached_tokens == 0
    assert with_perfect.true_cached_tokens_at_dispatch == 0
    assert with_perfect.cached_tokens == 32   # admitted together, reuse at execution

    with_shadow = second_request_with("shadow")
    assert with_shadow.estimated_cached_tokens == 32   # the shadow saw the first dispatch
    assert with_shadow.true_cached_tokens_at_dispatch == 0
    assert with_shadow.cached_tokens == 32


def test_ttl_view_stops_trusting_a_block_once_its_last_access_is_older_than_the_ttl():
    from sim.views import PerfectView, TtlView

    engine = Engine(seed=0)
    worker = build_worker(engine)
    worker.cache.insert(("a", "b", "c"), now=0.0)
    worker.cache.insert(("a",), now=8.0)        # "a" refreshed, "b" and "c" not

    view = TtlView(PerfectView(worker.cache, engine), ttl=10.0)

    engine.now = 5.0
    assert view.match(("a", "b", "c")) == 3     # all three touched within 10 s

    engine.now = 12.0
    # "a" is 4 s old and trusted; "b" is 12 s old and cut, and so is "c"
    # behind it even though its own age would not matter for a contiguous prefix
    assert view.match(("a", "b", "c")) == 1
    assert view.match_with_ages(("a", "b", "c")) == [4.0]

    # the truth still holds all three: the ttl view is deliberately pessimistic
    assert worker.cache.match(("a", "b", "c")) == 3


def test_ttl_wrapped_shadow_forgets_a_dispatch_after_the_ttl():
    from sim.views import ShadowView, TtlView

    engine = Engine(seed=0)
    view = TtlView(ShadowView(100, engine), ttl=10.0)

    view.record_dispatch(("a", "b"), now=0.0)

    engine.now = 5.0
    assert view.match(("a", "b")) == 2

    engine.now = 15.0
    assert view.match(("a", "b")) == 0


def test_survival_view_trusts_young_blocks_and_rescues_old_hot_ones():
    import math

    from sim.blockrates import BlockRateTracker
    from sim.views import SnapshotView, SurvivalView

    engine = Engine(seed=0)
    worker = build_worker(engine)
    worker.cache.insert(("hot", "cold"), now=0.0)

    tracker = BlockRateTracker(window=100.0)
    for reference_time in range(0, 100, 5):      # "hot" is referenced every 5 s
        tracker.observe(("hot",), now=float(reference_time))

    snapshot = SnapshotView(engine, worker.cache, period=10.0)
    engine.run(until=0.0)                        # take the scrape at t=0
    engine.now = 5.0                             # the scrape is now 5 s old

    # both blocks have a known age of 5 s. against a 20 s turnover that is
    # young: trusted outright, expectation equals the raw match
    young = SurvivalView(snapshot, engine, tracker, turnover=20.0)
    assert young.match_expected(("hot", "cold")) == pytest.approx(2.0)

    # against a 4 s turnover both are past their lifetime. "hot" may have been
    # re-referenced during the 5 s the scrape is old: 1 - exp(-0.2 * 5). "cold"
    # never is, so the path ends there
    old = SurvivalView(snapshot, engine, tracker, turnover=4.0)
    assert old.match_expected(("hot", "cold")) == pytest.approx(1.0 - math.exp(-1.0))

    # and the ordinal match is untouched: rankers never see the discount
    assert old.match(("hot", "cold")) == 2


def test_survival_view_with_a_residence_cdf_discounts_blocks_that_usually_die_young():
    from sim.blockrates import BlockRateTracker
    from sim.views import PerfectView, SurvivalView

    engine = Engine(seed=0)
    worker = build_worker(engine)
    worker.cache.insert(("a", "b"), now=0.0)
    tracker = BlockRateTracker(window=100.0)     # nothing observed: no rescue
    engine.now = 6.0

    # a cache whose blocks are all evicted between 4 and 8 s of idleness
    def residence_cdf(idle_age):
        return min(max((idle_age - 4.0) / 4.0, 0.0), 1.0)

    step = SurvivalView(PerfectView(worker.cache, engine), engine, tracker, turnover=14.0)
    curve = SurvivalView(PerfectView(worker.cache, engine), engine, tracker, turnover=14.0,
                         residence_cdf=residence_cdf)

    # the step trusts a 6 s old block completely; the curve says half of such
    # blocks are already gone. the two losses are nested, not independent: the
    # leaf b goes first and a only after it, so a is there whenever b is, and
    # the expected depth is the sum of the two marginals, 0.5 + 0.5, not the
    # 0.5 + 0.25 that treating them as independent coin flips would give
    assert step.match_expected(("a", "b")) == pytest.approx(2.0)
    assert curve.match_expected(("a", "b")) == pytest.approx(1.0)


def test_residence_cdf_is_applied_as_a_hazard_over_the_scrape_interval():
    from sim.blockrates import BlockRateTracker
    from sim.views import SnapshotView, SurvivalView

    engine = Engine(seed=0)
    worker = build_worker(engine)
    worker.cache.insert(("a",), now=0.0)
    tracker = BlockRateTracker(window=100.0)     # never referenced: no rescue

    # evictions spread uniformly over idle ages 0..10 s
    def residence_cdf(idle_age):
        return min(max(idle_age / 10.0, 0.0), 1.0)

    snapshot = SnapshotView(engine, worker.cache, period=100.0)
    engine.run(until=0.0)                        # scrape at t=0, block idle age 0
    engine.now = 4.0                             # scrape is 4 s old, block idle 4 s

    view = SurvivalView(snapshot, engine, tracker, turnover=10.0, residence_cdf=residence_cdf)

    # the block was idle 0 s when scraped and is idle 4 s now: the hazard over
    # [0, 4] is (F(4) - F(0)) / (1 - F(0)) = 0.4, so 0.6 of it is expected
    assert view.match_expected(("a",)) == pytest.approx(0.6)

    # a block that was already idle 6 s when scraped and is idle 10 s now has
    # hazard (F(10) - F(6)) / (1 - F(6)) = 0.4 / 0.4 = 1: certainly gone
    assert view.gone_if_not_refreshed(10.0, scrape_age=4.0) == pytest.approx(1.0)
    # while F(10) alone would also say gone, F(4) alone would have said 0.4 for
    # the first block instead of the hazard's 0.4 -- they agree only from idle 0
    assert view.gone_if_not_refreshed(4.0, scrape_age=0.0) == pytest.approx(0.4)


def test_snapshot_overlay_sees_the_routers_own_dispatches_until_the_next_refresh():
    from sim.views import SnapshotView

    engine = Engine(seed=0)
    worker = build_worker(engine)
    plain = SnapshotView(engine, worker.cache, period=10.0)
    overlaid = SnapshotView(engine, worker.cache, period=10.0, overlay=True)
    engine.run(until=0.0)   # first scrape of an empty cache

    # the router sends a and b at t=1. until the next scrape the plain
    # snapshot cannot know, the overlay knows at once, with dispatch-time ages
    engine.now = 1.0
    for view in (plain, overlaid):
        view.record_dispatch(("a", "b"), now=1.0)
    engine.now = 3.0

    assert plain.match(("a", "b")) == 0
    assert overlaid.match(("a", "b")) == 2
    assert overlaid.match_with_ages(("a", "b")) == [pytest.approx(2.0), pytest.approx(2.0)]

    # the worker kept a and b but never admitted z. the scrape at t=10 shows
    # a and b in the copy and drops the overlay, so z is forgotten with it
    worker.cache.insert(("a", "b"), now=3.0)
    overlaid.record_dispatch(("z",), now=3.0)
    assert overlaid.match(("z",)) == 1

    engine.run(until=10.0)

    assert overlaid.refreshes == 2
    assert overlaid.match(("a", "b")) == 2
    assert overlaid.match(("z",)) == 0
