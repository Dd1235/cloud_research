import numpy as np
import pytest

from sim.engine import Engine
from sim.metrics import summarize
from sim.policies.longest_prefix import LongestPrefix
from sim.request import Request
from sim.router import Router
from sim.views import PerfectView
from sim.workers import Worker
from sim.workload import generate


def build_workers(engine, count: int, cache_blocks: int = 10):
    return [
        Worker(
            engine,
            wid=worker_id,
            c_prefill=0.01,
            c_decode=0.01,
            cache_blocks=cache_blocks,
            block_size=16,
        )
        for worker_id in range(count)
    ]


def test_router_installs_a_perfect_view_by_default():
    engine = Engine(seed=0)
    workers = build_workers(engine, 2)

    Router(engine, LongestPrefix(), workers)

    for worker in workers:
        assert isinstance(worker.view, PerfectView)
        assert worker.view.match(("a",)) == worker.cache.match(("a",))


def test_router_records_estimate_truth_and_best_at_dispatch():
    engine = Engine(seed=0)
    workers = build_workers(engine, 2)
    router = Router(engine, LongestPrefix(), workers)

    # only worker 1 is warm for this prefix
    workers[1].cache.insert(("a", "b"), now=0.0)

    req = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=32,
        output_tokens=1,
        blocks=("a", "b"),
    )
    router.dispatch(req)
    engine.run()

    # longest prefix picks the warm worker, and with a perfect view what it
    # promised, what was there, and the best available are all the same 32 tokens
    assert req.worker_id == 1
    assert req.estimated_cached_tokens == 32
    assert req.true_cached_tokens_at_dispatch == 32
    assert req.best_cached_tokens_at_dispatch == 32
    assert req.cached_tokens == 32
    assert router.dispatches == [0, 1]


def test_estimate_is_capped_by_the_prompt_length():
    engine = Engine(seed=0)
    workers = build_workers(engine, 1)
    router = Router(engine, LongestPrefix(), workers)

    workers[0].cache.insert(("a", "b"), now=0.0)

    # two blocks of 16 would be 32 tokens, but the prompt is only 20 tokens long
    req = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=20,
        output_tokens=1,
        blocks=("a", "b"),
    )
    router.dispatch(req)
    engine.run()

    assert req.estimated_cached_tokens == 20
    assert req.cached_tokens == 20


def test_perfect_view_on_an_unpressured_cache_has_no_view_error():
    engine = Engine(seed=0)
    workers = build_workers(engine, 2, cache_blocks=1000)
    router = Router(engine, LongestPrefix(), workers)

    requests = generate(
        np.random.default_rng(0),
        20,
        rate=5.0,
        n_prefixes=2,
        prefix_blocks=4,
        suffix_blocks=(1, 2),
    )
    router.replay(requests)
    engine.run()

    # nothing is ever evicted and the view is the truth, so what was promised
    # is exactly what was there
    for req in requests:
        assert req.estimated_cached_tokens == req.true_cached_tokens_at_dispatch

    # longest prefix picks the argmax of the truth, so the best was always chosen
    for req in requests:
        assert req.true_cached_tokens_at_dispatch == req.best_cached_tokens_at_dispatch

    metrics = summarize(requests, workers, warmup_frac=0.0)
    assert metrics["n"] == 20


def test_recording_estimates_does_not_touch_cache_recency():
    """Instrumentation must be observational.

    The router reads every worker's true cache to record the best achievable
    match. If that read refreshed recency, merely measuring a block would keep
    it from going cold, and the sweep would be measuring itself.
    """
    engine = Engine(seed=0)
    workers = build_workers(engine, 1, cache_blocks=4)
    router = Router(engine, LongestPrefix(), workers)
    cache = workers[0].cache

    cache.insert(("a", "b", "c"), now=1.0)
    cache.insert(("a", "b", "d"), now=2.0)

    # the oracle looks at the older path at t=3 without dispatching anything
    engine.now = 3.0
    probe = Request(
        id=1,
        arrival=3.0,
        prompt_tokens=48,
        output_tokens=1,
        blocks=("a", "b", "c"),
    )
    router._record_estimates(probe, workers[0])
    assert probe.best_cached_tokens_at_dispatch == 48

    # one more leaf forces an eviction: "c" is still the oldest leaf, so it
    # goes, not "d". had the probe refreshed "c", "d" would have been evicted
    cache.insert(("a", "b", "e"), now=4.0)

    assert cache.match(("a", "b", "c")) == 2
    assert cache.match(("a", "b", "d")) == 3


def test_router_records_the_survival_views_expected_tokens():
    from sim.blockrates import BlockRateTracker
    from sim.views import PerfectView, SurvivalView

    engine = Engine(seed=0)
    workers = build_workers(engine, 1)
    tracker = BlockRateTracker(window=100.0)
    router = Router(
        engine,
        LongestPrefix(),
        workers,
        view_factory=lambda worker: SurvivalView(
            PerfectView(worker.cache, engine), engine, tracker, turnover=10.0
        ),
        tracker=tracker,
    )

    workers[0].cache.insert(("a", "b"), now=0.0)

    # at t=5 both blocks are 5 s old and never re-referenced: each survives
    # with probability 1 - 5/10, the path with 0.5 then 0.25, so 0.75 blocks
    engine.now = 5.0
    req = Request(id=1, arrival=5.0, prompt_tokens=32, output_tokens=1, blocks=("a", "b"))
    router.dispatch(req)

    assert req.estimated_cached_tokens == 32
    assert req.expected_cached_tokens == pytest.approx(0.75 * 16)

    # the dispatch itself was observed by the tracker only after the estimate
    assert tracker.rate("a", now=5.0) == pytest.approx(1 / 100)
