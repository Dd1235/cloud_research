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

    assert worker.cache.evictions == 2
    assert third.estimated_cached_tokens == 32   # the shadow promised the whole prompt
    assert third.true_cached_tokens_at_dispatch == 0
    assert third.cached_tokens == 0              # the worker had evicted it

    metrics = summarize(requests, [worker], warmup_frac=0.0)

    # 32 falsely promised tokens out of 96 prompt tokens, and since nothing
    # changed between dispatch and admission the execution error is the same
    assert metrics["view_fp_rate"] == pytest.approx(32 / 96)
    assert metrics["execution_fp_rate"] == pytest.approx(32 / 96)
    assert metrics["view_fn_rate"] == 0.0
