from sim.engine import Engine
from sim.policies.hybrid import Hybrid
from sim.request import Request
from sim.workers import Worker


def test_hybrid_extremes_reduce_to_known_policies():
    engine = Engine(seed=0)

    workers = [
        Worker(
            engine,
            wid=i,
            c_prefill=0.01,
            c_decode=0.01,
            cache_blocks=16,
        )
        for i in range(2)
    ]

    workers[0].cache.insert(("a", "b"), now=0.0)
    workers[0].queue.extend([object()] * 3)

    req = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=32,
        output_tokens=1,
        blocks=("a", "b"),
    )

    assert Hybrid(alpha=1.0, beta=0.0).choose(req, workers) is workers[0]
    assert Hybrid(alpha=0.0, beta=1.0).choose(req, workers) is workers[1]

def test_expected_overlap_source_lets_a_stale_promise_lose_to_load():
    from sim.blockrates import BlockRateTracker
    from sim.views import PerfectView, SurvivalView

    engine = Engine(seed=0)
    workers = [
        Worker(engine, wid=i, c_prefill=0.01, c_decode=0.01, cache_blocks=16)
        for i in range(2)
    ]

    # worker 0 holds the prefix but has not touched it for a whole cache
    # lifetime, and it is busy; worker 1 is cold and idle
    workers[0].cache.insert(("a", "b"), now=0.0)
    workers[0].queue.extend([object()] * 3)

    tracker = BlockRateTracker(window=100.0)
    for worker in workers:
        worker.view = SurvivalView(PerfectView(worker.cache, engine), engine, tracker, turnover=10.0)
    engine.now = 10.0

    req = Request(id=1, arrival=10.0, prompt_tokens=32, output_tokens=1, blocks=("a", "b"))

    # the raw promise is a full overlap, which outvotes the load term
    assert Hybrid(overlap_source="raw").choose(req, workers) is workers[0]
    # the expectation of that promise is ~0 at one turnover of age, so load decides
    assert Hybrid(overlap_source="expected").choose(req, workers) is workers[1]
