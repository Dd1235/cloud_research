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