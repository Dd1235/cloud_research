import numpy as np

from sim.engine import Engine
from sim.policies.power_of_two import PowerOfTwo
from sim.workers import Worker


# random sample two workers, and choose hte less loaded one, instead of least-outstanding globally best
# just try with two

def test_power_of_two_is_seeded_and_avoids_loaded_worker():
    engine = Engine(seed=0)
    workers = [
        Worker(
            engine,
            wid=i,
            c_prefill=0.01,
            c_decode=0.01,
            cache_blocks=16,
        )
        for i in range(4)
    ]

    workers[0].queue.extend([object()] * 5)

    first = PowerOfTwo(np.random.default_rng(1))
    second = PowerOfTwo(np.random.default_rng(1))

    assert first.choose(None, workers).id == second.choose(None, workers).id

    policy = PowerOfTwo(np.random.default_rng(2))
    assert all(
        policy.choose(None, workers).id != 0
        for _ in range(50)
    )