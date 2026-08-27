from sim.engine import Engine
from sim.policies.least_outstanding import LeastOutstanding
from sim.workers import Worker

def test_least_outstanding_picks_idlest_worker():

    engine = Engine(seed = 0)
    workers = [
        Worker(engine, wid = i, c_prefill=0.01, c_decode=0.01, cache_blocks=16,) for i in range(3)
    ]

    workers[0].queue.append(object())
    workers[1].queue.append(object())

    chosen = LeastOutstanding().choose(None, workers)

    assert chosen is workers[2]