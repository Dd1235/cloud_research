from sim.engine import Engine
from sim.policies.lmetric import Lmetric
from sim.request import Request
from sim.workers import Worker


def build_workers(count: int):
    engine = Engine(seed=0)
    return [
        Worker(engine, wid=i, c_prefill=0.01, c_decode=0.01, cache_blocks=16, block_size=16)
        for i in range(count)
    ]


def request(blocks=("a", "b")):
    return Request(id=1, arrival=0.0, prompt_tokens=16 * len(blocks), output_tokens=1, blocks=blocks)


def test_a_full_hit_scores_zero_however_busy_the_worker_is():
    workers = build_workers(2)
    workers[0].cache.insert(("a", "b"), now=0.0)
    workers[0].queue.extend([object()] * 5)

    # warm and busy: 0 new tokens x 6 = 0. cold and idle: 32 x 1 = 32
    assert Lmetric().choose(request(), workers) is workers[0]


def test_a_partial_hit_on_a_busy_worker_loses_to_a_cold_idle_one():
    workers = build_workers(2)
    workers[0].cache.insert(("a",), now=0.0)
    workers[0].queue.extend([object()] * 3)

    # half the prompt cached but 4 in the batch: 16 x 4 = 64 > 32 x 1
    assert Lmetric().choose(request(), workers) is workers[1]
