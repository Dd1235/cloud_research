from sim.engine import Engine
from sim.policies.dynamo_cost import DynamoCost
from sim.request import Request
from sim.workers import Worker


def build_workers(count: int):
    engine = Engine(seed=0)
    return [
        Worker(engine, wid=i, c_prefill=0.01, c_decode=0.01, cache_blocks=16, block_size=16)
        for i in range(count)
    ]


def request(blocks=("a", "b", "c", "d")):
    return Request(id=1, arrival=0.0, prompt_tokens=16 * len(blocks), output_tokens=1, blocks=blocks)


def test_cost_trades_prefill_blocks_against_active_blocks():
    workers = build_workers(2)
    workers[0].cache.insert(("a", "b", "c", "d"), now=0.0)
    workers[0].queue.extend([object()] * 1)

    # warm with one request: 0 prefill + 1 * 4 active = 4. cold idle: 4 + 0 = 4.
    # a tie, broken by id -> the warm worker
    assert DynamoCost().choose(request(), workers) is workers[0]

    # one more request on the warm worker tips it: 0 + 2 * 4 = 8 > 4
    workers[0].queue.append(object())
    assert DynamoCost().choose(request(), workers) is workers[1]


def test_overlap_weight_scales_how_much_a_hit_is_worth():
    workers = build_workers(2)
    workers[0].cache.insert(("a", "b", "c", "d"), now=0.0)
    workers[0].queue.extend([object()] * 2)

    # with weight 3 the cold worker's prefill costs 12 > the warm one's 8 active
    assert DynamoCost(overlap_weight=3.0).choose(request(), workers) is workers[0]
