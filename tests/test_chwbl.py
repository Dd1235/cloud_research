from sim.engine import Engine
from sim.policies.chwbl import Chwbl
from sim.request import Request
from sim.workers import Worker


def build_workers(count: int):
    engine = Engine(seed=0)

    return [
        Worker(
            engine,
            wid=worker_id,
            c_prefill=0.01,
            c_decode=0.01,
            cache_blocks=16,
        )
        for worker_id in range(count)
    ]


def build_request(request_id: int, blocks: tuple):
    return Request(
        id=request_id,
        arrival=0.0,
        prompt_tokens=16 * len(blocks),
        output_tokens=1,
        blocks=blocks,
    )


def test_requests_sharing_a_prefix_land_on_the_same_worker_when_loads_are_equal():
    workers = build_workers(4)
    policy = Chwbl()

    # same first block, different suffixes: the one-block session key is
    # identical, and with every worker idle the bound is ceil(1.25 * (0 + 1)) = 2
    # while the owner would carry 0 + 1 = 1, so the ring's owner always wins
    first = build_request(1, (("p", 0, 0), ("s", 1, 0)))
    second = build_request(2, (("p", 0, 0), ("s", 2, 0)))

    chosen_for_first = policy.choose(first, workers)
    chosen_for_second = policy.choose(second, workers)

    assert chosen_for_first is chosen_for_second

    # and the answer does not drift when the same request is asked again, nor
    # when a fresh policy instance rebuilds the ring from scratch
    assert policy.choose(first, workers) is chosen_for_first
    assert Chwbl().choose(first, workers) is chosen_for_first


def test_an_owner_over_the_bound_hands_the_request_to_another_worker():
    workers = build_workers(3)
    policy = Chwbl(load_bound=1.25)

    req = build_request(1, (("p", 0, 0), ("s", 1, 0)))

    # with all three idle the mean is 0, the bound is ceil(1.25 * (0 + 1)) = 2,
    # and 0 + 1 = 1 fits, so this is the pure affinity choice
    affinity_choice = policy.choose(req, workers)

    # now bury the owner: outstanding becomes 6, 0, 0, so the mean is 2 and the
    # bound is ceil(1.25 * (2 + 1)) = ceil(3.75) = 4. the owner would land at
    # 6 + 1 = 7 > 4 and is refused; the next worker the walk reaches sits at
    # 0 + 1 = 1 <= 4 and takes it
    affinity_choice.queue.extend([object()] * 6)

    assert affinity_choice.outstanding == 6
    assert sum(worker.outstanding for worker in workers) == 6

    overflow_choice = policy.choose(req, workers)

    assert overflow_choice is not affinity_choice
    assert overflow_choice.outstanding == 0


def test_a_bound_no_worker_can_meet_falls_back_to_the_least_loaded():
    workers = build_workers(3)
    policy = Chwbl(load_bound=0.5)

    req = build_request(1, (("p", 0, 0), ("s", 1, 0)))

    # equal and high: outstanding 5, 5, 5 gives mean 5 and a bound of
    # ceil(0.5 * (5 + 1)) = 3, which 5 + 1 = 6 blows past on every worker. the
    # walk runs out of probes and the request still has to go somewhere
    for worker in workers:
        worker.queue.extend([object()] * 5)

    chosen = policy.choose(req, workers)

    assert chosen in workers
    assert chosen.outstanding == 5

    # unequal and still unsatisfiable: 5, 5, 4 gives mean 14 / 3 = 4.667 and a
    # bound of ceil(0.5 * 5.667) = ceil(2.83) = 3, so 6, 6 and 5 are all over it
    # and the fallback has to name the genuinely least loaded worker
    workers[2].queue.pop()

    assert [worker.outstanding for worker in workers] == [5, 5, 4]

    assert policy.choose(req, workers) is workers[2]
