from sim.engine import Engine
from sim.policies.dualmap import DualMap
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


def test_hot_prefix_spreads_over_at_most_two_workers():
    workers = build_workers(8)
    policy = DualMap()

    # every request shares prefix block ("p", 0, 0) but has its own suffix,
    # so the session key is identical and only the two rings decide placement
    chosen_ids = {
        policy.choose(
            build_request(
                request_id,
                (("p", 0, 0), ("s", request_id, 0)),
            ),
            workers,
        ).id
        for request_id in range(50)
    }

    assert 1 <= len(chosen_ids) <= 2


def test_two_rings_pick_different_workers_for_some_key():
    workers = build_workers(8)
    policy = DualMap()

    # over many distinct prefixes at least one must land on two different
    # workers, otherwise the second ring is not doing anything
    spread_counts = {
        len(
            {
                ring.lookup(f"key:{prefix_id}")
                for ring in policy._build_rings(len(workers))
            }
        )
        for prefix_id in range(50)
    }

    assert 2 in spread_counts


def test_warm_candidate_wins_over_idle_candidate():
    workers = build_workers(8)
    policy = DualMap()

    req = build_request(1, (("p", 0, 0), ("p", 0, 1)))

    first_choice = policy.choose(req, workers)
    first_choice.cache.insert(req.blocks, now=0.0)
    first_choice.queue.extend([object()] * 3)

    # cache match is the first ranking key, so the warm worker keeps winning
    # even though it now has three requests queued and its partner has none
    assert policy.choose(req, workers) is first_choice


def test_placement_is_stable_across_policy_instances():
    workers = build_workers(8)
    req = build_request(1, (("p", 7, 0),))

    first_run = DualMap().choose(req, workers).id
    second_run = DualMap().choose(req, workers).id

    assert first_run == second_run
