from sim.policies.round_robin import RoundRobin


def test_round_robin_cycles_through_workers():
    workers = [object(), object(), object()]
    policy = RoundRobin()

    chosen = [
        policy.choose(None, workers)
        for _ in range(5)
    ]

    assert chosen == [
        workers[0],
        workers[1],
        workers[2],
        workers[0],
        workers[1],
    ]