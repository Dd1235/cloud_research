from sim.engine import Engine
from sim.policies.longest_prefix import LongestPrefix
from sim.request import Request
from sim.workers import Worker


def test_longest_prefix_chooses_best_cache_match():
    engine = Engine(seed=0)
    first = Worker(
        engine,
        wid=0,
        c_prefill=0.01,
        c_decode=0.01,
        cache_blocks=10,
    )
    second = Worker(
        engine,
        wid=1,
        c_prefill=0.01,
        c_decode=0.01,
        cache_blocks=10,
    )

    first.cache.insert(("a", "b"), now=0.0)
    second.cache.insert(("a",), now=0.0)

    req = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=48,
        output_tokens=1,
        blocks=("a", "b", "c"),
    )

    chosen = LongestPrefix().choose(req, [first, second])

    assert chosen is first


def test_longest_prefix_breaks_ties_by_fewer_outstanding():
    engine = Engine(seed=0)
    first = Worker(
        engine,
        wid=0,
        c_prefill=0.01,
        c_decode=0.01,
        cache_blocks=10,
    )
    second = Worker(
        engine,
        wid=1,
        c_prefill=0.01,
        c_decode=0.01,
        cache_blocks=10,
    )

    first.queue.append(object())

    req = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=16,
        output_tokens=1,
        blocks=("a",),
    )

    chosen = LongestPrefix().choose(req, [first, second])

    assert chosen is second

def test_a_match_below_the_threshold_routes_by_load_instead():
    engine = Engine(seed=0)
    workers = [
        Worker(engine, wid=wid, c_prefill=0.01, c_decode=0.01, cache_blocks=10)
        for wid in (0, 1)
    ]

    # worker 0 holds 1 of the request's 4 blocks; worker 1 holds none but has
    # the shorter queue (outstanding counts queued requests, so queue three).
    # sglang's rule: a quarter of the prompt is not worth chasing, go to the
    # shorter queue
    workers[0].cache.insert(("a",), now=0.0)
    workers[0].queue.extend(["q1", "q2", "q3"])

    req = Request(id=1, arrival=0.0, prompt_tokens=64, output_tokens=1, blocks=("a", "b", "c", "d"))

    pure = LongestPrefix()
    guarded = LongestPrefix(match_threshold=0.5)

    assert pure.choose(req, workers) is workers[0]
    assert guarded.choose(req, workers) is workers[1]

    # half the prompt on the shelf clears the 0.5 threshold and the ranker acts
    workers[0].cache.insert(("a", "b"), now=0.0)
    assert guarded.choose(req, workers) is workers[0]
