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