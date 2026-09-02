from pathlib import Path

from sim.traces import load_mooncake


FIXTURE = Path(__file__).parent / "fixtures" / "mooncake_tiny.jsonl"


def test_mooncake_loader_maps_fields_and_blocks():
    requests = load_mooncake(FIXTURE)

    assert [request.arrival for request in requests] == [0.0, 0.5, 2.0]
    assert requests[0].blocks == (("m", 1), ("m", 2))
    assert requests[1].output_tokens == 1
    assert requests[2].blocks == ()
    assert [request.prompt_tokens for request in requests] == [1000, 1500, 300]
    assert [request.id for request in requests] == [0, 1, 2]


def test_speedup_compresses_arrivals_and_limit_truncates():
    speedup_requests = load_mooncake(FIXTURE, speedup=2.0)
    limited_requests = load_mooncake(FIXTURE, limit=2)

    assert [request.arrival for request in speedup_requests] == [0.0, 0.25, 1.0]
    assert len(limited_requests) == 2


def test_replaying_the_fixture_reuses_shared_blocks_at_512_tokens():
    from sim.engine import Engine
    from sim.metrics import summarize
    from sim.policies.longest_prefix import LongestPrefix
    from sim.router import Router
    from sim.traces import MOONCAKE_BLOCK_SIZE
    from sim.workers import Worker

    engine = Engine(seed=0)
    worker = Worker(
        engine,
        wid=0,
        c_prefill=0.001,
        c_decode=0.001,
        cache_blocks=16,
        block_size=MOONCAKE_BLOCK_SIZE,
    )
    router = Router(engine, LongestPrefix(), [worker])

    requests = load_mooncake(FIXTURE)
    router.replay(requests)
    engine.run()

    # request 2 shares blocks 1 and 2 with request 1: 2 * 512 = 1024 of its 1500
    # tokens are served from the cache; the other two requests reuse nothing
    assert requests[1].cached_tokens == 1024
    assert summarize(requests, [worker], warmup_frac=0.0)["hit_rate"] == 1024 / 2800
