import pytest

from sim.batched_worker import BatchedWorker
from sim.engine import Engine
from sim.request import Request
from sim.workers import Worker


# chosen so the arithmetic in these tests is easy to do by hand:
# an empty iteration costs 0.018, each prompt token 0.01, each decode step 0.002
C_ITER = 0.018
C_PREFILL = 0.01
C_DECODE = 0.002


def build_worker(engine, **overrides):
    settings = dict(
        c_prefill=C_PREFILL,
        c_decode=C_DECODE,
        c_iter=C_ITER,
        cache_blocks=64,
    )
    settings.update(overrides)

    return BatchedWorker(engine, wid=0, **settings)


def build_request(request_id: int, prompt_tokens: int, output_tokens: int, blocks=()):
    return Request(
        id=request_id,
        arrival=0.0,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        blocks=blocks,
    )


def test_single_request_timeline_is_one_prefill_then_one_decode_per_token():
    engine = Engine(seed=0)
    worker = build_worker(engine)

    req = build_request(1, prompt_tokens=10, output_tokens=3)
    worker.submit(req)
    engine.run()

    # iteration 1 prefills all 10 tokens and emits the first token:
    #   0.018 + 0.01 * 10 = 0.118
    assert req.first_token == pytest.approx(0.118)

    # then one decode iteration per remaining token, each 0.018 + 0.002 = 0.020
    assert req.finish == pytest.approx(0.118 + 2 * 0.020)
    assert worker.iterations == 3
    assert worker.completed == 1


def test_requests_arriving_together_share_the_same_iteration():
    engine = Engine(seed=0)
    worker = build_worker(engine)

    first = build_request(1, prompt_tokens=10, output_tokens=5)
    second = build_request(2, prompt_tokens=10, output_tokens=5)

    worker.submit(first)
    worker.submit(second)
    engine.run()

    # this is what the delay-0 kick buys: both prompts are prefilled in the same
    # forward pass, 0.018 + 0.01 * 20 = 0.218, rather than one waiting for the other
    assert first.first_token == pytest.approx(0.218)
    assert second.first_token == pytest.approx(0.218)
    assert first.start == second.start == 0.0


def test_batching_beats_running_the_same_requests_sequentially():
    def finish_time_with(worker_factory):
        engine = Engine(seed=0)
        worker = worker_factory(engine)

        requests = [
            build_request(i, prompt_tokens=10, output_tokens=50)
            for i in range(2)
        ]
        for req in requests:
            worker.submit(req)

        engine.run()

        return max(req.finish for req in requests)

    # the sequential worker is given c_decode = c_iter + c_decode, so a single
    # stream costs exactly the same in both models and only batching differs
    sequential_finish = finish_time_with(
        lambda engine: Worker(
            engine,
            wid=0,
            c_prefill=C_PREFILL,
            c_decode=C_ITER + C_DECODE,
            cache_blocks=64,
        )
    )
    batched_finish = finish_time_with(build_worker)

    assert batched_finish < 0.7 * sequential_finish


def test_per_token_cost_grows_with_batch_size():
    engine = Engine(seed=0)
    worker = build_worker(engine)

    requests = [
        build_request(i, prompt_tokens=10, output_tokens=20)
        for i in range(2)
    ]
    for req in requests:
        worker.submit(req)

    engine.run()

    # two sequences decoding together means each iteration pays 2 * c_decode,
    # so throughput improves but per-token latency gets worse. that is the
    # batching tradeoff, and it is why TPOT is reported next to TTFT
    assert requests[0].tpot == pytest.approx(C_ITER + 2 * C_DECODE)


def test_request_arriving_mid_iteration_waits_for_the_boundary():
    engine = Engine(seed=0)
    worker = build_worker(engine)

    # a long prompt makes iteration 1 last 0.018 + 0.01 * 100 = 1.018
    long_request = build_request(1, prompt_tokens=100, output_tokens=2)
    late_request = build_request(2, prompt_tokens=10, output_tokens=1)

    worker.submit(long_request)
    engine.schedule(0.5, worker.submit, late_request)
    engine.run()

    # it arrived at 0.5, mid iteration, so it is admitted at the next boundary
    assert late_request.start == pytest.approx(1.018)


def test_cached_prefix_removes_the_prefill_cost():
    engine = Engine(seed=0)
    worker = build_worker(engine, block_size=16)

    first = build_request(1, prompt_tokens=32, output_tokens=1, blocks=("x", "y"))
    worker.submit(first)
    engine.run()

    second = build_request(2, prompt_tokens=32, output_tokens=1, blocks=("x", "y"))
    second.arrival = engine.now
    worker.submit(second)
    engine.run()

    assert second.cached_tokens == 32
    # nothing left to prefill, but the first token still costs one forward pass
    assert second.first_token - second.start == pytest.approx(C_ITER + C_DECODE)


def test_max_batch_limits_how_many_run_together():
    engine = Engine(seed=0)
    worker = build_worker(engine, max_batch=2)

    for request_id in range(5):
        worker.submit(
            build_request(request_id, prompt_tokens=10, output_tokens=2)
        )

    # the step runs at delay 0, so advance the clock just past it
    engine.run(until=0.0)

    assert len(worker.running) == 2
    assert len(worker.queue) == 3
