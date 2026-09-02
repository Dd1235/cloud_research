import pytest

from sim.engine import Engine
from sim.request import Request
from sim.sampler import OutstandingSampler
from sim.workers import Worker


def test_sampler_averages_outstanding_per_worker_over_time():
    engine = Engine(seed=0)
    workers = [
        Worker(engine, wid=worker_id, c_prefill=0.1, c_decode=0.1)
        for worker_id in range(2)
    ]
    sampler = OutstandingSampler(engine, workers, interval=0.05)

    # three 10 token requests queued on worker 0 at t=0, nothing on worker 1;
    # each takes 0.1 * 10 + 0.1 * 1 = 1.1 s, so within 0.1 s all three are still there
    for request_id in range(3):
        workers[0].submit(
            Request(id=request_id, arrival=0.0, prompt_tokens=10, output_tokens=1)
        )

    engine.run(until=0.1)

    # ticks at 0.0, 0.05 and 0.1 all saw [3, 0]
    assert len(sampler.samples) == 3
    assert sampler.mean_outstanding() == [3.0, 0.0]


def test_sampler_can_skip_the_warm_up():
    engine = Engine(seed=0)
    workers = [Worker(engine, wid=0, c_prefill=0.1, c_decode=0.1)]
    sampler = OutstandingSampler(engine, workers, interval=1.0)

    # one request arrives at 1.5 and finishes at 1.5 + 1.1 = 2.6
    engine.schedule(
        1.5,
        workers[0].submit,
        Request(id=1, arrival=1.5, prompt_tokens=10, output_tokens=1),
    )
    engine.run(until=3.0)

    # ticks at 0, 1, 2, 3 saw 0, 0, 1, 0
    assert sampler.mean_outstanding() == pytest.approx(0.25)
    assert sampler.mean_outstanding(since=2.0) == pytest.approx([0.5])
