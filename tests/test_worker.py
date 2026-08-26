import pytest

from sim.engine import Engine
from sim.request import Request
from sim.workers import Worker


def test_worker_records_request_lifecycle():
    engine = Engine(seed=0)
    worker = Worker(
        engine,
        wid=0,
        c_prefill=0.01,
        c_decode=0.02,
    )
    req = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=10,
        output_tokens=3,
    )

    worker.submit(req)
    engine.run()

    assert req.worker_id == 0
    assert req.start == 0.0
    assert req.first_token == pytest.approx(0.10)
    assert req.finish == pytest.approx(0.16)
    assert req.done
    assert req.queue_wait == 0.0
    assert req.ttft == pytest.approx(0.10)
    assert req.tpot == pytest.approx(0.03)


def test_second_request_waits_for_first_request():
    engine = Engine(seed=0)
    worker = Worker(
        engine,
        wid=0,
        c_prefill=0.01,
        c_decode=0.02,
    )

    first = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=10,
        output_tokens=3,
    )
    second = Request(
        id=2,
        arrival=0.0,
        prompt_tokens=5,
        output_tokens=1,
    )

    worker.submit(first)
    worker.submit(second)
    engine.run()

    assert first.finish == pytest.approx(0.16)

    assert second.start == pytest.approx(0.16)
    assert second.queue_wait == pytest.approx(0.16)
    assert second.first_token == pytest.approx(0.21)
    assert second.finish == pytest.approx(0.23)