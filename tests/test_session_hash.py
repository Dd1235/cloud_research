from sim.engine import Engine
from sim.policies.session_hash import SessionHash
from sim.request import Request
from sim.workers import Worker


def test_session_hash_keeps_prefix_on_one_worker():
    engine = Engine(seed=0)

    workers = [
        Worker(
            engine,
            wid=i,
            c_prefill=0.01,
            c_decode=0.01,
            cache_blocks=16,
        )
        for i in range(4)
    ]

    policy = SessionHash()

    chosen_ids = {
        policy.choose(
            Request(
                id=request_id,
                arrival=0.0,
                prompt_tokens=16,
                output_tokens=1,
                blocks=(("p", 3, 0), ("s", request_id, 0)),
            ),
            workers,
        ).id
        for request_id in range(20)
    }

    assert len(chosen_ids) == 1