import numpy as np
import pytest

from sim.engine import Engine
from sim.policies import POLICIES, make_policy
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


def build_request(request_id: int = 1):
    return Request(
        id=request_id,
        arrival=0.0,
        prompt_tokens=32,
        output_tokens=1,
        blocks=("a", "b"),
    )


@pytest.mark.parametrize("policy_name", list(POLICIES))
def test_every_registered_policy_chooses_a_real_worker(policy_name):
    workers = build_workers(3)

    policy = make_policy(
        policy_name,
        np.random.default_rng(0),
    )

    chosen = policy.choose(build_request(), workers)

    assert chosen in workers


@pytest.mark.parametrize("policy_name", list(POLICIES))
def test_registry_name_matches_policy_name(policy_name):
    policy = make_policy(
        policy_name,
        np.random.default_rng(0),
    )

    assert policy.name == policy_name


def test_unknown_policy_name_is_rejected():
    with pytest.raises(KeyError):
        make_policy("no_such_policy", np.random.default_rng(0))
