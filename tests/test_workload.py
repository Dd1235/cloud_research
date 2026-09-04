import numpy as np

from sim.workload import generate


def test_workload_is_seeded_and_has_shared_prefixes():
    kwargs = dict(
        n_requests=3,
        rate=1.0,
        n_prefixes=1,
        prefix_blocks=2,
        suffix_blocks=(1, 1),
        output_tokens=(5, 5),
        block_size=16,
    )

    first = generate(np.random.default_rng(7), **kwargs)
    second = generate(np.random.default_rng(7), **kwargs)

    assert first == second
    assert [req.arrival for req in first] == sorted(
        req.arrival for req in first
    )

    for req in first:
        assert req.blocks[:2] == (
            ("p", 0, 0),
            ("p", 0, 1),
        )
        assert req.prompt_tokens == 48
        assert req.output_tokens == 5

def test_universal_blocks_begin_every_request_like_a_shared_system_prompt():
    requests = generate(
        np.random.default_rng(0),
        20,
        rate=1.0,
        universal_blocks=3,
        prefix_blocks=2,
    )

    universal_prefix = (("u", 0), ("u", 1), ("u", 2))
    for req in requests:
        assert req.blocks[:3] == universal_prefix
        # the shared prefix follows the universal one, so every request has
        # exactly one distinct first block, the trace pathology on demand
        assert req.blocks[3][0] == "p"
        assert req.prompt_tokens == len(req.blocks) * 16
