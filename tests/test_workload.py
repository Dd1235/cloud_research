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