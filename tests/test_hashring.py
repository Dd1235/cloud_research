import numpy as np

from sim.policies.hashring import HashRing



def test_hash_ring_is_stable_and_reasonably_balanced():
    first = HashRing(4, vnodes=64)
    second = HashRing(4, vnodes=64)

    assert first.lookup("key-1") == second.lookup("key-1")

    counts = np.bincount(
        [first.lookup(f"key-{i}") for i in range(4_000)],
        minlength=4,
    )

    assert counts.min() > 0.15 * 4_000
    assert counts.max() < 0.35 * 4_000