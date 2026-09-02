import pytest

from sim.blockrates import BlockRateTracker


def test_rate_counts_references_inside_the_window_only():
    tracker = BlockRateTracker(window=10.0)

    tracker.observe(("a", "b"), now=0.0)
    tracker.observe(("a",), now=4.0)
    tracker.observe(("a",), now=8.0)

    assert tracker.rate("a", now=8.0) == pytest.approx(3 / 10)
    assert tracker.rate("b", now=8.0) == pytest.approx(1 / 10)
    assert tracker.rate("zzz", now=8.0) == 0.0

    # at t=12 the reference at 0.0 has left the window; the ones at 4 and 8 remain
    assert tracker.rate("a", now=12.0) == pytest.approx(2 / 10)
    assert tracker.rate("b", now=12.0) == 0.0


def test_rates_are_kept_per_worker_when_a_worker_id_is_given():
    tracker = BlockRateTracker(window=10.0)

    # the same block routed to two workers: each copy refreshes only its own cache
    tracker.observe(("a",), now=0.0, worker_id=0)
    tracker.observe(("a",), now=1.0, worker_id=0)
    tracker.observe(("a",), now=2.0, worker_id=1)

    assert tracker.rate("a", now=2.0, worker_id=0) == pytest.approx(2 / 10)
    assert tracker.rate("a", now=2.0, worker_id=1) == pytest.approx(1 / 10)
    # nothing was observed fleet-wide, so the unkeyed rate stays empty
    assert tracker.rate("a", now=2.0) == 0.0
