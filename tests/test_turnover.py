import math

import pytest

from sim.turnover import che_characteristic_time, turnover_from_evictions


def test_turnover_is_capacity_over_eviction_rate():
    # 256 blocks, 512 evictions in 100 s: the cache turned over twice
    assert turnover_from_evictions(256, evictions=512, duration=100.0) == pytest.approx(50.0)
    assert turnover_from_evictions(256, evictions=0, duration=100.0) == float("inf")


def test_che_fixed_point_matches_the_closed_form_for_equal_rates():
    # n identical blocks at rate lam into a cache of C < n blocks:
    # n * (1 - exp(-lam * T)) = C  ->  T = -ln(1 - C/n) / lam
    rates = [1.0] * 4
    expected = -math.log(1.0 - 2 / 4) / 1.0

    assert che_characteristic_time(rates, capacity=2) == pytest.approx(expected, rel=1e-6)


def test_che_time_is_infinite_when_everything_fits():
    assert che_characteristic_time([1.0, 2.0, 3.0], capacity=3) == float("inf")
    assert che_characteristic_time([1.0, 0.0, 0.0], capacity=1) == float("inf")


def test_hotter_blocks_shorten_the_characteristic_time():
    cold = che_characteristic_time([0.1] * 10, capacity=5)
    hot = che_characteristic_time([1.0] * 10, capacity=5)

    assert hot < cold
