import pytest

from scripts.compare_policies import seed_statistics


def test_median_and_mean_match_hand_computed_values_for_three_seeds():
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]

    stats = seed_statistics(rows)["x"]

    # median and mean of 1, 2, 3 are both 2 exactly, no floating point slop
    assert stats["median"] == pytest.approx(2.0)
    assert stats["mean"] == pytest.approx(2.0)

    # a percentile bootstrap of the mean can only resample from {1, 2, 3}, so
    # every resample mean lands in [1, 3] and the true mean sits inside the
    # interval it produces
    assert stats["ci_low"] >= 1.0
    assert stats["ci_high"] <= 3.0
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_default_rng_gives_identical_ci_across_repeated_calls():
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}]

    # rng=None means np.random.default_rng(0) internally, so the resample draw
    # is the same every time and a sweep script's reported CI does not jitter
    # between runs
    first_call = seed_statistics(rows)["x"]
    second_call = seed_statistics(rows)["x"]

    assert first_call["ci_low"] == second_call["ci_low"]
    assert first_call["ci_high"] == second_call["ci_high"]


def test_single_seed_collapses_ci_to_the_point_value():
    rows = [{"x": 5.0}]

    stats = seed_statistics(rows)["x"]

    # nothing to resample with only one seed, so every statistic is just 5.0
    assert stats["median"] == pytest.approx(5.0)
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["ci_low"] == pytest.approx(5.0)
    assert stats["ci_high"] == pytest.approx(5.0)
