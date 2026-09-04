import numpy as np
import pytest

from sim.engine import Engine
from sim.metrics import summarize, view_error_rates
from sim.policies.longest_prefix import LongestPrefix
from sim.request import Request
from sim.router import Router
from sim.workers import Worker
from sim.workload import generate


def routed_request(
    request_id: int,
    *,
    prompt_tokens: int,
    estimated: int,
    true_at_dispatch: int,
    best_at_dispatch: int,
    cached_at_admission: int,
):
    """A finished request with the dispatch-time fields filled in by hand."""
    return Request(
        id=request_id,
        arrival=0.0,
        prompt_tokens=prompt_tokens,
        output_tokens=2,
        cached_tokens=cached_at_admission,
        start=0.5,
        first_token=1.0,
        finish=2.0,
        estimated_cached_tokens=estimated,
        true_cached_tokens_at_dispatch=true_at_dispatch,
        best_cached_tokens_at_dispatch=best_at_dispatch,
    )


def test_each_error_rate_is_computed_from_its_own_pair_of_fields():
    requests = [
        # the view promised 60 but only 40 were there (fp 20); a worker with 80
        # existed (regret 40); by admission only 30 remained (execution fp 30)
        routed_request(
            1,
            prompt_tokens=100,
            estimated=60,
            true_at_dispatch=40,
            best_at_dispatch=80,
            cached_at_admission=30,
        ),
        # the view promised 10 but 50 were really there (fn 40); it was the
        # best worker anyway, and everything promised was honoured
        routed_request(
            2,
            prompt_tokens=100,
            estimated=10,
            true_at_dispatch=50,
            best_at_dispatch=50,
            cached_at_admission=50,
        ),
    ]

    rates = view_error_rates(requests)

    assert rates["view_fp_rate"] == pytest.approx(20 / 200)
    assert rates["view_fn_rate"] == pytest.approx(40 / 200)
    assert rates["routing_regret_rate"] == pytest.approx(40 / 200)
    assert rates["execution_fp_rate"] == pytest.approx(30 / 200)
    assert rates["overlap_mae"] == pytest.approx((0.2 + 0.4) / 2)
    assert rates["predicted_fp_rate"] == 0.0          # no view offered an expectation


def test_predicted_fp_is_promised_minus_expected():
    req = routed_request(
        1,
        prompt_tokens=100,
        estimated=60,
        true_at_dispatch=40,
        best_at_dispatch=60,
        cached_at_admission=40,
    )
    req.expected_cached_tokens = 45.0

    assert view_error_rates([req])["predicted_fp_rate"] == pytest.approx(15 / 100)


def test_requests_that_never_saw_a_router_are_ignored():
    unrouted = Request(
        id=1,
        arrival=0.0,
        prompt_tokens=100,
        output_tokens=2,
        start=0.0,
        first_token=1.0,
        finish=2.0,
    )

    rates = view_error_rates([unrouted])

    assert rates == {
        "mean_view_age": 0.0,
        "view_fp_rate": 0.0,
        "view_fn_rate": 0.0,
        "routing_regret_rate": 0.0,
        "execution_fp_rate": 0.0,
        "predicted_fp_rate": 0.0,
        "overlap_mae": 0.0,
    }


def test_perfect_view_and_longest_prefix_have_zero_view_error_and_zero_regret():
    engine = Engine(seed=0)
    workers = [
        Worker(
            engine,
            wid=worker_id,
            c_prefill=0.01,
            c_decode=0.01,
            cache_blocks=1000,
        )
        for worker_id in range(2)
    ]
    router = Router(engine, LongestPrefix(), workers)

    requests = generate(
        np.random.default_rng(0),
        40,
        rate=5.0,
        n_prefixes=2,
        prefix_blocks=4,
        suffix_blocks=(1, 2),
    )
    router.replay(requests)
    engine.run()

    metrics = summarize(requests, workers, warmup_frac=0.0)

    # the view is the truth, so it cannot be wrong either way; and longest
    # prefix picks the argmax of that truth, so no better worker ever existed
    assert metrics["view_fp_rate"] == 0.0
    assert metrics["view_fn_rate"] == 0.0
    assert metrics["overlap_mae"] == 0.0
    assert metrics["routing_regret_rate"] == 0.0

    # a huge cache never evicts, so nothing promised can disappear before
    # admission either: no drift, no execution false positives
    assert metrics["execution_fp_rate"] == 0.0


def test_windowed_series_buckets_by_arrival_and_splits_reuse_per_worker():
    from sim.metrics import windowed_series

    def make(request_id, arrival, worker_id, cached, first_token):
        req = Request(id=request_id, arrival=arrival, prompt_tokens=32, output_tokens=1, blocks=())
        req.worker_id = worker_id
        req.cached_tokens = cached
        req.first_token = first_token
        req.finish = first_token
        return req

    requests = [
        # window [0, 10): two requests, 48 of 64 tokens reused, ttft 1 and 3
        make(1, 0.0, worker_id=0, cached=32, first_token=1.0),
        make(2, 4.0, worker_id=1, cached=16, first_token=7.0),
        # window [10, 20): one cold request on worker 1
        make(3, 12.0, worker_id=1, cached=0, first_token=14.0),
    ]

    series = windowed_series(requests, window=10.0)

    assert len(series) == 2
    assert series[0]["n"] == 2
    assert series[0]["hit_rate"] == pytest.approx(48 / 64)
    assert series[0]["ttft_p50"] == pytest.approx(2.0)
    assert series[0]["hit_rate_by_worker"] == {0: pytest.approx(1.0), 1: pytest.approx(0.5)}
    assert series[0]["share_by_worker"] == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}
    assert series[1]["hit_rate"] == pytest.approx(0.0)
    assert series[1]["t"] == pytest.approx(10.0)
