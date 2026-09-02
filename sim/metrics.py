import numpy as np


def summarize(
    requests,
    workers,
    warmup_frac: float = 0.1,
) -> dict:
    done = sorted(
        (req for req in requests if req.done),
        key=lambda req: req.arrival,
    )

    warmup_count = int(len(done) * warmup_frac)
    done = done[warmup_count:]

    ttft = np.asarray([req.ttft for req in done])
    tpot = np.asarray([req.tpot for req in done])

    # every gap between consecutive tokens, pooled over all requests. empty for
    # the sequential worker, which does not record per token times
    tbt_gaps = np.asarray(
        [
            gap
            for req in done
            for gap in req.tbt_gaps
        ]
    )


    tokens_processed = sum(
        worker.tokens_processed
        for worker in workers
    )
    tokens_reused = sum(
        worker.tokens_reused
        for worker in workers
    )

    load = np.asarray(
        [worker.busy_time for worker in workers],
        dtype=float,
    )

    view_errors = view_error_rates(done)

    return {
        "n": len(done),
        "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p95": float(np.percentile(ttft, 95)),
        "ttft_p99": float(np.percentile(ttft, 99)),
        "tpot_p50": float(np.percentile(tpot, 50)),
        "tpot_p99": float(np.percentile(tpot, 99)),
        "tbt_p99": (
            float(np.percentile(tbt_gaps, 99))
            if tbt_gaps.size
            else 0.0
        ),
        "hit_rate": (
            tokens_reused / tokens_processed
            if tokens_processed
            else 0.0
        ),
        "load_cv": (
            float(load.std() / load.mean())
            if load.mean() > 0
            else 0.0
        ),
        "evictions": sum(
            worker.cache.evictions
            for worker in workers
        ),
        **view_errors,
    }


def view_error_rates(requests) -> dict:
    """How wrong the router's cache view was, split by where the error lives.

    Only requests that went through a router carry the dispatch-time fields.
    Every rate is a fraction of the prompt tokens of those requests, so the
    numbers compare across policies and view models.

        view fp        the view promised tokens the chosen worker did not hold
        view fn        the chosen worker held tokens the view did not know about
        routing regret a warmer worker existed at the instant of the decision
        execution fp   promised tokens that were gone by the time the worker
                       admitted the request: view error plus queueing drift

    With a perfect view the two view errors are zero by construction and the
    regret of longest-prefix is zero too; anything left in execution fp is
    purely the drift between dispatch and admission.
    """
    routed = [
        req
        for req in requests
        if req.estimated_cached_tokens is not None
    ]

    if not routed:
        return {
            "view_fp_rate": 0.0,
            "view_fn_rate": 0.0,
            "routing_regret_rate": 0.0,
            "execution_fp_rate": 0.0,
            "overlap_mae": 0.0,
        }

    prompt_tokens = sum(req.prompt_tokens for req in routed)

    view_fp_tokens = sum(
        max(req.estimated_cached_tokens - req.true_cached_tokens_at_dispatch, 0)
        for req in routed
    )
    view_fn_tokens = sum(
        max(req.true_cached_tokens_at_dispatch - req.estimated_cached_tokens, 0)
        for req in routed
    )
    routing_regret_tokens = sum(
        req.best_cached_tokens_at_dispatch - req.true_cached_tokens_at_dispatch
        for req in routed
    )
    execution_fp_tokens = sum(
        max(req.estimated_cached_tokens - req.cached_tokens, 0)
        for req in routed
    )

    # per request, so a long prompt does not dominate the mean
    overlap_errors = [
        abs(req.estimated_cached_tokens - req.true_cached_tokens_at_dispatch)
        / req.prompt_tokens
        for req in routed
        if req.prompt_tokens
    ]

    return {
        "view_fp_rate": view_fp_tokens / prompt_tokens,
        "view_fn_rate": view_fn_tokens / prompt_tokens,
        "routing_regret_rate": routing_regret_tokens / prompt_tokens,
        "execution_fp_rate": execution_fp_tokens / prompt_tokens,
        "overlap_mae": float(np.mean(overlap_errors)) if overlap_errors else 0.0,
    }


def fmt(row: dict) -> str:
    return " ".join(
        f"{key}={value:.3f}"
        if isinstance(value, float)
        else f"{key}={value}"
        for key, value in row.items()
    )