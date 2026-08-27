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

    return {
        "n": len(done),
        "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p95": float(np.percentile(ttft, 95)),
        "ttft_p99": float(np.percentile(ttft, 99)),
        "tpot_p50": float(np.percentile(tpot, 50)),
        "tpot_p99": float(np.percentile(tpot, 99)),
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
    }