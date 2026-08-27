import numpy as np

from sim.engine import Engine
from sim.policies.longest_prefix import LongestPrefix
from sim.policies.round_robin import RoundRobin
from sim.workers import Worker
from sim.workload import generate


def run(
    policy,
    *,
    seed: int,
    n_workers: int,
    rate: float,
    n_requests: int,
    c_prefill: float,
    c_decode: float,
    cache_blocks: int,
    zipf_alpha: float,
):

    engine = Engine(seed)

    workers = [
        Worker(
            engine,
            wid=worker_id,
            c_prefill=c_prefill,
            c_decode=c_decode,
            cache_blocks=cache_blocks,
        )
        for worker_id in range(n_workers)
    ]

    requests = generate(
        np.random.default_rng(seed),
        n_requests,
        rate,
        zipf_alpha=zipf_alpha,
    )

    def dispatch(req) -> None:
        chosen_worker = policy.choose(req, workers)
        chosen_worker.submit(req)

    for req in requests:
        delay = req.arrival - engine.now
        engine.schedule(delay, dispatch, req)

    engine.run()

    ttfts = np.asarray([req.ttft for req in requests])

    tokens_processed = sum(
        worker.tokens_processed
        for worker in workers
    )
    tokens_reused = sum(
        worker.tokens_reused
        for worker in workers
    )

    return {
        "mean_ttft": float(ttfts.mean()),
        "reuse_rate": tokens_reused / tokens_processed,
        "completed_per_worker": [
            worker.completed
            for worker in workers
        ],
    }


def main():
    common = dict(
        n_workers=4,
        rate=2.0,
        n_requests=4_000,
        c_prefill=1e-3,
        c_decode=1e-2,
        cache_blocks=256,
        zipf_alpha=1.0,
    )

    for policy_type in (RoundRobin, LongestPrefix):
        result = run(
            policy_type(),
            seed=0,
            **common,
        )

        print(
            f"{policy_type.name:>15} "
            f"mean_ttft={result['mean_ttft']:.3f} "
            f"reuse_rate={result['reuse_rate']:.1%} "
            f"completed={result['completed_per_worker']}"
        )


if __name__ == "__main__":
    main()