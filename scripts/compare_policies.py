import numpy as np

from sim.engine import Engine
from sim.policies.longest_prefix import LongestPrefix
from sim.policies.round_robin import RoundRobin
from sim.workers import Worker
from sim.workload import generate
from sim.metrics import fmt, summarize

import argparse


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

    return summarize(requests, workers)

def main(seeds: int = 5):
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
        rows = [
            run(
                policy_type(),
                seed=seed,
                **common,
            )
            for seed in range(seeds)
        ]

        median_row = {
            key: (
                float(np.median([row[key] for row in rows]))
                if isinstance(rows[0][key], float)
                else rows[0][key]
            )
            for key in rows[0]
        }

        print(f"{policy_type.name:>15}: {fmt(median_row)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare cache-blind and prefix-aware routing."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of random seeds per policy (default: 5)",
    )
    args = parser.parse_args()

    main(seeds=args.seeds)