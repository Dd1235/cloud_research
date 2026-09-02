import argparse

import numpy as np

from sim.engine import Engine
from sim.metrics import fmt, summarize
from sim.policies import POLICIES, make_policy
from sim.workers import Worker
from sim.workload import generate


# The policy rng is offset from the workload rng so that switching policy does
# not also change the arrival stream. Every policy therefore sees exactly the
# same requests for a given seed, which is what makes the comparison paired.
POLICY_SEED_OFFSET = 1_000_003


def run(
    policy_name: str,
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

    policy = make_policy(
        policy_name,
        np.random.default_rng(seed + POLICY_SEED_OFFSET),
    )

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


def median_across_seeds(rows: list[dict]) -> dict:
    # counts stay as they are, floats become the median over seeds
    return {
        key: (
            float(np.median([row[key] for row in rows]))
            if isinstance(rows[0][key], float)
            else rows[0][key]
        )
        for key in rows[0]
    }


def resolve_policy_names(requested: str) -> list[str]:
    if requested == "all":
        return list(POLICIES)

    names = [name.strip() for name in requested.split(",")]

    for name in names:
        if name not in POLICIES:
            known_names = ", ".join(POLICIES)
            raise SystemExit(
                f"unknown policy {name!r}; known policies: {known_names}"
            )

    return names


def main(policy_names: list[str], seeds: int = 5, **common):
    for policy_name in policy_names:
        rows = [
            run(
                policy_name,
                seed=seed,
                **common,
            )
            for seed in range(seeds)
        ]

        print(f"{policy_name:>18}: {fmt(median_across_seeds(rows))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare cache-blind and prefix-aware routing."
    )
    parser.add_argument(
        "--policies",
        default="all",
        help="comma separated policy names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of random seeds per policy (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="aggregate arrival rate in requests per second (default: 2.0)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=4_000,
        help="requests per run (default: 4000)",
    )
    parser.add_argument(
        "--zipf",
        type=float,
        default=1.0,
        help="prefix popularity skew (default: 1.0)",
    )
    parser.add_argument(
        "--cache-blocks",
        type=int,
        default=256,
        help="prefix cache capacity per worker, in blocks (default: 256)",
    )
    args = parser.parse_args()

    main(
        resolve_policy_names(args.policies),
        seeds=args.seeds,
        n_workers=args.workers,
        rate=args.rate,
        n_requests=args.requests,
        c_prefill=1e-3,
        c_decode=1e-2,
        cache_blocks=args.cache_blocks,
        zipf_alpha=args.zipf,
    )
