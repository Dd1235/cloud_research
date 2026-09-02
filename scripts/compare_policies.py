import argparse

import numpy as np

from sim.batched_worker import BatchedWorker
from sim.engine import Engine
from sim.metrics import fmt, summarize
from sim.policies import POLICIES, make_policy
from sim.router import Router
from sim.views import make_view_factory
from sim.workers import Worker
from sim.workload import generate


# The policy rng is offset from the workload rng so that switching policy does
# not also change the arrival stream. Every policy therefore sees exactly the
# same requests for a given seed, which is what makes the comparison paired.
POLICY_SEED_OFFSET = 1_000_003

# One decode token on an unbatched stream costs c_iter + c_decode in the batched
# worker, so these are split out of the sequential worker's c_decode. A single
# request therefore takes the same time under both models, and any difference in
# the results comes from batching rather than from a change of constants.
C_PREFILL = 1e-3
C_DECODE_SEQUENTIAL = 1e-2
C_ITER_BATCHED = 8e-3
C_DECODE_BATCHED = C_DECODE_SEQUENTIAL - C_ITER_BATCHED


def build_workers(
    engine,
    worker_kind: str,
    n_workers: int,
    cache_blocks: int,
    prefill_budget: int | None = None,
):
    if worker_kind == "sequential":
        return [
            Worker(
                engine,
                wid=worker_id,
                c_prefill=C_PREFILL,
                c_decode=C_DECODE_SEQUENTIAL,
                cache_blocks=cache_blocks,
            )
            for worker_id in range(n_workers)
        ]

    return [
        BatchedWorker(
            engine,
            wid=worker_id,
            c_prefill=C_PREFILL,
            c_decode=C_DECODE_BATCHED,
            c_iter=C_ITER_BATCHED,
            cache_blocks=cache_blocks,
            prefill_budget=prefill_budget,
        )
        for worker_id in range(n_workers)
    ]


def run(
    policy_name: str,
    *,
    seed: int,
    n_workers: int,
    rate: float,
    n_requests: int,
    cache_blocks: int,
    zipf_alpha: float,
    worker_kind: str = "sequential",
    prefill_budget: int | None = None,
    view_kind: str = "perfect",
    view_period: float | None = None,
    shadow_blocks: int | None = None,
):

    engine = Engine(seed)

    workers = build_workers(
        engine,
        worker_kind,
        n_workers,
        cache_blocks,
        prefill_budget,
    )

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

    # the router installs the cache view on every worker and owns dispatch, so
    # the view error accounting happens here for every script
    router = Router(
        engine,
        policy,
        workers,
        make_view_factory(
            view_kind,
            engine,
            period=view_period,
            shadow_blocks=shadow_blocks or cache_blocks,
        ),
    )

    router.replay(requests)
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
    parser.add_argument(
        "--worker",
        choices=["sequential", "batched"],
        default="sequential",
        help="worker model: one request at a time, or continuous batching",
    )
    parser.add_argument(
        "--prefill-budget",
        type=int,
        default=0,
        help="batched worker only: prompt tokens per iteration, 0 for unchunked",
    )
    args = parser.parse_args()

    main(
        resolve_policy_names(args.policies),
        seeds=args.seeds,
        n_workers=args.workers,
        rate=args.rate,
        n_requests=args.requests,
        cache_blocks=args.cache_blocks,
        zipf_alpha=args.zipf,
        worker_kind=args.worker,
        prefill_budget=args.prefill_budget or None,
    )
