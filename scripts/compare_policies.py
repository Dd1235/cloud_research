import argparse

import numpy as np

from sim.batched_worker import BatchedWorker
from sim.blockrates import BlockRateTracker
from sim.engine import Engine
from sim.metrics import fmt, summarize
from sim.policies import POLICIES, make_policy
from sim.router import Router
from sim.sampler import OutstandingSampler
from sim.traces import MOONCAKE_BLOCK_SIZE, load_mooncake
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
    *,
    block_size: int = 16,
    c_prefill: float = C_PREFILL,
    c_decode_batched: float = C_DECODE_BATCHED,
    max_batch: int = 16,
    kv_available_at: str = "admission",
):
    if worker_kind == "sequential":
        return [
            Worker(
                engine,
                wid=worker_id,
                c_prefill=c_prefill,
                c_decode=C_DECODE_SEQUENTIAL,
                cache_blocks=cache_blocks,
                block_size=block_size,
            )
            for worker_id in range(n_workers)
        ]

    return [
        BatchedWorker(
            engine,
            wid=worker_id,
            c_prefill=c_prefill,
            c_decode=c_decode_batched,
            c_iter=C_ITER_BATCHED,
            cache_blocks=cache_blocks,
            block_size=block_size,
            max_batch=max_batch,
            prefill_budget=prefill_budget,
            kv_available_at=kv_available_at,
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
    view_ttl: float | None = None,
    survival_turnover: float | None = None,
    residence_cdf=None,
    kv_available_at: str = "admission",
    workload=None,
    block_size: int = 16,
    c_prefill: float = C_PREFILL,
    c_decode_batched: float = C_DECODE_BATCHED,
    max_batch: int = 16,
    policy_options: dict | None = None,
):

    engine = Engine(seed)

    workers = build_workers(
        engine,
        worker_kind,
        n_workers,
        cache_blocks,
        prefill_budget,
        block_size=block_size,
        c_prefill=c_prefill,
        c_decode_batched=c_decode_batched,
        max_batch=max_batch,
        kv_available_at=kv_available_at,
    )

    policy = make_policy(
        policy_name,
        np.random.default_rng(seed + POLICY_SEED_OFFSET),
        policy_options,
    )

    # a workload factory takes the seed, so a trace (which ignores it) and the
    # synthetic generator (which needs it) fit the same slot
    requests = (
        workload(seed)
        if workload is not None
        else generate(
            np.random.default_rng(seed),
            n_requests,
            rate,
            zipf_alpha=zipf_alpha,
        )
    )

    # the survival view needs per-block reference rates, which the router
    # collects from its own dispatches over roughly one cache lifetime
    tracker = (
        BlockRateTracker(window=survival_turnover)
        if survival_turnover is not None
        else None
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
            ttl=view_ttl,
            tracker=tracker,
            turnover=survival_turnover,
            residence_cdf=residence_cdf,
        ),
        tracker=tracker,
    )

    sampler = OutstandingSampler(engine, workers)

    router.replay(requests)
    engine.run()

    return summarize(requests, workers, sampler=sampler)


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


def trace_workload(path: str, *, speedup: float, limit: int | None):
    """A workload factory that replays a Mooncake trace, reloaded per run.

    Reloading is deliberate: Request objects carry per-run state (token times,
    dispatch fields), so sharing one list across seeds would leak it.
    """
    return lambda seed: load_mooncake(path, speedup=speedup, limit=limit)


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
    parser.add_argument(
        "--view",
        choices=["perfect", "snapshot", "shadow"],
        default="perfect",
        help="the router's picture of each worker's cache (default: perfect)",
    )
    parser.add_argument(
        "--view-period",
        type=float,
        default=None,
        help="snapshot view only: seconds between refreshes of the router's copy",
    )
    parser.add_argument(
        "--shadow-blocks",
        type=int,
        default=None,
        help="shadow view only: capacity of the router's own index (default: cache blocks)",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="replay a mooncake jsonl trace instead of the synthetic workload; --rate and --zipf are then ignored",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="trace only: compress arrival gaps by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--c-prefill",
        type=float,
        default=C_PREFILL,
        help=f"seconds per uncached prompt token (default: {C_PREFILL})",
    )
    parser.add_argument(
        "--c-decode",
        type=float,
        default=C_DECODE_BATCHED,
        help=f"batched worker: seconds per decoding sequence per iteration (default: {C_DECODE_BATCHED})",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="batched worker: sequences resident at once (default: 16)",
    )
    parser.add_argument(
        "--session-depth",
        type=int,
        default=1,
        help="session hash and dualmap: leading blocks that identify a session (default: 1)",
    )
    parser.add_argument(
        "--view-ttl",
        type=float,
        default=None,
        help="snapshot/shadow views: distrust entries whose last access is older than this many seconds",
    )
    parser.add_argument(
        "--survival-turnover",
        type=float,
        default=None,
        help="snapshot/shadow views: wrap in a survival view using this cache turnover in seconds",
    )
    parser.add_argument(
        "--kv-available-at",
        choices=["admission", "prefill_done"],
        default="admission",
        help="batched worker: when a request's blocks become reusable (default: admission)",
    )
    parser.add_argument(
        "--overlap-source",
        choices=["raw", "expected"],
        default="raw",
        help="cardinal scorers read the view's raw promise or its survival expectation (default: raw)",
    )
    args = parser.parse_args()

    workload = None
    block_size = 16
    if args.trace is not None:
        workload = trace_workload(args.trace, speedup=args.speedup, limit=args.requests)
        block_size = MOONCAKE_BLOCK_SIZE
        print(f"replaying {args.trace} (first {args.requests} requests, speedup {args.speedup}); --rate and --zipf ignored")

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
        view_kind=args.view,
        view_period=args.view_period,
        shadow_blocks=args.shadow_blocks,
        workload=workload,
        block_size=block_size,
        c_prefill=args.c_prefill,
        c_decode_batched=args.c_decode,
        max_batch=args.max_batch,
        policy_options={"key_blocks": args.session_depth, "overlap_source": args.overlap_source},
        view_ttl=args.view_ttl,
        survival_turnover=args.survival_turnover,
        kv_available_at=args.kv_available_at,
    )
