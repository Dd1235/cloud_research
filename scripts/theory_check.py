import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_policies import (
    POLICY_SEED_OFFSET,
    build_workers,
    median_across_seeds,
    resolve_policy_names,
    run,
)
from sim.engine import Engine
from sim.policies import make_policy
from sim.radix import PrefixCache
from sim.router import Router
from sim.sampler import OutstandingSampler
from sim.turnover import che_characteristic_time, turnover_from_evictions
from sim.workload import generate


# Does the router's view really age at the cache's characteristic time?
#
# Che's approximation says an LRU cache of C blocks keeps a block for T_C after
# its last access. Two checks, both on the perfect-view run first:
#   1. the idle time before eviction (evicted_at - last_access) should pile up
#      near one value, T_C, and that value should match capacity / eviction
#      rate and Che's fixed point from the per-block request rates
#   2. the false positives of a snapshot view should be a function of
#      age / T_C alone: runs at different capacities should collapse on that
#      axis, and the survival view's own prediction should sit on the curve


def residence_times(*, seed, n_workers, rate, n_requests, cache_blocks, zipf_alpha, worker_kind,
                    policy_name="longest_prefix"):
    """Perfect-view run with the eviction log switched on; returns idle times and rates."""
    engine = Engine(seed)
    workers = build_workers(engine, worker_kind, n_workers, cache_blocks)

    for worker in workers:
        worker.cache = PrefixCache(cache_blocks, record_residence=True)
        worker.view = worker.cache   # the router re-installs a perfect view below

    policy = make_policy(policy_name, np.random.default_rng(seed + POLICY_SEED_OFFSET))
    requests = generate(np.random.default_rng(seed), n_requests, rate, zipf_alpha=zipf_alpha)
    router = Router(engine, policy, workers)
    OutstandingSampler(engine, workers)
    router.replay(requests)
    engine.run()

    idle_before_eviction = np.sort(np.array(
        [evicted_at - last_access
         for worker in workers
         for _, _, last_access, evicted_at in worker.cache.residence_log]
    ))

    duration = requests[-1].arrival
    evictions = sum(worker.cache.evictions for worker in workers)

    # Che's fixed point wants each block's request rate *at the worker that
    # holds it*. every request records where it went, so count references per
    # (worker, block) and solve one fixed point per worker
    references = {}
    for req in requests:
        for block in req.blocks:
            references[(req.worker_id, block)] = references.get((req.worker_id, block), 0) + 1

    per_worker_times = []
    for worker in workers:
        rates = [count / duration for (worker_id, _), count in references.items() if worker_id == worker.id]
        per_worker_times.append(che_characteristic_time(rates, cache_blocks))

    def residence_cdf(idle_age):
        # empirical P[evicted by this idle age]
        return float(np.searchsorted(idle_before_eviction, idle_age, side="right") / len(idle_before_eviction))

    return dict(
        idle=idle_before_eviction,
        turnover_evictions=turnover_from_evictions(cache_blocks, evictions // n_workers, duration),
        turnover_che=float(np.mean(per_worker_times)),
        residence_cdf=residence_cdf,
    )


def fp_versus_age(policy_name, periods, capacities, *, seeds, predict_with, **common):
    """Snapshot views wrapped in a survival view, so each run carries both the
    measured false positives and the model's prediction of them.

    predict_with is "step" (Che's deterministic lifetime at the measured
    turnover) or "cdf" (the residence-time distribution measured on a
    perfect-view run of the same policy and capacity)."""
    rows = {}

    for cache_blocks in capacities:
        residence = residence_times(seed=0, cache_blocks=cache_blocks, policy_name=policy_name, **common)
        turnover = residence["turnover_evictions"]
        residence_cdf = residence["residence_cdf"] if predict_with == "cdf" else None

        for period in periods:
            row = median_across_seeds(
                [run(policy_name, seed=seed, view_kind="snapshot", view_period=period,
                     survival_turnover=turnover, residence_cdf=residence_cdf,
                     cache_blocks=cache_blocks, **common)
                 for seed in range(seeds)]
            )
            rows[(cache_blocks, period)] = dict(row, turnover=turnover)
            print(
                f"{policy_name:>15} C={cache_blocks:>5} P={period:>5} T_C={turnover:6.1f}s "
                f"age={row['mean_view_age']:6.2f} age/T_C={row['mean_view_age'] / turnover:5.3f} "
                f"fp={row['view_fp_rate']:.4f} predicted={row['predicted_fp_rate']:.4f} "
                f"hit={row['hit_rate']:.3f}"
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Check the characteristic-time model of view staleness.")
    parser.add_argument("--policies", default="longest_prefix,hybrid", help="(default: longest_prefix,hybrid)")
    parser.add_argument("--capacities", default="128,256,512,1024", help="blocks per worker (default: 128,256,512,1024)")
    parser.add_argument("--periods", default="0.5,1,2,5,10,30", help="snapshot periods in seconds (default: 0.5,1,2,5,10,30)")
    parser.add_argument("--rate", type=float, default=6.0)
    parser.add_argument("--zipf", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", choices=["sequential", "batched"], default="batched")
    parser.add_argument("--predict-with", choices=["step", "cdf"], default="cdf", help="che's step at the turnover, or the measured residence cdf (default: cdf)")
    parser.add_argument("--output-prefix", default="out/theory_check")
    args = parser.parse_args()

    policy_names = resolve_policy_names(args.policies)
    capacities = [int(c) for c in args.capacities.split(",")]
    periods = [float(p) for p in args.periods.split(",")]
    common = dict(n_workers=args.workers, rate=args.rate, n_requests=args.requests,
                  zipf_alpha=args.zipf, worker_kind=args.worker)

    os.makedirs("out", exist_ok=True)

    # 1. residence times at the reference capacity
    residence = residence_times(seed=0, cache_blocks=256, **common)
    print(
        f"residence (C=256): idle-before-eviction p10={np.percentile(residence['idle'], 10):.1f}s "
        f"p50={np.percentile(residence['idle'], 50):.1f}s p90={np.percentile(residence['idle'], 90):.1f}s; "
        f"turnover from evictions {residence['turnover_evictions']:.1f}s, "
        f"che fixed point {residence['turnover_che']:.1f}s"
    )

    # 2. false positives vs age / T_C, per policy and capacity
    results = {name: fp_versus_age(name, periods, capacities, seeds=args.seeds,
                                   predict_with=args.predict_with, **common)
               for name in policy_names}

    with open(f"{args.output_prefix}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "cache_blocks", "period", "turnover_s", "mean_view_age",
                         "age_over_turnover", "view_fp_rate", "predicted_fp_rate", "hit_rate", "routing_regret_rate"])
        for name, rows in results.items():
            for (cache_blocks, period), row in rows.items():
                writer.writerow([name, cache_blocks, period, f"{row['turnover']:.2f}",
                                 f"{row['mean_view_age']:.3f}", f"{row['mean_view_age'] / row['turnover']:.4f}",
                                 f"{row['view_fp_rate']:.5f}", f"{row['predicted_fp_rate']:.5f}",
                                 f"{row['hit_rate']:.4f}", f"{row['routing_regret_rate']:.5f}"])
    print(f"wrote {args.output_prefix}.csv")

    figure, axes = plt.subplots(1, 1 + len(policy_names), figsize=(5 * (1 + len(policy_names)), 4))

    axes[0].hist(residence["idle"], bins=60, density=True, cumulative=True, histtype="step")
    axes[0].axvline(residence["turnover_evictions"], color="C1", linestyle="--", label="capacity / eviction rate")
    axes[0].axvline(residence["turnover_che"], color="C2", linestyle=":", label="che fixed point")
    axes[0].set_title("idle time before eviction, CDF (C=256)")
    axes[0].set_xlabel("seconds since last access at eviction")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for axis, (name, rows) in zip(axes[1:], results.items()):
        for index, cache_blocks in enumerate(capacities):
            xs = [rows[(cache_blocks, period)]["mean_view_age"] / rows[(cache_blocks, period)]["turnover"] for period in periods]
            measured = [rows[(cache_blocks, period)]["view_fp_rate"] for period in periods]
            predicted = [rows[(cache_blocks, period)]["predicted_fp_rate"] for period in periods]
            axis.plot(xs, measured, "o-", color=f"C{index}", label=f"C={cache_blocks} measured")
            axis.plot(xs, predicted, "x--", color=f"C{index}", alpha=0.7, label=f"C={cache_blocks} predicted")
        axis.set_title(f"view false positives, {name} (prediction: {args.predict_with})")
        axis.set_xlabel("mean view age / cache turnover")
        axis.set_xscale("symlog", linthresh=0.1)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}.png", dpi=120)
    print(f"wrote {args.output_prefix}.png")


if __name__ == "__main__":
    main()
