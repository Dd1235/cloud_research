"""Where, in believed-age space, do a stale view's false promises live?

theory_check compares the survival model's aggregate prediction with the
measured false-positive rate and finds it 10-20x too high at small view ages.
This bins every promised block by how old the view believed it to be and
compares, bin by bin, the measured false-positive fraction with what the
residence distribution implies over the view's age. If the model is wrong in
every bin the population is wrong; if only in some, the shape is. The rescue
by re-reference during the view's age is ignored here, so the prediction is an
upper bound.
"""
import argparse
import csv
import os

import numpy as np

from compare_policies import POLICY_SEED_OFFSET, build_workers
from sim.engine import Engine
from sim.policies import make_policy
from sim.router import Router
from sim.sampler import OutstandingSampler
from sim.views import make_view_factory
from sim.workload import generate
from theory_check import DEPTH_BINS, depth_bin, residence_times

BIN_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0, float("inf")]


def block_samples(policy_name, *, seed, period, cache_blocks, n_workers, rate, n_requests, zipf_alpha,
                  worker_kind):
    """One snapshot-view run; returns (known age, view age, was false) per promised block."""
    engine = Engine(seed)
    workers = build_workers(engine, worker_kind, n_workers, cache_blocks)
    policy = make_policy(policy_name, np.random.default_rng(seed + POLICY_SEED_OFFSET))
    requests = generate(np.random.default_rng(seed), n_requests, rate, zipf_alpha=zipf_alpha)
    router = Router(
        engine,
        policy,
        workers,
        view_factory=make_view_factory("snapshot", engine, period=period),
        record_block_samples=True,
    )
    OutstandingSampler(engine, workers)
    router.replay(requests)
    engine.run()

    samples = np.array(router.block_samples, dtype=float)
    # the first tenth is warm-up, as in every other script
    return samples[len(samples) // 10:]


def hazard_over_view_age(residence_cdf, known_age, view_age):
    """P[evicted during the view's age | still present when the view was taken]."""
    idle_at_scrape = known_age - view_age
    gone_by_now = residence_cdf(known_age)
    survived_to_scrape = 1.0 - residence_cdf(idle_at_scrape)
    if survived_to_scrape <= 0.0:
        return 1.0
    return min(max((gone_by_now - (1.0 - survived_to_scrape)) / survived_to_scrape, 0.0), 1.0)


def step_over_view_age(turnover, known_age, view_age):
    """Che's deterministic lifetime: gone iff the block crossed the turnover while the view aged."""
    return 1.0 if known_age - view_age < turnover <= known_age else 0.0


def main():
    parser = argparse.ArgumentParser(description="Bin a stale view's false positives by believed block age.")
    parser.add_argument("--policy", default="longest_prefix")
    parser.add_argument("--periods", default="0.5,5", help="snapshot periods in seconds (default: 0.5,5)")
    parser.add_argument("--cache-blocks", type=int, default=256)
    parser.add_argument("--rate", type=float, default=6.0)
    parser.add_argument("--zipf", type=float, default=0.9)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", choices=["sequential", "batched"], default="batched")
    parser.add_argument("--residence-population", choices=["all", "reused"], default="all")
    parser.add_argument("--output", default="out/fp_by_age.csv")
    args = parser.parse_args()

    common = dict(n_workers=args.workers, rate=args.rate, n_requests=args.requests,
                  zipf_alpha=args.zipf, worker_kind=args.worker)
    residence = residence_times(seed=0, cache_blocks=args.cache_blocks, policy_name=args.policy,
                                population=args.residence_population, **common)
    turnover = residence["turnover_evictions"]
    residence_cdf = residence["residence_cdf"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = []
    for period in [float(p) for p in args.periods.split(",")]:
        samples = block_samples(args.policy, seed=0, period=period, cache_blocks=args.cache_blocks, **common)
        known_ages, view_ages, was_false, depths = (samples[:, 0], samples[:, 1], samples[:, 2],
                                                    samples[:, 3].astype(int))

        print(f"\n{args.policy} C={args.cache_blocks} P={period} T_C={turnover:.1f}s: "
              f"{len(samples)} promised blocks, fp overall {was_false.mean():.4f}")
        print(f"{'believed age (s)':>18} {'share':>7} {'measured fp':>12} {'cdf hazard':>11} {'che step':>9}")
        for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            in_bin = (known_ages >= low) & (known_ages < high)
            if not in_bin.any():
                continue
            measured = was_false[in_bin].mean()
            predicted_cdf = np.mean([hazard_over_view_age(residence_cdf, x, a)
                                     for x, a in zip(known_ages[in_bin], view_ages[in_bin])])
            predicted_step = np.mean([step_over_view_age(turnover, x, a)
                                      for x, a in zip(known_ages[in_bin], view_ages[in_bin])])
            label = f"[{low:g}, {high:g})"
            print(f"{label:>18} {in_bin.mean():7.3f} {measured:12.4f} {predicted_cdf:11.4f} {predicted_step:9.4f}")
            rows.append([args.policy, args.cache_blocks, period, low, high, f"{in_bin.mean():.4f}",
                         f"{measured:.5f}", f"{predicted_cdf:.5f}", f"{predicted_step:.5f}"])

        # the same promises cut by generation instead of by believed age: where
        # along the match do the false promises actually live, and does either
        # cdf see it? the aggregate cdf gives every depth the same hazard, so a
        # gap between generations here is exactly what depth conditioning buys
        residence_cdf_by_depth = residence["residence_cdf_by_depth"]
        print(f"{'depth':>12} {'share':>7} {'measured fp':>12} {'cdf hazard':>11} {'depth hazard':>13}")
        for index, (low, high) in enumerate(DEPTH_BINS):
            in_bin = np.array([depth_bin(depth) == index for depth in depths])
            if not in_bin.any():
                continue
            measured = was_false[in_bin].mean()
            predicted_all = np.mean([hazard_over_view_age(residence_cdf, x, a)
                                     for x, a in zip(known_ages[in_bin], view_ages[in_bin])])
            predicted_depth = np.mean([
                hazard_over_view_age(lambda idle: residence_cdf_by_depth(idle, depth), x, a)
                for x, a, depth in zip(known_ages[in_bin], view_ages[in_bin], depths[in_bin])
            ])
            label = f"{low}-{high}" if high is not None else f"{low}+"
            print(f"{label:>12} {in_bin.mean():7.3f} {measured:12.4f} {predicted_all:11.4f} {predicted_depth:13.4f}")
            rows.append([args.policy, args.cache_blocks, period, f"depth {label}", "", f"{in_bin.mean():.4f}",
                         f"{measured:.5f}", f"{predicted_all:.5f}", f"{predicted_depth:.5f}"])

    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "cache_blocks", "period", "age_low", "age_high", "share",
                         "measured_fp", "predicted_cdf_hazard", "predicted_che_step"])
        writer.writerows(rows)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
