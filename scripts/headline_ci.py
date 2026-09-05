"""The headline staleness cells at 10 seeds, with bootstrap intervals.

The sweeps report 3-seed medians, which is fine for shapes and indefensible
for a paper table. This re-runs the cells the paper actually quotes (three
policies at a fresh, a mid-age, and a one-turnover-old view) at 10 paired
seeds and prints mean with a 95% bootstrap CI from seed_statistics.
"""
import argparse
import csv
import os

from compare_policies import run, seed_statistics

CELLS = ("longest_prefix", "hybrid", "dualmap")
PERIODS = (0.0, 5.0, 30.0)
METRICS = ("hit_rate", "ttft_p99", "view_fp_rate", "queue_cv")


def main():
    parser = argparse.ArgumentParser(description="Headline cells with confidence intervals.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--rate", type=float, default=6.0)
    parser.add_argument("--zipf", type=float, default=0.9)
    parser.add_argument("--output", default="out/headline_ci.csv")
    args = parser.parse_args()

    os.makedirs("out", exist_ok=True)
    rows = []

    for policy_name in CELLS:
        for period in PERIODS:
            per_seed = [
                run(
                    policy_name,
                    seed=seed,
                    n_workers=4,
                    rate=args.rate,
                    n_requests=args.requests,
                    cache_blocks=256,
                    zipf_alpha=args.zipf,
                    worker_kind="batched",
                    view_kind="perfect" if period == 0.0 else "snapshot",
                    view_period=period or None,
                )
                for seed in range(args.seeds)
            ]
            stats = seed_statistics(per_seed)
            cells = {
                metric: stats[metric]
                for metric in METRICS
            }
            print(f"{policy_name:>15} P={period:>4}: " + "  ".join(
                f"{metric}={cells[metric]['mean']:.4f} [{cells[metric]['ci_low']:.4f}, {cells[metric]['ci_high']:.4f}]"
                for metric in ("hit_rate", "view_fp_rate")
            ))
            for metric in METRICS:
                rows.append(dict(
                    policy=policy_name, period=period, metric=metric, seeds=args.seeds,
                    mean=f"{cells[metric]['mean']:.5f}", median=f"{cells[metric]['median']:.5f}",
                    ci_low=f"{cells[metric]['ci_low']:.5f}", ci_high=f"{cells[metric]['ci_high']:.5f}",
                ))

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
