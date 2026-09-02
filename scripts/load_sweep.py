import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_policies import median_across_seeds, resolve_policy_names, run


# Where does each policy's tail latency explode, and does prefix affinity buy
# headroom before it does? Cache-blind policies pay the full prefill for every
# request, so they run out of worker time earlier than the cache-aware ones on
# the same arrival rate. That gap is capacity, not just latency.
def sweep(policy_names, rates, *, seeds: int, **common):
    results = {}

    for policy_name in policy_names:
        for rate in rates:
            rows = [
                run(
                    policy_name,
                    seed=seed,
                    rate=rate,
                    **common,
                )
                for seed in range(seeds)
            ]
            median_row = median_across_seeds(rows)
            results[(policy_name, rate)] = median_row

            print(
                f"{policy_name:>16} rate={rate:4.1f} "
                f"ttft_p50={median_row['ttft_p50']:.3f} "
                f"ttft_p99={median_row['ttft_p99']:.3f} "
                f"tpot_p50={median_row['tpot_p50']:.4f} "
                f"tbt_p99={median_row['tbt_p99']:.3f} "
                f"hit_rate={median_row['hit_rate']:.3f} "
                f"load_cv={median_row['load_cv']:.3f}"
            )

    return results


def plot(results, policy_names, rates, n_workers: int, output_path: str):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = (
        ("ttft_p99", "TTFT p99 (s)"),
        ("hit_rate", "prefix hit rate"),
        ("load_cv", "load CV (busy time)"),
    )

    for axis, (metric_key, title) in zip(axes, panels):
        for policy_name in policy_names:
            axis.plot(
                rates,
                [results[(policy_name, rate)][metric_key] for rate in rates],
                "o-",
                label=policy_name,
            )
        axis.set_title(title)
        axis.set_xlabel(f"arrival rate (req/s, {n_workers} workers)")
        axis.grid(alpha=0.3)

    # tails span orders of magnitude once a policy saturates
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("out", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep routing policies across arrival rate."
    )
    parser.add_argument(
        "--policies",
        default="round_robin,p2c,session_hash,longest_prefix,hybrid,dualmap",
        help="comma separated policy names, or 'all' (default: six representative ones)",
    )
    parser.add_argument(
        "--rates",
        default="2,4,6,8,10,12",
        help="aggregate arrival rates in req/s (default: 2,4,6,8,10,12)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="random seeds per point (default: 3)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=3_000,
        help="requests per run (default: 3000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--cache-blocks",
        type=int,
        default=256,
        help="prefix cache capacity per worker, in blocks (default: 256)",
    )
    parser.add_argument(
        "--zipf",
        type=float,
        default=1.0,
        help="prefix popularity skew (default: 1.0)",
    )
    parser.add_argument(
        "--worker",
        choices=["sequential", "batched"],
        default="batched",
        help="worker model (default: batched)",
    )
    parser.add_argument(
        "--prefill-budget",
        type=int,
        default=0,
        help="batched worker only: prompt tokens per iteration, 0 for unchunked (default: 0)",
    )
    parser.add_argument(
        "--output",
        default="out/load_sweep.png",
        help="where to write the figure (default: out/load_sweep.png)",
    )
    args = parser.parse_args()

    policy_names = resolve_policy_names(args.policies)
    rates = [float(rate) for rate in args.rates.split(",")]

    results = sweep(
        policy_names,
        rates,
        seeds=args.seeds,
        n_workers=args.workers,
        n_requests=args.requests,
        cache_blocks=args.cache_blocks,
        zipf_alpha=args.zipf,
        worker_kind=args.worker,
        prefill_budget=args.prefill_budget or None,
    )

    plot(results, policy_names, rates, args.workers, args.output)


if __name__ == "__main__":
    main()
