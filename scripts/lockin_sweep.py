"""The lock-in, reproduced on demand: pure ranking plus a universal prefix.

On the real traces, longest-prefix routing with any record-insert view locked
onto one worker because every request begins with the same blocks. The traces
only show it happening; this sweep controls it. The synthetic workload gets a
--universal-blocks knob (k shared blocks beginning every request) and the
sweep crosses it with the candidate cures, under the perfect view and the
shadow index.

One prediction worth writing down before running: on the traces the perfect
view escaped lock-in only because coarse timestamps made the first requests
tie (nothing admitted yet, so the ranker spread them by load). Synthetic
arrivals never tie, so the first dispatch is admitted before the second
arrives, and even the perfect view should show one worker strictly warmest
from then on. If so, the lock-in is not a stale-view disease at all: it is
pure ranking plus a universal prefix, and freshness does not save it.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from compare_policies import median_across_seeds, run

# label -> (policy, options). dualmap with a one-block session key hashes the
# universal prefix itself, the exact first-block mistake from the traces
CONFIGS = {
    "longest_prefix": ("longest_prefix", {}),
    "lpm_threshold_.5": ("longest_prefix", {"match_threshold": 0.5}),
    "hybrid": ("hybrid", {}),
    "dualmap_depth16": ("dualmap", {"key_blocks": 16}),
    "dualmap_depth1": ("dualmap", {"key_blocks": 1}),
}

VIEWS = ("perfect", "shadow")


def main():
    parser = argparse.ArgumentParser(description="Lock-in vs universal prefix length and its cures.")
    parser.add_argument("--universal", default="0,1,4,8", help="universal blocks per request (default: 0,1,4,8)")
    parser.add_argument("--rate", type=float, default=6.0)
    parser.add_argument("--zipf", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-blocks", type=int, default=256)
    parser.add_argument("--output-prefix", default="out/lockin")
    args = parser.parse_args()

    universal_counts = [int(k) for k in args.universal.split(",")]
    os.makedirs("out", exist_ok=True)

    rows = []
    for view_kind in VIEWS:
        for label, (policy_name, options) in CONFIGS.items():
            for universal_blocks in universal_counts:
                row = median_across_seeds([
                    run(
                        policy_name,
                        seed=seed,
                        n_workers=args.workers,
                        rate=args.rate,
                        n_requests=args.requests,
                        cache_blocks=args.cache_blocks,
                        zipf_alpha=args.zipf,
                        worker_kind="batched",
                        view_kind=view_kind,
                        shadow_blocks=args.cache_blocks,
                        universal_blocks=universal_blocks,
                        policy_options=options,
                    )
                    for seed in range(args.seeds)
                ])
                rows.append(dict(
                    view=view_kind, config=label, universal_blocks=universal_blocks,
                    hit_rate=row["hit_rate"], ttft_p50=row["ttft_p50"], ttft_p99=row["ttft_p99"],
                    queue_cv=row["queue_cv"], load_cv=row["load_cv"],
                ))
                print(f"{view_kind:>8} {label:>18} u={universal_blocks:>2} hit={row['hit_rate']:.3f} "
                      f"p50={row['ttft_p50']:.3f} p99={row['ttft_p99']:.2f} qcv={row['queue_cv']:.2f}")

    with open(f"{args.output_prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_prefix}.csv")

    figure, axes = plt.subplots(2, len(VIEWS), figsize=(6 * len(VIEWS), 7.5), sharex=True)
    for column, view_kind in enumerate(VIEWS):
        for index, label in enumerate(CONFIGS):
            of_config = [row for row in rows if row["view"] == view_kind and row["config"] == label]
            axes[0][column].plot(universal_counts, [row["hit_rate"] for row in of_config],
                                 "o-", color=f"C{index}", label=label)
            axes[1][column].plot(universal_counts, [row["ttft_p99"] for row in of_config],
                                 "o-", color=f"C{index}", label=label)

        axes[0][column].set_title(f"{view_kind} view")
        axes[0][column].set_ylabel("hit rate")
        axes[1][column].set_yscale("log")
        axes[1][column].set_ylabel("ttft p99 (s, log)")
        axes[1][column].set_xlabel("universal blocks per request")
        for axis in (axes[0][column], axes[1][column]):
            axis.grid(alpha=0.3)
        axes[0][column].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}.png", dpi=120)
    print(f"wrote {args.output_prefix}.png")


if __name__ == "__main__":
    main()
