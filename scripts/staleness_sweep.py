import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_policies import (
    C_PREFILL,
    median_across_seeds,
    resolve_policy_names,
    run,
    trace_workload,
)
from sim.traces import MOONCAKE_BLOCK_SIZE, load_mooncake


# The research question. Production routers never see a worker's true cache:
# they scrape it every few seconds (snapshot view) or reconstruct it from their
# own routing history (shadow view). This sweeps how old the router's picture
# is and reports, at each age, how wrong the picture was (view fp/fn), how much
# that cost the decision (routing regret), and what it did to reuse and tail
# latency. Cache-blind policies never look at a view, so they are flat lines.

PERFECT = 0.0


def sweep_policy(policy_name, periods, *, seeds: int, **common):
    """{period: median row}. period 0 means the perfect view."""
    rows_by_period = {}

    for period in periods:
        view_settings = (
            dict(view_kind="perfect")
            if period == PERFECT
            else dict(view_kind="snapshot", view_period=period)
        )

        rows = [
            run(
                policy_name,
                seed=seed,
                **view_settings,
                **common,
            )
            for seed in range(seeds)
        ]

        rows_by_period[period] = median_across_seeds(rows)

    return rows_by_period


def run_shadow(policy_name, *, seeds: int, **common):
    rows = [
        run(
            policy_name,
            seed=seed,
            view_kind="shadow",
            **common,
        )
        for seed in range(seeds)
    ]

    return median_across_seeds(rows)


def run_reference(policy_name, *, seeds: int, **common):
    rows = [
        run(policy_name, seed=seed, **common)
        for seed in range(seeds)
    ]

    return median_across_seeds(rows)


def cache_turnover_seconds(perfect_row, *, cache_blocks, n_workers, n_requests, rate):
    """How long the fleet takes to evict one full cache's worth of blocks.

    Measured from the perfect-view run, so it describes the workload and the
    cache size, not the view. A snapshot older than this is describing a
    cache that has been entirely replaced since.
    """
    duration = n_requests / rate
    evictions_per_second = perfect_row["evictions"] / duration

    if evictions_per_second == 0:
        return float("inf")

    return cache_blocks * n_workers / evictions_per_second


REPORTED = (
    "mean_view_age",
    "hit_rate",
    "ttft_p50",
    "ttft_p99",
    "view_fp_rate",
    "view_fn_rate",
    "routing_regret_rate",
    "execution_fp_rate",
    "load_cv",
    "queue_cv",
)


def print_row(label, period, row):
    values = " ".join(f"{key}={row[key]:.3f}" for key in REPORTED)
    print(f"{label:>16} period={period:>5} {values}")


def sweep(policy_names, reference_names, periods, zipfs, *, include_shadow, seeds, **common):
    """Everything the figures need, keyed by (zipf, policy, period)."""
    results = {}
    turnovers = {}

    for zipf_alpha in zipfs:
        settings = dict(zipf_alpha=zipf_alpha, seeds=seeds, **common)

        for policy_name in policy_names:
            rows_by_period = sweep_policy(policy_name, periods, **settings)

            for period, row in rows_by_period.items():
                results[(zipf_alpha, policy_name, period)] = row
                print_row(f"{policy_name} a={zipf_alpha}", period, row)

            turnovers[(zipf_alpha, policy_name)] = cache_turnover_seconds(
                rows_by_period[PERFECT],
                cache_blocks=common["cache_blocks"],
                n_workers=common["n_workers"],
                n_requests=common["n_requests"],
                rate=common["rate"],
            )

            if include_shadow:
                row = run_shadow(policy_name, **settings)
                results[(zipf_alpha, policy_name, "shadow")] = row
                print_row(f"{policy_name} a={zipf_alpha}", "shadow", row)

        for reference_name in reference_names:
            row = run_reference(reference_name, **settings)
            results[(zipf_alpha, reference_name, "reference")] = row
            print_row(f"{reference_name} a={zipf_alpha}", "ref", row)

    return results, turnovers


def write_csv(results, turnovers, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["zipf", "policy", "period", "turnover_s", *REPORTED])

        for (zipf_alpha, policy_name, period), row in results.items():
            writer.writerow(
                [
                    zipf_alpha,
                    policy_name,
                    period,
                    f"{turnovers.get((zipf_alpha, policy_name), float('nan')):.2f}",
                    *(f"{row[key]:.5f}" for key in REPORTED),
                ]
            )

    print(f"wrote {path}")


PANELS = (
    ("hit_rate", "prefix hit rate", False),
    ("ttft_p99", "TTFT p99 (s)", True),
    ("view_fp_rate", "view false positive rate", False),
    ("routing_regret_rate", "routing regret rate", False),
)


def plot_sweep(results, policy_names, reference_names, periods, zipfs, *, include_shadow, path):
    figure, axes = plt.subplots(
        len(zipfs),
        len(PANELS),
        figsize=(4.5 * len(PANELS), 3.8 * len(zipfs)),
        squeeze=False,
    )

    for row_index, zipf_alpha in enumerate(zipfs):
        for axis, (metric_key, title, log_scale) in zip(axes[row_index], PANELS):
            for policy_name in policy_names:
                rows = [results[(zipf_alpha, policy_name, period)] for period in periods]
                ages = [row["mean_view_age"] for row in rows]
                axis.plot(
                    ages,
                    [row[metric_key] for row in rows],
                    "o-",
                    label=policy_name,
                )

                if include_shadow:
                    shadow = results[(zipf_alpha, policy_name, "shadow")]
                    axis.axhline(
                        shadow[metric_key],
                        linestyle=":",
                        color=axis.lines[-1].get_color(),
                        label=f"{policy_name} shadow" if metric_key == "hit_rate" else None,
                    )

            # the blind references never read a view, so they are horizontal
            if metric_key in ("hit_rate", "ttft_p99"):
                for reference_name in reference_names:
                    reference = results[(zipf_alpha, reference_name, "reference")]
                    axis.axhline(
                        reference[metric_key],
                        linestyle="--",
                        color="grey",
                        alpha=0.7,
                        label=reference_name if metric_key == "hit_rate" else None,
                    )

            workload_label = "trace" if zipf_alpha != zipf_alpha else f"zipf {zipf_alpha}"
            axis.set_title(f"{title} ({workload_label})")
            axis.set_xlabel("mean observed view age (s)")
            axis.set_xscale("symlog", linthresh=1.0)
            axis.grid(alpha=0.3)

            if log_scale:
                axis.set_yscale("log")

        axes[row_index][0].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"wrote {path}")


def plot_scaling(twin_results, twin_turnovers, policy_names, periods, twin_rates, zipf_alpha, path):
    """Is the damage set by arrivals per refresh, or by cache churn per refresh?

    Left: x = rate * period, the batched-allocation prediction. Right:
    x = mean age / cache turnover time. Whichever makes the two rates
    collapse onto one curve is the knob that matters.
    """
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    line_styles = {twin_rates[0]: "-", twin_rates[1]: "--"}

    for row_index, (metric_key, title, log_scale) in enumerate(PANELS[:2]):
        for policy_index, policy_name in enumerate(policy_names):
            color = f"C{policy_index}"

            for rate in twin_rates:
                rows = [twin_results[(rate, policy_name, period)] for period in periods]
                ages = [row["mean_view_age"] for row in rows]
                values = [row[metric_key] for row in rows]
                turnover = twin_turnovers[(rate, policy_name)]

                axes[row_index][0].plot(
                    [rate * period for period in periods],
                    values,
                    line_styles[rate],
                    marker="o",
                    color=color,
                    label=f"{policy_name} rate {rate:g}",
                )
                axes[row_index][1].plot(
                    [age / turnover for age in ages],
                    values,
                    line_styles[rate],
                    marker="o",
                    color=color,
                )

        for axis, x_label in zip(
            axes[row_index],
            ("arrivals per refresh (rate x period)", "mean view age / cache turnover time"),
        ):
            axis.set_title(f"{title} (zipf {zipf_alpha})")
            axis.set_xlabel(x_label)
            axis.set_xscale("symlog", linthresh=1.0)
            axis.grid(alpha=0.3)
            if log_scale:
                axis.set_yscale("log")

    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep how stale the router's cache view can be before prefix-aware routing stops paying."
    )
    parser.add_argument("--policies", default="longest_prefix,hybrid,dualmap", help="cache-aware policies to sweep (default: longest_prefix,hybrid,dualmap)")
    parser.add_argument("--reference", default="p2c,round_robin", help="cache-blind policies drawn as flat lines (default: p2c,round_robin)")
    parser.add_argument("--periods", default="0,0.5,1,2,5,10,30", help="snapshot refresh periods in seconds, 0 for the perfect view (default: 0,0.5,1,2,5,10,30)")
    parser.add_argument("--zipf", default="0.9,1.2", help="prefix skews, one panel row each (default: 0.9,1.2)")
    parser.add_argument("--rate", type=float, default=6.0, help="arrival rate for the main figure (default: 6.0)")
    parser.add_argument("--twin-rates", default="4,8", help="two rates for the scaling figure, at the first zipf (default: 4,8)")
    parser.add_argument("--no-shadow", action="store_true", help="skip the shadow index view")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per point (default: 3)")
    parser.add_argument("--requests", type=int, default=3_000, help="requests per run (default: 3000)")
    parser.add_argument("--workers", type=int, default=4, help="workers (default: 4)")
    parser.add_argument("--cache-blocks", type=int, default=256, help="blocks per worker (default: 256)")
    parser.add_argument("--worker", choices=["sequential", "batched"], default="batched", help="worker model (default: batched)")
    parser.add_argument("--prefill-budget", type=int, default=0, help="batched worker prompt tokens per iteration, 0 unchunked (default: 0)")
    parser.add_argument("--output-prefix", default="out/staleness", help="figure and csv prefix (default: out/staleness)")
    parser.add_argument("--trace", default=None, help="replay a mooncake jsonl trace instead of the synthetic workload; zipf, rate and the scaling figure are then not applicable")
    parser.add_argument("--speedup", type=float, default=1.0, help="trace only: compress arrival gaps by this factor (default: 1.0)")
    parser.add_argument("--c-prefill", type=float, default=C_PREFILL, help=f"seconds per uncached prompt token (default: {C_PREFILL})")
    args = parser.parse_args()

    policy_names = resolve_policy_names(args.policies)
    reference_names = resolve_policy_names(args.reference)
    periods = [float(period) for period in args.periods.split(",")]
    zipfs = [float(zipf) for zipf in args.zipf.split(",")]
    twin_rates = [float(rate) for rate in args.twin_rates.split(",")]
    include_shadow = not args.no_shadow

    common = dict(
        n_workers=args.workers,
        n_requests=args.requests,
        cache_blocks=args.cache_blocks,
        worker_kind=args.worker,
        prefill_budget=args.prefill_budget or None,
        c_prefill=args.c_prefill,
    )

    rate = args.rate
    seeds = args.seeds

    if args.trace is not None:
        # the trace fixes the arrival process, so its own rate is what the
        # turnover time and the arrivals-per-refresh axis must use. only p2c
        # draws random numbers, so extra seeds would just repeat the same run
        replayed = load_mooncake(args.trace, speedup=args.speedup, limit=args.requests)
        rate = len(replayed) / replayed[-1].arrival
        seeds = 1
        zipfs = [float("nan")]
        common.update(
            workload=trace_workload(args.trace, speedup=args.speedup, limit=args.requests),
            block_size=MOONCAKE_BLOCK_SIZE,
            n_requests=len(replayed),
        )
        trace_stem = os.path.splitext(os.path.basename(args.trace))[0]
        args.output_prefix = f"{args.output_prefix}_{trace_stem}"
        print(
            f"replaying {args.trace}: {len(replayed)} requests over "
            f"{replayed[-1].arrival:.0f}s = {rate:.2f} req/s at speedup {args.speedup}"
        )

    os.makedirs("out", exist_ok=True)

    results, turnovers = sweep(
        policy_names,
        reference_names,
        periods,
        zipfs,
        include_shadow=include_shadow,
        seeds=seeds,
        rate=rate,
        **common,
    )
    for (zipf_alpha, policy_name), turnover in turnovers.items():
        print(f"cache turnover a={zipf_alpha} {policy_name}: {turnover:.1f} s")

    write_csv(results, turnovers, f"{args.output_prefix}_sweep.csv")
    plot_sweep(
        results,
        policy_names,
        reference_names,
        periods,
        zipfs,
        include_shadow=include_shadow,
        path=f"{args.output_prefix}_sweep.png",
    )

    if args.trace is not None:
        # arrival rate is not a knob on a trace, so there is no scaling figure
        return

    # the scaling figure: same policies, the first zipf, two rates
    twin_results = {}
    twin_turnovers = {}
    for rate in twin_rates:
        for policy_name in policy_names:
            rows_by_period = sweep_policy(
                policy_name,
                periods,
                seeds=args.seeds,
                rate=rate,
                zipf_alpha=zipfs[0],
                **common,
            )
            for period, row in rows_by_period.items():
                twin_results[(rate, policy_name, period)] = row
                print_row(f"{policy_name} r={rate:g}", period, row)
            twin_turnovers[(rate, policy_name)] = cache_turnover_seconds(
                rows_by_period[PERFECT],
                cache_blocks=args.cache_blocks,
                n_workers=args.workers,
                n_requests=args.requests,
                rate=rate,
            )
            print(f"cache turnover r={rate:g} {policy_name}: {twin_turnovers[(rate, policy_name)]:.1f} s")

    plot_scaling(
        twin_results,
        twin_turnovers,
        policy_names,
        periods,
        twin_rates,
        zipfs[0],
        f"{args.output_prefix}_scaling.png",
    )


if __name__ == "__main__":
    main()
