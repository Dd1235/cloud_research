"""When is routing the lever at all? Hit rate against cache capacity on the traces.

On the Mooncake traces reuse is capacity-limited: the median gap between two
uses of the same block (~114 s on the conversation trace) is about the cache
turnover at 1024 blocks per worker, so much of the reuse falls outside the
cache's lifetime whatever the router does. This sweeps blocks per worker and
draws, per policy, hit rate against capacity, with two bounds: the trace's
reuse ceiling (an infinite cache) and a global-pool oracle (one cache holding
the whole fleet's capacity, so placement can never lose a block). The routing
gain, cache-aware minus blind, is what matters: where it peaks is where a
router earns its keep, and where it vanishes the fix is capacity, not routing.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from compare_policies import run, trace_workload
from sim.traces import MOONCAKE_BLOCK_SIZE, load_mooncake

ORACLE = "global pool"


def trace_reuse(requests, block_size):
    """The trace's reuse ceiling and its median repeat gap.

    Ceiling: the share of prompt tokens whose blocks appeared in an earlier
    request, which is the hit rate of an infinite cache with perfect placement.
    Repeat gap: seconds between successive uses of the same block, whose
    median tells how long a cache has to hold a block for the reuse to count.
    Same-instant repeats (a burst of tool calls sharing a prefix) are counted
    but left out of the median: they are decided by in-flight sharing and
    batching, not by how long the cache remembers.
    """
    last_seen = {}
    gaps = []
    reusable_tokens = 0
    prompt_tokens = 0

    for req in requests:
        seen_before = 0
        for block in req.blocks:
            if block in last_seen:
                seen_before += 1
                gaps.append(req.arrival - last_seen[block])
            last_seen[block] = req.arrival

        reusable_tokens += min(seen_before * block_size, req.prompt_tokens)
        prompt_tokens += req.prompt_tokens

    positive_gaps = sorted(gap for gap in gaps if gap > 0)
    median_gap = positive_gaps[len(positive_gaps) // 2] if positive_gaps else float("inf")
    same_instant_share = 1 - len(positive_gaps) / len(gaps) if gaps else 0.0

    return reusable_tokens / prompt_tokens, median_gap, same_instant_share


def turnover_seconds(row, *, fleet_blocks, duration):
    evictions_per_second = row["evictions"] / duration
    return float("inf") if evictions_per_second == 0 else fleet_blocks / evictions_per_second


def main():
    parser = argparse.ArgumentParser(description="Hit rate vs cache capacity on the traces, per policy.")
    parser.add_argument("--traces", default="traces/toolagent_trace.jsonl,traces/conversation_trace.jsonl")
    parser.add_argument("--capacities", default="256,1024,4096,16384", help="blocks per worker (default: 256,1024,4096,16384)")
    parser.add_argument("--policies", default="round_robin,longest_prefix,hybrid,dualmap")
    parser.add_argument("--blind", default="round_robin", help="the cache-blind policy the routing gain is measured against")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--requests", type=int, default=6000)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--prefill-budget", type=int, default=512)
    parser.add_argument("--c-prefill", type=float, default=5e-5, help="uncalibrated, as in the trace sweeps")
    parser.add_argument("--c-decode", type=float, default=2.5e-4)
    parser.add_argument("--session-depth", type=int, default=16)
    parser.add_argument("--output-prefix", default="out/capacity_map")
    args = parser.parse_args()

    capacities = [int(c) for c in args.capacities.split(",")]
    policy_names = args.policies.split(",")
    os.makedirs("out", exist_ok=True)

    rows = []
    bounds = {}
    for trace_path in args.traces.split(","):
        requests = load_mooncake(trace_path, speedup=1.0, limit=args.requests)
        duration = requests[-1].arrival
        ceiling, median_gap, same_instant_share = trace_reuse(requests, MOONCAKE_BLOCK_SIZE)
        trace_name = os.path.splitext(os.path.basename(trace_path))[0].replace("_trace", "")
        bounds[trace_name] = dict(ceiling=ceiling, median_gap=median_gap)
        print(f"{trace_name}: {len(requests)} requests over {duration:.0f}s, reuse ceiling {ceiling:.3f}, "
              f"median repeat gap {median_gap:.1f}s ({same_instant_share:.0%} of repeats are same-instant)")

        common = dict(
            seed=0,
            rate=len(requests) / duration,
            n_requests=len(requests),
            zipf_alpha=float("nan"),
            worker_kind="batched",
            prefill_budget=args.prefill_budget,
            workload=trace_workload(trace_path, speedup=1.0, limit=args.requests),
            block_size=MOONCAKE_BLOCK_SIZE,
            c_prefill=args.c_prefill,
            c_decode_batched=args.c_decode,
            policy_options={"key_blocks": args.session_depth},
        )

        for cache_blocks in capacities:
            fleet_blocks = cache_blocks * args.workers

            # the oracle: one worker holding the fleet's blocks and batch slots.
            # its latency is meaningless (one machine doing eight machines' work)
            # but its hit rate is what perfect placement would reach
            oracle = run(args.blind, n_workers=1, cache_blocks=fleet_blocks,
                         max_batch=args.max_batch * args.workers, **common)
            runs = {ORACLE: oracle}

            for policy_name in policy_names:
                runs[policy_name] = run(policy_name, n_workers=args.workers, cache_blocks=cache_blocks,
                                        max_batch=args.max_batch, **common)

            for name, row in runs.items():
                turnover = turnover_seconds(row, fleet_blocks=fleet_blocks, duration=duration)
                rows.append(dict(
                    trace=trace_name, cache_blocks=cache_blocks, policy=name,
                    turnover_s=turnover, turnover_over_gap=turnover / median_gap,
                    hit_rate=row["hit_rate"], ttft_p50=row["ttft_p50"], ttft_p99=row["ttft_p99"],
                    queue_cv=row["queue_cv"],
                ))
                print(f"{trace_name:>13} C={cache_blocks:>6} {name:>15} T_C={turnover:8.1f}s "
                      f"T_C/gap={turnover / median_gap:6.2f} hit={row['hit_rate']:.3f} "
                      f"p50={row['ttft_p50']:.3f} p99={row['ttft_p99']:.2f} qcv={row['queue_cv']:.2f}")

    with open(f"{args.output_prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_prefix}.csv")

    traces = list(bounds)
    figure, axes = plt.subplots(1, len(traces), figsize=(6 * len(traces), 4.2), squeeze=False)
    for axis, trace_name in zip(axes[0], traces):
        of_trace = [row for row in rows if row["trace"] == trace_name]

        def series(name):
            return [row["hit_rate"] for row in of_trace if row["policy"] == name]

        axis.axhline(bounds[trace_name]["ceiling"], color="black", linestyle=":", label="reuse ceiling")
        axis.plot(capacities, series(ORACLE), "k--", label=ORACLE)
        for index, policy_name in enumerate(policy_names):
            axis.plot(capacities, series(policy_name), "o-", color=f"C{index}", label=policy_name)
        if args.blind in policy_names and "longest_prefix" in policy_names:
            axis.fill_between(capacities, series(args.blind), series("longest_prefix"),
                              color="C1", alpha=0.15, label="routing gain")

        # a second x axis in the units that decide whether reuse survives:
        # how many median repeat gaps one cache lifetime spans
        gaps = [row["turnover_over_gap"] for row in of_trace if row["policy"] == "longest_prefix"]
        for cache_blocks, ratio in zip(capacities, gaps):
            axis.annotate(f"T_C/gap {ratio:.1f}", (cache_blocks, 0.02), fontsize=7, ha="center",
                          xycoords=("data", "axes fraction"))

        axis.set_xscale("log", base=2)
        axis.set_xlabel("blocks per worker")
        axis.set_ylabel("hit rate")
        axis.set_title(f"{trace_name}: median repeat gap {bounds[trace_name]['median_gap']:.0f}s")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}.png", dpi=120)
    print(f"wrote {args.output_prefix}.png")


if __name__ == "__main__":
    main()
