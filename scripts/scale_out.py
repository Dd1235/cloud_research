"""What a cold worker costs, and whether shielding or prewarming pays (E14/E15).

Finding 7 said memory is only worth buying with a router that can use it. An
autoscaler buys memory *cold*: a new worker holds nothing, so a cache-aware
router either ignores it (no match anywhere) or feeds it misses. This script
adds an empty worker mid-run and watches the windowed time series through the
event, under four arms:

  none     no scale-out, the baseline the dip is measured against
  naive    the worker simply joins the routing set
  shield   for one turnover it receives only low-affinity orders (best match
           on the veterans below a fraction of the prompt), so it builds a
           shelf out of work the veterans were bad at anyway
  prewarm  it joins only after a transfer delay, with the hottest trunks of
           the recent past already inserted (kv_available_at = prefill_done in
           spirit: nothing is matchable before it has arrived)

Scale-in is the same story backwards: --remove-at takes one worker away, and
which one (coldest / random / busiest) is the E15 comparison.
"""
import argparse
import csv
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from compare_policies import POLICY_SEED_OFFSET, build_workers, trace_workload
from sim.traces import MOONCAKE_BLOCK_SIZE
from sim.engine import Engine
from sim.metrics import summarize, windowed_series
from sim.policies import make_policy
from sim.router import Router
from sim.sampler import OutstandingSampler
from sim.workload import generate

ARMS = ("none", "naive", "shield", "prewarm")


class ShieldedPolicy:
    """Route to the newcomer only what the veterans are cold for, until `until`.

    A cache-aware policy would starve a cold worker (it never has the best
    match) or, blind ones, would feed it the same mix as everyone and pay a
    miss on every popular prefix. The shield sends it exactly the orders whose
    best veteran match is below `threshold` of the prompt: work that was going
    to be mostly-miss wherever it landed. After `until`, the inner policy sees
    the full set.
    """

    def __init__(self, inner, engine, newcomer_id, until: float, threshold: float = 0.25):
        self.inner = inner
        self.engine = engine
        self.newcomer_id = newcomer_id
        self.until = until
        self.threshold = threshold

    def choose(self, req, workers):
        veterans = [worker for worker in workers if worker.id != self.newcomer_id]

        if self.engine.now >= self.until or len(veterans) == len(workers):
            return self.inner.choose(req, workers)

        best_veteran_match = max(worker.view.match(req.blocks) for worker in veterans)
        if best_veteran_match < self.threshold * len(req.blocks):
            return next(worker for worker in workers if worker.id == self.newcomer_id)

        return self.inner.choose(req, veterans)


def hot_trunks(requests, *, before: float, window: float, min_count: int, budget_blocks: int):
    """The block paths worth copying to a new worker: recent, and shared.

    Blocks are counted over the trailing window; each request then contributes
    its leading run of blocks that cleared min_count. Interior blocks count at
    least as often as their descendants, so this keeps trunks and drops the
    unique tails, which is the point: a tail would be dead weight on arrival.
    """
    recent = [req for req in requests if before - window <= req.arrival < before]
    counts = Counter(block for req in recent for block in req.blocks)

    paths = []
    total = 0
    for req in sorted(recent, key=lambda req: req.arrival, reverse=True):
        trunk = []
        for block in req.blocks:
            if counts[block] < min_count:
                break
            trunk.append(block)

        if trunk:
            paths.append(tuple(trunk))
            total += len(trunk)
        if total >= budget_blocks:
            break

    return paths


def run_arm(arm, *, seed, policy_name, n_workers, cache_blocks, requests_for, policy_options,
            add_at, shield_turnovers, shield_threshold, prewarm_transfer_turnovers,
            prewarm_min_count, turnover, prefill_budget=None, remove_at=None, remove_which=None,
            worker_settings=None):
    engine = Engine(seed)
    workers = build_workers(engine, "batched", n_workers + 1, cache_blocks, prefill_budget,
                            **(worker_settings or {}))
    veterans, newcomer = workers[:n_workers], workers[n_workers]

    policy = make_policy(policy_name, np.random.default_rng(seed + POLICY_SEED_OFFSET), policy_options)
    requests = requests_for(seed)
    span = requests[-1].arrival
    t_add = add_at * span

    if arm == "shield":
        policy = ShieldedPolicy(policy, engine, newcomer.id, until=t_add + shield_turnovers * turnover,
                                threshold=shield_threshold)

    router = Router(engine, policy, list(veterans))
    sampler = OutstandingSampler(engine, router.workers)

    if arm == "naive" or arm == "shield":
        engine.schedule(t_add, router.add_worker, newcomer)
    elif arm == "prewarm":
        def join_prewarmed():
            for path in hot_trunks(requests, before=engine.now, window=turnover,
                                   min_count=prewarm_min_count, budget_blocks=cache_blocks):
                newcomer.cache.insert(path, now=engine.now)
            router.add_worker(newcomer)

        engine.schedule(t_add + prewarm_transfer_turnovers * turnover, join_prewarmed)

    if remove_at is not None:
        t_remove = remove_at * span

        def remove_one():
            candidates = router.workers
            if remove_which == "busiest":
                victim = max(candidates, key=lambda worker: worker.outstanding)
            elif remove_which == "random":
                victim = candidates[np.random.default_rng(seed).integers(len(candidates))]
            else:   # coldest: least reuse per token processed so far
                victim = min(candidates, key=lambda worker: (
                    worker.tokens_reused / worker.tokens_processed if worker.tokens_processed else 0.0
                ))
            print(f"    remove_at t={t_remove:.0f}s: removing worker {victim.id} ({remove_which})")
            router.remove_worker(victim.id)

        engine.schedule(t_remove, remove_one)

    router.replay(requests)
    engine.run()

    return dict(
        requests=requests,
        workers=workers,
        newcomer_id=newcomer.id,
        t_add=t_add,
        summary=summarize(requests, workers, sampler=sampler),
    )


def measure_turnover(*, seed, policy_name, n_workers, cache_blocks, requests_for, policy_options,
                     prefill_budget=None, worker_settings=None):
    """One plain run to read the per-worker cache turnover off the eviction counters."""
    engine = Engine(seed)
    workers = build_workers(engine, "batched", n_workers, cache_blocks, prefill_budget,
                            **(worker_settings or {}))
    policy = make_policy(policy_name, np.random.default_rng(seed + POLICY_SEED_OFFSET), policy_options)
    requests = requests_for(seed)
    router = Router(engine, policy, workers)
    router.replay(requests)
    engine.run()

    duration = requests[-1].arrival
    evictions_per_worker = sum(worker.cache.evictions for worker in workers) / n_workers
    if evictions_per_worker == 0:
        return float("inf")

    return cache_blocks / (evictions_per_worker / duration)


def main():
    parser = argparse.ArgumentParser(description="Cold-replica scale-out and scale-in time series.")
    parser.add_argument("--policies", default="longest_prefix,hybrid")
    parser.add_argument("--arms", default="none,naive,shield,prewarm")
    parser.add_argument("--rate", type=float, default=8.0, help="near the knee for the blind policies (default: 8)")
    parser.add_argument("--zipf", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-blocks", type=int, default=256)
    parser.add_argument("--universal-blocks", type=int, default=0)
    parser.add_argument("--window", type=float, default=5.0, help="series window in seconds (default: 5)")
    parser.add_argument("--add-at", type=float, default=0.4, help="scale-out instant as a fraction of the run (default: 0.4)")
    parser.add_argument("--shield-turnovers", type=float, default=1.0)
    parser.add_argument("--shield-threshold", type=float, default=0.25)
    parser.add_argument("--prewarm-transfer-turnovers", type=float, default=0.25)
    parser.add_argument("--prewarm-min-count", type=int, default=3)
    parser.add_argument("--remove-at", type=float, default=None, help="scale-in instant as a fraction; enables E15")
    parser.add_argument("--remove-which", choices=["coldest", "random", "busiest"], default="coldest")
    parser.add_argument("--trace", default=None, help="replay a mooncake jsonl trace instead of the synthetic workload")
    parser.add_argument("--session-depth", type=int, default=1)
    parser.add_argument("--prefill-budget", type=int, default=0)
    parser.add_argument("--c-prefill", type=float, default=None)
    parser.add_argument("--c-decode", type=float, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--output-prefix", default="out/scale_out")
    args = parser.parse_args()

    os.makedirs("out", exist_ok=True)
    arms = args.arms.split(",")

    worker_settings = {}
    if args.trace is not None:
        requests_for = trace_workload(args.trace, speedup=1.0, limit=args.requests)
        worker_settings["block_size"] = MOONCAKE_BLOCK_SIZE
        trace_stem = os.path.splitext(os.path.basename(args.trace))[0].replace("_trace", "")
        args.output_prefix = f"{args.output_prefix}_{trace_stem}"
        args.seeds = 1   # a trace is deterministic for these policies
    else:
        requests_for = lambda seed: generate(
            np.random.default_rng(seed), args.requests, args.rate,
            zipf_alpha=args.zipf, universal_blocks=args.universal_blocks,
        )
    if args.c_prefill is not None:
        worker_settings["c_prefill"] = args.c_prefill
    if args.c_decode is not None:
        worker_settings["c_decode_batched"] = args.c_decode
    if args.max_batch is not None:
        worker_settings["max_batch"] = args.max_batch

    policy_options = {"key_blocks": args.session_depth}
    common = dict(
        n_workers=args.workers, cache_blocks=args.cache_blocks, requests_for=requests_for,
        policy_options=policy_options, prefill_budget=args.prefill_budget or None,
        worker_settings=worker_settings,
    )

    rows = []
    for policy_name in args.policies.split(","):
        turnover = measure_turnover(seed=0, policy_name=policy_name, **common)
        print(f"{policy_name}: measured turnover {turnover:.1f}s "
              f"(shield {args.shield_turnovers * turnover:.0f}s, "
              f"prewarm transfer {args.prewarm_transfer_turnovers * turnover:.0f}s)")

        for arm in arms:
            for seed in range(args.seeds):
                result = run_arm(
                    arm, seed=seed, policy_name=policy_name, add_at=args.add_at,
                    shield_turnovers=args.shield_turnovers, shield_threshold=args.shield_threshold,
                    prewarm_transfer_turnovers=args.prewarm_transfer_turnovers,
                    prewarm_min_count=args.prewarm_min_count, turnover=turnover,
                    remove_at=args.remove_at, remove_which=args.remove_which, **common,
                )
                newcomer_id = result["newcomer_id"]
                for point in windowed_series(result["requests"], window=args.window):
                    rows.append(dict(
                        policy=policy_name, arm=arm, seed=seed, t=point["t"], n=point["n"],
                        hit_rate=point["hit_rate"], ttft_p50=point["ttft_p50"], ttft_p99=point["ttft_p99"],
                        newcomer_share=point["share_by_worker"].get(newcomer_id, 0.0),
                        newcomer_hit=point["hit_rate_by_worker"].get(newcomer_id, float("nan")),
                        t_add=result["t_add"], turnover=turnover,
                    ))
                if seed == 0:
                    summary = result["summary"]
                    print(f"  {arm:>8} seed 0: hit={summary['hit_rate']:.3f} "
                          f"p50={summary['ttft_p50']:.3f} p99={summary['ttft_p99']:.2f}")

    with open(f"{args.output_prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_prefix}.csv")

    # dip accounting against the no-scale-out arm, seed-averaged per window
    def series_of(policy_name, arm, field):
        by_t = {}
        for row in rows:
            if row["policy"] == policy_name and row["arm"] == arm:
                by_t.setdefault(row["t"], []).append(row[field])
        return {t: float(np.mean(values)) for t, values in sorted(by_t.items())}

    policies = args.policies.split(",")
    figure, axes = plt.subplots(3, len(policies), figsize=(7 * len(policies), 10), squeeze=False, sharex=True)
    for column, policy_name in enumerate(policies):
        t_add = next(row["t_add"] for row in rows if row["policy"] == policy_name)
        turnover = next(row["turnover"] for row in rows if row["policy"] == policy_name)
        baseline = series_of(policy_name, "none", "hit_rate")

        for index, arm in enumerate(arms):
            hit = series_of(policy_name, arm, "hit_rate")
            p99 = series_of(policy_name, arm, "ttft_p99")
            newcomer_hit = series_of(policy_name, arm, "newcomer_hit")
            axes[0][column].plot(list(hit), list(hit.values()), color=f"C{index}", label=arm, lw=1.2)
            axes[1][column].plot(list(p99), list(p99.values()), color=f"C{index}", lw=1.2)
            if arm != "none":
                axes[2][column].plot(list(newcomer_hit), list(newcomer_hit.values()), color=f"C{index}", lw=1.2)

                shared = [t for t in hit if t in baseline and t >= t_add]
                dip = sum((baseline[t] - hit[t]) * args.window for t in shared if baseline[t] > hit[t])
                gain = sum((hit[t] - baseline[t]) * args.window for t in shared if hit[t] > baseline[t])
                warm = [t for t in shared
                        if not np.isnan(newcomer_hit.get(t, float("nan")))
                        and newcomer_hit[t] >= 0.9 * baseline[t]]
                time_to_warm = (warm[0] - t_add) if warm else float("nan")
                print(f"{policy_name:>15} {arm:>8}: dip integral {dip:6.1f} hit·s, gain {gain:6.1f} hit·s, "
                      f"time-to-warm {time_to_warm:6.1f}s = {time_to_warm / turnover:5.2f} T_C")

        for row_axes in axes:
            row_axes[column].axvline(t_add, color="grey", linestyle=":", lw=1)
            row_axes[column].grid(alpha=0.3)
        axes[0][column].set_title(f"{policy_name} (T_C {turnover:.0f}s, scale-out at {t_add:.0f}s)")
        axes[0][column].set_ylabel("fleet hit rate")
        axes[1][column].set_ylabel("ttft p99 (s)")
        axes[2][column].set_ylabel("newcomer hit rate")
        axes[2][column].set_xlabel("time (s)")
        axes[0][column].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}.png", dpi=120)
    print(f"wrote {args.output_prefix}.png")


if __name__ == "__main__":
    main()
