# methodology: claims, evidence, and artifact register

the working rule for every headline claim: it must rest on at least two of
{synthetic control, trace validation, real-engine measurement, production
corroboration}. the synthetic workload is the controlled instrument (one
variable at a time), the traces are the validity instrument, the real engine
closes the execution-model gap, and production corroboration means a shipping
system independently built an ad-hoc version of the mechanism. a
synthetic-only result is a hypothesis, not a claim.

## instruments

- **simulator** (`sim/`): discrete-event, clock validated against m/m/1;
  batched workers with chunked prefill; radix caches with lru-on-leaves;
  view models (perfect / snapshot / shadow / overlay) and treatments (ttl /
  survival); per-decision error accounting (view fp, view fn, regret,
  execution fp). costs are placeholders until calibration: latency numbers
  are relative, hit rates and orderings are the results.
- **synthetic workloads**: poisson arrivals over zipf prefixes (+ universal
  blocks knob); session-structured generator planned (R3) for nonstationary
  reuse.
- **traces**: mooncake toolagent + conversation (primary), mooncake arxiv +
  mooncake synthetic (secondary, R2); all one vendor, stated as a limitation;
  replayed at native timestamps, deterministic for the policies used.
- **real engine** (planned, C1/C3): vllm with prefix caching on mac hardware;
  the staleness mechanism needs correct cache dynamics, not fast tokens, so
  slow backends are evidentially fine for hit-rate and view-error claims.

## claims and their legs

| claim | synthetic | trace | real/production grounding | falsifier | carrying figure |
|---|---|---|---|---|---|
| C1 view error is linear in age/turnover with slope = cold share; the survival model (nested losses, per-worker rescue, measured residence cdf) predicts fp within 2x where fp > 1% | 4 capacities, 2 policies | both traces, predicted fp within ~25% with trace-measured inputs, zero refitting | che/aoi literature for the form | prediction off >2x where fp > 1% | theory_check |
| C2 pure ranking + universal prefix locks onto one worker at any view freshness; a match threshold / load term / deep hash cap each cure it fully | controlled k-sweep, perfect view locks in identically | discovered on both traces (p99 1546s / p50 2017s) | sglang ships the threshold | a spaced-arrival workload with k>=1 where pure lpm does not lock | lockin |
| C3 capacity raises reuse almost only through cache-aware routing; blind routing loses the ordering at any capacity | - | both traces, 4 capacities, global-pool + ceiling bounds | reuse-distance literature | routing gain shrinking at high capacity | capacity_map |
| C4 a cold replica costs utilisation, not fleet damage; naive warm-up ~ 1 T_C; prewarm the only arm that works for every policy family; affinity-only shields saturate on capacity-limited traffic | 4 arms, 2 policies | toolagent: ranker sends the newcomer zero requests; shield p99 110s; prewarm 0.16-0.46 T_C | warmserve et al. (weights); chwbl for the load-bounded shield | a fleet-level dip dominating the utilisation effect | scale_out both scales |
| C5 an oversized record-insert index collapses cardinal scorers in proportion to how worker-differential the promised overlap is; rankers immune; ttl at the turnover recovers 80-90% where it bites | shadow ablation at 6 capacities (differential zipf overlap: collapse) | toolagent 1x/2x/4x (common-mode universal overlap: no collapse — the falsifier fired and *scoped* the claim: score differences cancel common-mode lies) | llm-d approximate mode is the default record-insert | a worker-differential workload where 4x overgrowth leaves scorers intact (session workload run is the discriminating case) | treatment_* + toolagent ablation + R3 sessions |
| C6 a scrape's loss is its blind spot; overlay + survival discount makes scorers flat to 1.5 T_C | full treatment matrix | null where staleness doesn't bite (prod-size caches); carried by C5's regime | dynamo fixed expiry + llm-d speculative entries are the two halves | overlay+survival worse than raw anywhere | treatment_overlay_survival |
| C7 scale-in: the coldest member of a ranker fleet is its miss absorber and not free to remove | 3 victim rules x 3 policies, near the knee | **measured and demoted (5/9)**: on toolagent at moderate load every victim rule is within noise — specialisation needs skew and pressure; the claim is conditional on operating near the knee under pure affinity | - | (fired: coldest ≈ random on the trace) | scale_in_* both scales |
| C8 no collapse under stale views: cache state is slow state (the thesis) | full staleness map | both traces (staleness nearly irrelevant at trace cache sizes) | contra stale-queue load balancing (mitzenmacher) | any crossover below blind within one turnover | staleness sweeps + (C-track) real engine |

## artifact register

| # | risk | status | evidence / next |
|---|---|---|---|
| 1 | free in-flight kv sharing inflates trace hit rates | **measured, resolved**: `kv_available_at` both ways, every hit rate within 0.5 pt | both settings reported |
| 2 | same-instant trace timestamps (trace coarseness is itself an artifact) | **measured, informative**: lock-in with ties (record-insert only) and without (any view) | both regimes in C2's evidence |
| 3 | stationary hot set: fixed zipf prefixes + poisson flatter staleness results; real reuse is nonstationary sessions (within-session serial deep reuse vs cross-session concurrent shallow reuse) | **open** | R3 session generator; re-run staleness law, ordinal/cardinal, lock-in under churn; theory predicts the slope rises with cold share |
| 4 | placeholder worker costs: knees, p99s, the shield's 110s are relative numbers | **bounded next, fixed later** | R5 cost-sensitivity (orderings must survive x0.5/x2); C2 mac calibration for absolutes |
| 5 | cache capacity decoupled from batch occupancy (real engines share hbm between running kv and prefix cache; a busy worker has a smaller effective cache) | **open** | R6(A) coupled-capacity flag, one staleness sweep |
| 6 | seed/ci discipline: headline synthetic tables at 1-3 seeds without intervals | **open** | R4: 10 seeds + bootstrap ci on headline tables; trace runs labeled deterministic |
| 7 | single trace vendor (all mooncake) | **partially addressable** | R2 adds arxiv + synthetic mooncake traces (different domains, same vendor); qwen-bailian wishlist; limitation stated |
| 8 | execution model: linear costs, fixed batch, no preemption, no tensor parallelism, no multi-tenancy | **open, the deepest one** | C1/C3 real-engine legs; claims are about caching + routing, which the real engine exercises directly |

## statistics policy (R4 target)

synthetic: headline tables at 10 seeds, mean with 95% bootstrap ci, paired
seeds across policies (same arrivals). traces: deterministic for all reported
policies (only p2c draws randomness), so single runs reported exactly and
labeled as such; no ci is honest there. windowed series: window size stated
per figure; no smoothing.

## prediction-before-run rule

every new experiment writes its expected outcome and falsifier into the
script docstring or notes before the run (as the lock-in sweep did). wrong
predictions are kept in the notes with the correction; three so far
(burstiness blamed for the session-key failure, herding-on-bursts for the
lock-in, capacity-limited for the conversation trace) plus two model
corrections (nested losses, per-worker rescue) and one covariate rejection
(depth, screened off by idle age).
