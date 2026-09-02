- 4 workers, 2 request/sec, 4000 req, 1ms/prompt token, 10ms/output tokens, 256 cached blocks per worker, and zipf alpha = 1.0
    - we see lpm routes requests to cache-warm workers so avoids too much prompt refill
    - concetrated workers may have worse p95, p99 at ttft

-                          Round robin   Longest prefix
TTFT p50                    0.256          0.216
TTFT p95                    0.768          1.049
TTFT p99                    1.124          1.803
prefix hit rate             44.1%          71.8%
load CV                      0.004          0.414
evictions                   93,004         46,421


- load sweep (2/9/26): 4 batched workers (8ms fixed per iteration + 2ms per decode step, 1ms/prompt token, unchunked), 3000 req, 256 cached blocks per worker, zipf alpha = 1.0, rates 2..12 req/s, 3 seeds (medians). scripts/load_sweep.py, out/load_sweep.png
    - cache blind policies (round robin, p2c) collapse between 8 and 10 req/s: p99 8s → 62s. hit rate is stuck at 0.44, so every request pays ~56% of its prefill and that prefill serialises inside the iteration
    - cache aware policies still carry 12 req/s: hybrid p99 3.5s at rate 12 vs 108s for round robin. same 4 workers, ~1.5x the load before the tail explodes. prefix affinity is capacity, not just latency
    - session hash: best p50 at low load (0.191) and the worst p99 from rate 8 on (14.7s, then 85s). one worker per hot prefix is a hotspot
    - longest prefix is ragged at the knee (16.4s at 10, 5.7s at 12): pure locality herds onto warm workers. dualmap smooths that until 12 (23s), where two rings per prefix are no longer enough
    - hybrid has the best p99 at every rate. it gives up hit rate (0.60 vs 0.71) for balance (load CV 0.30 → 0.004) and wins the tail. its hit rate rises with load (0.60 → 0.67): with deep queues the load term ties more often and locality breaks the tie
    - tbt p99 climbs to ~1.2s for the blind policies at saturation: queue wait leaking into the token stream. chunking would not help here, the prompts are only ~600 tokens

-  TTFT p99 (s)            2       4       6       8      10      12
round robin           0.776   1.061   1.881   8.217  61.694 107.672
p2c                   1.091   1.531   2.327   5.376  60.635 107.343
session hash          0.798   1.145   1.887  14.741  85.512 132.841
longest prefix        0.760   0.834   1.053   1.445  16.355   5.681
hybrid                0.776   0.869   1.191   1.480   2.068   3.548
dualmap               0.760   0.888   1.159   1.692   3.536  23.366

-  prefix hit rate         2       4       6       8      10      12
round robin           44.7%   44.7%   44.6%   44.8%   44.8%   44.7%
session hash          70.9%   70.9%   70.8%   70.9%   70.9%   70.8%
longest prefix        70.6%   71.2%   71.9%   71.4%   72.3%   71.0%
hybrid                60.2%   58.5%   61.0%   63.8%   66.1%   67.0%
dualmap               71.6%   71.8%   71.9%   71.9%   71.8%   71.3%

-  load CV                 2       4       6       8      10      12
round robin           0.009   0.009   0.006   0.003   0.016   0.016
session hash          0.492   0.451   0.413   0.375   0.383   0.404
longest prefix        0.431   0.261   0.208   0.113   0.136   0.036
hybrid                0.299   0.113   0.048   0.021   0.005   0.004
dualmap               0.227   0.182   0.151   0.119   0.098   0.093



- staleness sweep (2/9/26): the router reads a *snapshot* of each worker's cache refreshed every P seconds (view model A, a metrics scrape), P in {0 = perfect, 0.5, 1, 2, 5, 10, 30}. 4 batched workers (8ms/iter + 2ms/decode step, 1ms/prompt token, unchunked), 6 req/s, 3000 req, 256 blocks/worker, zipf 0.9 and 1.2, 3 seeds (medians). x axis is the mean view age the router actually saw (~P/2). cache turnover = capacity / eviction rate from the perfect-view run: 14.7s longest prefix, 10.3s hybrid, 15.2s dualmap. scripts/staleness_sweep.py, out/staleness_sweep.png, out/staleness_scaling.png, out/staleness_sweep.csv
    - no crossover. a view one full cache turnover old (mean age 15s) still keeps most of the benefit: longest prefix hit rate 0.723 → 0.645 against 0.41 for the blind policies, dualmap 0.722 → 0.697. cache contents are slow state (a block cached 15s ago is usually still there); queue lengths are fast state. that is why the herding collapse from the stale-load literature does not show up here
    - the view error is small and grows about linearly with age: view fp for longest prefix is 0.4% of prompt tokens at 0.5s, 2.8% at 5s, 5.2% at 15s, roughly 5% per cache turnover of age. execution fp equals view fp everywhere (dispatch → admission drift ~0.1%), so the error is in the picture, not in the queue
    - dualmap is the most staleness-robust policy: hashing pins a prefix to two candidates, the view only ranks those two, and both are usually warm. routing regret at 15s age is 0.012, vs 0.033 longest prefix and 0.132 hybrid
    - hybrid is the most sensitive, the opposite of what I expected. its load term spreads a prefix over more workers, so its caches churn faster (turnover 10s vs 15s) and the same snapshot age is older in turnover units. its view fp is 2x longest prefix at every age, hit rate 0.597 → 0.539
    - shadow view (the router's own lru index at the worker's capacity, never sees evictions) is indistinguishable from the perfect view here: view fp 0.000-0.001, hit rate equal or slightly higher because it sees in-flight dispatches. a matched-capacity replica fed by the router's own dispatches reproduces the worker's evictions almost exactly. llm-d's approximate mode is much better than its docs suggest *when the capacities match*; the mismatch ablation is next
    - a little staleness helps pure affinity's tail: longest prefix p99 0.986 → 0.936 at 5s age, load cv 0.18 → 0.08, because a stale view stops it herding onto the currently warmest worker. the p2p paper's "staleness spreads hotspots", reproduced
    - at 6 req/s (below the knee at ~8) tail latency barely moves for longest prefix and dualmap; hybrid's p99 goes 1.19 → 1.65 (+39%). the scaling runs at rate 8 show the same: hybrid 1.64 → 2.18, dualmap 1.59 → 1.30 (improves), longest prefix flat
    - scaling figure: rate 4 and rate 8 overlap for longest prefix and dualmap on both x axes (arrivals per refresh, and age / cache turnover). on a fixed workload the two are proportional (turnover ~ 1/rate), so this run cannot tell them apart; a prompt-length sweep (blocks per request) is the experiment that can. hybrid overlaps on neither, because its perfect-view level itself depends on load

- longest prefix, zipf 0.9       P=0     0.5      1       2       5      10      30
mean view age (s)               0.00    0.25    0.49    1.00    2.50    5.01   15.00
prefix hit rate                72.3%   71.6%   71.5%   71.4%   69.7%   69.2%   64.5%
TTFT p99 (s)                   0.986   0.987   1.030   1.123   1.110   0.936   1.249
view fp rate                   0.000   0.002   0.004   0.008   0.018   0.028   0.052
routing regret                 0.000   0.001   0.001   0.005   0.009   0.013   0.033
(blind reference: hit rate 41.1-41.7%, TTFT p99 2.02-2.76)

- hybrid, zipf 0.9               P=0     0.5      1       2       5      10      30
prefix hit rate                59.7%   59.6%   59.5%   59.3%   56.6%   54.9%   53.9%
TTFT p99 (s)                   1.192   1.245   1.345   1.399   1.584   1.531   1.653
view fp rate                   0.000   0.004   0.008   0.021   0.055   0.086   0.100
routing regret                 0.087   0.085   0.085   0.088   0.112   0.124   0.132

- dualmap, zipf 0.9              P=0     0.5      1       2       5      10      30
prefix hit rate                72.2%   72.2%   72.4%   72.3%   70.9%   70.8%   69.7%
TTFT p99 (s)                   1.218   1.218   1.297   1.233   1.131   1.041   1.132
view fp rate                   0.000   0.002   0.002   0.007   0.015   0.024   0.034
routing regret                 0.000   0.000   0.001   0.001   0.006   0.006   0.012

- shadow view, zipf 0.9        longest prefix    hybrid    dualmap
prefix hit rate                     72.5%         60.7%     72.4%
view fp rate                        0.000         0.001     0.000


- shadow index capacity ablation (2/9/26): view model C, the router keeps its own lru prefix index per worker with capacity Ĉ and inserts what it routed; it never sees the worker evict. 4 batched workers, 6 req/s, 3000 req, worker cache 256 blocks, zipf 0.9, 3 seeds (medians). Ĉ swept from 32 to 1024 blocks; Ĉ = 256 is the matched case from the sweep above. scripts/compare_policies.py --view shadow --shadow-blocks Ĉ
    - two different failure modes, one per direction. too small: the router forgets blocks the worker still holds (false negatives), so it stops seeing warm workers and drifts toward blind routing. too big: the router remembers blocks the worker has evicted (false positives, capped at ~3.5% of prompt tokens by the eviction rate)
    - forgetting hurts everyone except dualmap: longest prefix hit rate 72.5% → 64.4% → 55.1% → 46.5% at Ĉ = 128, 64, 32 (blind is 41%). dualmap still 64.1% at Ĉ = 32, one eighth of the real cache, because the hash rings supply the affinity and the view only has to rank two candidates
    - remembering too much is harmless for policies that *rank* by overlap: longest prefix and dualmap keep 72% hit rate at Ĉ = 1024 despite 3.4% false positives, because a stale positive on the worker that *was* warm still ranks it first. it is catastrophic for hybrid, which *adds* overlap to a load term: its inflated overlap on a cold worker outvotes load, hit rate 60.7% → 46.7%, false positives 29%, regret 0.205
    - design rule that falls out: use the cache view ordinally (to rank) and it survives staleness in both directions; use it cardinally (in a weighted score) and false positives break it. any age or confidence discount (plan N2) matters for the cardinal scorers, not the rankers
    - side effect of over-remembering for longest prefix: queue cv 0.46 → 0.71 and p99 0.96 → 1.25. believing more workers are warm makes it herd harder

- shadow capacity Ĉ (cache 256)     32      64     128     256     512    1024
longest prefix hit rate          46.5%   55.1%   64.4%   72.5%   72.3%   72.5%
longest prefix regret            0.226   0.154   0.070   0.000   0.000   0.000
longest prefix view fp           0.014   0.008   0.001   0.000   0.033   0.034
longest prefix TTFT p99          2.071   1.868   1.302   0.958   1.033   1.249
hybrid hit rate                  44.2%   48.1%   54.0%   60.7%   54.3%   46.7%
hybrid regret                    0.241   0.207   0.151   0.083   0.137   0.205
hybrid view fp                   0.016   0.010   0.003   0.001   0.174   0.292
hybrid TTFT p99                  2.041   1.981   1.629   1.298   1.585   1.926
dualmap hit rate                 64.1%   66.1%   68.9%   72.4%   72.0%   72.2%
dualmap regret                   0.062   0.054   0.029   0.000   0.000   0.000
dualmap view fp                  0.001   0.002   0.001   0.000   0.037   0.037
dualmap TTFT p99                 1.407   1.468   1.307   1.278   1.149   1.195


- staleness sweep on the mooncake toolagent trace (2/9/26): first 6000 requests, replayed at their own timestamps (6.04 req/s over 993s), 512-token hash blocks, mean prompt 9.3k tokens (p50 6.4k), mean output 184, block reuse ceiling 0.553, 61% of repeats within 10s. 8 batched workers, max batch 64, chunked prefill 512, 1024 blocks/worker (524k tokens), *uncalibrated placeholder costs* 0.05ms/prompt token, 8ms/iter + 0.25ms/decode step (chosen so the perfect-view system is not saturated: longest prefix p50 0.19s). one seed (only p2c is random on a trace). cache turnover 113-139s. relative comparisons only until the mac calibration. scripts/staleness_sweep.py --trace --session-depth 16, out/staleness_toolagent_trace_sweep.png/.csv
    - staleness barely matters on this trace: a 17s-old view is 0.13 turnovers old, view fp ≤ 1.7%, regret ≤ 0.008. the synthetic sweep's "5% per turnover" holds up in order of magnitude; the turnover here is just 10x longer
    - longest prefix gets *better* with staleness: hit rate 0.408 → 0.431, queue cv 0.52 → 0.11, p99 6.56 → 5.98. with a fresh view it herds a burst of same-session requests onto the one warm worker; a stale view spreads them and the shared prefix ends up warm on several. herding relief is the dominant effect on a bursty trace
    - the cache-aware gain over blind is small here: hit rate 0.41-0.43 vs 0.34, p50 0.19-0.21s vs 0.30s, p99 ~6-7s for everyone (the bursts). the trace's reuse ceiling is 0.55 but the fleet holds 8192 blocks against ~180k distinct blocks per hour, so reuse is capacity-limited, not routing-limited
    - first run had dualmap at p99 279s, queue cv 1.73, hit rate = blind, and i first blamed burstiness. wrong: the session key was the *first block*, and this trace has only 4 distinct first blocks in 6000 requests (a shared tool prompt; the conversation trace has exactly 1). the whole trace hashed onto two of eight workers. the synthetic workload never showed this because there block 0 *is* the prefix id. dualmap's paper grows the hashed prefix when a key gets hot for exactly this reason; the fix here is a deeper fixed key (--session-depth 16, 8k tokens: 5537 distinct keys, hottest 0.3%)
    - with the deeper key dualmap is the best-balanced policy on this trace: hit rate 0.412 (longest prefix 0.408), p99 6.06s, queue cv 0.09 (longest prefix 0.52, hybrid 0.20), and flat across the whole staleness range. its hit rate does not move with age at all (0.412 → 0.410) because the view only ever ranks two hash-pinned candidates
    - the shadow view herds, but only under longest prefix: longest prefix + shadow p99 1545s, queue cv 2.64, hit rate below blind. the shadow sees the first request of a burst the moment it is routed, so every same-instant follower matches on that worker and piles on; and because every request shares the tool prompt, the pile never stops. with the perfect view the followers see nothing (the first is not admitted yet), tie, and spread by outstanding count. "shadow sees in-flight dispatches" was a small benefit on the synthetic workload and is the failure mode here. hybrid + shadow (p99 6.3s) and dualmap + shadow (5.98s) are fine: a load term or a hash cap stops the cascade
    - so: record-insert plus pure longest-prefix ranking is the dangerous combination on real traces, and a universal shared prefix is what makes it explode. anything that bounds the candidate set (hash) or pushes back on load (hybrid) is safe

- toolagent trace                 P=0     0.5      1       2       5      10      30   shadow
mean view age (s)               0.00    0.50    1.00    1.50    3.04    5.54   17.20    0.00
longest prefix hit rate        40.8%   40.8%   40.8%   40.8%   42.6%   43.1%   42.4%   32.9%
longest prefix TTFT p99 (s)     6.56    6.56    6.56    6.56    6.30    6.19    5.98  1545.6
longest prefix queue cv         0.52    0.52    0.52    0.52    0.26    0.12    0.11    2.64
hybrid hit rate                41.8%   41.8%   41.8%   41.8%   41.6%   42.3%   41.6%   42.1%
hybrid TTFT p99 (s)             6.22    6.22    6.22    6.22    6.29    6.39    6.45    6.33
dualmap hit rate (depth 16)    41.2%   41.2%   41.2%   41.2%   41.3%   41.2%   41.0%   41.4%
dualmap TTFT p99 (s)            6.06    6.06    6.06    6.06    6.06    6.09    6.20    5.98
dualmap queue cv                0.09    0.09    0.09    0.09    0.09    0.09    0.09    0.09
(dualmap with the one-block key, for the record: hit 33.3%, p99 279s, queue cv 1.73 at every age)
(blind reference: p2c hit 33.6% p99 6.76, round robin hit 34.4% p99 7.06)


- staleness sweep on the mooncake conversation trace (2/9/26): first 6000 requests at their own timestamps (3.21 req/s over 1872s), mean prompt 12.8k tokens (p50 7.6k, max 123k), mean output 347, block reuse ceiling 0.366, repeat gap p50 114s, and *every* request starts with the same block (a system prompt). same 8 uncalibrated batched workers as the toolagent run, session depth 16, one seed. cache turnover 128-131s. out/staleness_conversation_trace_sweep.png/.csv
    - cache-aware routing triples reuse over blind: hit rate 0.180 (longest prefix), 0.167 (hybrid), 0.165 (dualmap) vs 0.064. it is still well under the 0.37 ceiling because the typical repeat comes 114s later and the cache turns over in 130s: about half the reuse opportunities fall outside the cache lifetime. capacity-limited, not routing-limited
    - the latency gain is modest and only at the median: p50 0.61s vs 0.65 (round robin) / 0.78 (p2c); p99 6.8-7.5s for every policy, set by 100k-token prompts and bursts, not by routing
    - staleness is irrelevant again: view fp ≤ 1.7% at 16.6s mean age (0.13 turnovers), hit rate 0.180 → 0.174, regret ≤ 0.006. same order as toolagent and as the synthetic "5% per turnover"
    - no herding relief here, unlike toolagent: longest prefix queue cv is 0.12 flat. with repeats 114s apart there is no burst to herd. the two traces bracket the mechanism: staleness *helps* pure affinity exactly when repeats arrive faster than the router can spread them
    - dualmap with the deep key: hit rate 0.165, queue cv 0.10, flat across staleness; with the one-block key it was hit 0.048 and p50 451s, because all 6000 requests share block 0
    - longest prefix + shadow explodes here too: p50 2017s, queue cv 2.65, hit rate 0.044 (below blind). one universal first block, record-insert, pure ranking. hybrid + shadow and dualmap + shadow are unaffected

- conversation trace              P=0     0.5      1       2       5      10      30   shadow
mean view age (s)               0.00    0.50    1.00    1.51    3.02    5.52   16.62    0.00
longest prefix hit rate        18.0%   18.0%   18.0%   18.0%   18.6%   18.3%   17.4%    4.4%
longest prefix TTFT p50 (s)    0.605   0.605   0.605   0.605   0.613   0.635   0.636  2017.2
longest prefix queue cv         0.12    0.12    0.12    0.12    0.11    0.13    0.12    2.65
hybrid hit rate                16.7%   16.7%   16.7%   16.7%   16.5%   16.7%   16.0%   16.6%
hybrid TTFT p50 (s)            0.618   0.618   0.618   0.618   0.635   0.638   0.640   0.637
dualmap hit rate (depth 16)    16.5%   16.5%   16.5%   16.5%   16.6%   16.4%   16.4%   16.5%
dualmap TTFT p50 (s)           0.665   0.665   0.665   0.665   0.668   0.666   0.658   0.667
dualmap queue cv                0.10    0.10    0.10    0.10    0.10    0.10    0.10    0.10
(blind reference: p2c hit 6.4% p50 0.775, round robin hit 6.5% p50 0.647; p99 6.8-7.6s for everyone)


- treatments for a stale view (2/9/26, E12): three cardinal scorers (hybrid, lmetric = new prefill tokens x batch size, dynamo cost = prefill blocks + active blocks) and two rankers (longest prefix, dualmap), same setup as the staleness sweep (4 batched workers, 6 req/s, zipf 0.9, 256 blocks, 3 seeds). the view is read three ways: raw; ttl = ignore a block whose last access is older than the policy's own measured cache turnover; survival = weight each block by its chance of still being there (che step at the turnover, plus a rescue for re-references the scrape could not see). out/treatment_{raw,ttl,survival}_sweep.png/.csv. shadow index at 4x the worker's capacity with a ttl sweep in the second table
    - cache turnover by policy (perfect view): longest prefix 14.7s, dualmap 14.9s, lmetric 12.7s, hybrid 10.3s, dynamo cost 8.2s. the more a policy balances load, the more it spreads prefixes, the faster its caches churn. staleness sensitivity follows the same order
    - lmetric is a strong tuning-free baseline: perfect-view hit rate 0.673 and p99 1.13s beat hybrid (0.597, 1.19s). it degrades with staleness like the others (0.599 at 15s age)
    - on the over-sized shadow index (Ĉ = 4C, false positives 21-30% of prompt tokens) a ttl at the policy's own turnover recovers 80-90% of the lost hit rate: hybrid 0.467 → 0.591 (matched capacity 0.607), lmetric 0.553 → 0.660 (0.673), dynamo cost 0.432 → 0.486 (0.499). the best ttl sits at each policy's turnover (8-10s), and a ttl twice too long (15-20s) gives back half the gain. survival = ttl here, because a record-insert index refreshes its own timestamps and has no scrape age to rescue
    - on periodic snapshots neither treatment helps the cardinal scorers: at 15s mean age hybrid is 0.539 raw, 0.447 ttl, 0.510 survival; lmetric 0.599 / 0.458 / 0.563. the snapshot's false positives are only 5-10% there, so its *ranking* of workers is still right, and any discount shifts the scorer toward its load term and away from locality. ttl is the worst because it also cannot see re-references since the scrape (false negatives, regret 0.077 → 0.203 for lmetric)
    - the rankers are unaffected by every treatment, as they should be: longest prefix and dualmap read the view ordinally and the treatments only rescale it
    - so the treatment must depend on how wrong the view is: a discount pays when false positives dominate (a record-insert index that outgrew the cache) and costs when the ranking is intact (a scrape a turnover old). a router can measure its own false-positive rate from execution feedback (the engine reports cached tokens per request) and switch treatments on that. that is the next mechanism to build
    - the survival model's own prediction of the false-positive rate under-shoots on snapshots (hybrid 15s: predicted 0.050, measured 0.100), because a radix cache evicts leaves well before the turnover; see the theory check entry

-  shadow at 4C, hit rate    raw   ttl=5   ttl=8  ttl=10  ttl=12  ttl=15  ttl=20  ttl=30   matched Ĉ=C
hybrid  (T_C 10.3s)        0.467   0.574   0.591   0.579   0.566   0.550   0.528   0.499   0.607
lmetric (T_C 12.7s)        0.553   0.635   0.659   0.660   0.657   0.642   0.614   0.587   0.673
dynamo  (T_C  8.2s)        0.432   0.486   0.485   0.475   0.468   0.463   0.440   0.437   0.499
longest prefix (ranker)    0.725   0.715 (ttl=15)                                          0.725

-  snapshot, hit rate at mean age   0.25s   0.49s   1.0s    2.5s    5.0s    15s
hybrid raw                         0.596   0.595   0.593   0.566   0.549   0.539
hybrid ttl                         0.595   0.590   0.594   0.570   0.524   0.447
hybrid survival                    0.595   0.590   0.589   0.568   0.539   0.510
lmetric raw                        0.678   0.671   0.666   0.641   0.623   0.599
lmetric ttl                        0.676   0.666   0.665   0.637   0.599   0.458
lmetric survival                   0.676   0.666   0.666   0.642   0.611   0.563
dynamo raw                         0.497   0.487   0.490   0.469   0.450   0.447
dynamo survival                    0.496   0.496   0.480   0.471   0.458   0.439
longest prefix (any treatment)     0.716   0.715   0.714   0.697   0.692   0.645
