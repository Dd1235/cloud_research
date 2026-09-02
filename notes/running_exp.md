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


- theory check (2/9/26, E11): does the router's view age at the cache's characteristic time? longest prefix and hybrid, 4 batched workers, 6 req/s, zipf 0.9, cache 128 / 256 / 512 / 1024 blocks, snapshot periods 0.5..30s, 3 seeds. the perfect-view run logs every eviction (block, inserted, last access, evicted). scripts/theory_check.py, out/theory_check.png/.csv
    - cache turnover (capacity / eviction rate) scales with capacity as it should: longest prefix 4.9 / 14.4 / 34.5 / 75.0s at 128 / 256 / 512 / 1024 blocks; hybrid 3.8 / 10.1 / 27.2 / 67.9s, shorter at every size because it spreads prefixes and churns
    - che's picture is too clean for a radix cache. idle time before eviction at 256 blocks: p10 5.6s, p50 8.7s, p90 13.0s, against a 14.4s turnover. the cache evicts leaves first, and leaves are the unique tail of a prompt while the shared beginning sits near the root, so the garnish goes long before the base sauce. che's fixed point from per-worker block rates gives 7.5s, close to the median, not the turnover
    - age / turnover linearises every false-positive curve at every capacity (measured fp is ~linear in it up to about one turnover), but the curves do not collapse: at the same age / turnover, fp is higher for a small cache (128: 0.040 at 0.2 turnovers; 256: 0.018 at 0.17; 512: 0.004 at 0.15). the slope is the share of promised blocks that are cold, and a bigger cache promises mostly hot prefix blocks. so the law is fp ≈ (cold share) x age / turnover, with a workload- and capacity-dependent coefficient, not a universal curve
    - the model's own prediction of fp is off in both directions depending on the assumption. che's deterministic lifetime (a block is gone iff its idle age exceeds the turnover) under-predicts by ~5x at moderate age (256 blocks, age 2.5s: predicted 0.003, measured 0.018), because blocks die well before the turnover. the measured residence distribution, applied correctly as the hazard over the scrape interval, over-predicts by 10-20x at small ages (256 blocks, age 0.25s: predicted 0.042, measured 0.002). the second error is a population mismatch: the residence distribution is measured on *evicted* blocks, mostly short-lived leaves, while the blocks a router promises are matched prefix blocks, the long-lived survivors. their hazard is an order of magnitude lower
    - so: the mechanism (view error grows with age in units of cache turnover, slope = cold share) is confirmed and explains the staleness sweep and hybrid's extra sensitivity; the quantitative predictor needs the residence distribution of the *promised* population, e.g. conditioned on depth in the tree or on having been re-referenced. that is the next iteration of the model, and until then the formula is a shape, not a number

-  view fp, longest prefix     age/T_C   measured   predicted (step)   predicted (cdf, hazard)
  C=256, P=0.5                  0.017     0.0021        0.0030              0.0418
  C=256, P=5                    0.173     0.0180        0.0025              0.1425
  C=256, P=30                   1.038     0.0525        0.0638              0.1260
  C=128, P=5                    0.507     0.0673        0.0889              0.2928
  C=512, P=30                   0.435     0.0127        0.0096              0.0597
  C=1024, P=30                  0.200     0.0040        0.0000              0.0137


- theory check, corrected model (2/9/26, E11 final): same setup as the theory check above, after two changes to how the survival view turns per-block survival into an expected match. supersedes the prediction columns above. scripts/theory_check.py, scripts/fp_by_age.py, out/theory_check.png/.csv, out/fp_by_age.csv
    - first, a diagnostic: bin every promised block by how old the view believed it to be (router `record_block_samples`). at 0.25s view age the residence-cdf hazard is only ~2x off per block (blocks believed 5-8s old: measured 1.3% false, hazard 3.1%; 8-12s: 4.3% vs 7.7%; below 5s both are zero), while the view's aggregate prediction was 15x off. so the residence distribution was roughly right and the way per-block survivals were combined was wrong
    - correction 1, nested losses: the expected surviving match was Σ_j Π_{i≤j} S_i, every block an independent coin flip. in a radix cache a block is only evicted after everything below it, so "block j is present" implies its ancestors are, and the expected depth is Σ_j S_j (running-min clamped). the product turned a 3% per-block risk into a 27% loss over a 20-block path
    - correction 2, rescue at the chosen worker: a stale entry is rescued only if *this* worker saw the block again, so the rescue rate is the per-(worker, block) dispatch rate, not the fleet rate. with the fleet rate hybrid, which spreads a prefix across workers, was rescued too often and under-predicted 2.6x at 1.5 turnovers of age; per-worker rates bring it to 0.85x
    - result: predicted fp within 2x of measured at every point where measured fp exceeds 1%, within 1.4x at ages above a quarter turnover, ~2x over at the smallest ages (the leaf-heavy eviction log; absolute error below half a point there) and ~20% under past one turnover. the shape from the first check stands: fp is linear in age / turnover with a capacity-dependent slope, the cold share of promised blocks: longest prefix 0.19 / 0.11 / 0.03 / 0.02 per turnover at 128 / 256 / 512 / 1024 blocks, hybrid 0.26 / 0.21 / 0.13 / 0.05 (churn plus spreading)
    - what this buys: the router can now say, per decision, how much of a promise to believe, from things it already has (the snapshot's last-access ages, its own dispatch counts, the worker's eviction counter). the survival treatment in E12 was measured with the product form and is being re-run

-  view fp, corrected model      age/T_C   measured   predicted
  longest prefix, C=128, P=2      0.20      0.040      0.067
  longest prefix, C=128, P=10     1.02      0.090      0.097
  longest prefix, C=256, P=5      0.17      0.018      0.031
  longest prefix, C=256, P=30     1.04      0.053      0.044
  longest prefix, C=512, P=30     0.44      0.013      0.018
  hybrid, C=128, P=5              0.65      0.109      0.131
  hybrid, C=256, P=5              0.25      0.055      0.074
  hybrid, C=256, P=30             1.49      0.100      0.084
  hybrid, C=512, P=30             0.55      0.053      0.062
  hybrid, C=1024, P=30            0.22      0.010      0.022


- toolagent trace under realistic kv timing (2/9/26): the toolagent sweep above re-run with `--kv-available-at prefill_done`, so a request's blocks count as cached only after the iteration that prefilled them, and a same-iteration follower recomputes the shared prefix instead of sharing it for free. everything else identical. out/staleness_toolagent_prefill_done_sweep.png/.csv
    - both burst findings stand. herding relief: longest prefix hit rate 0.407 → 0.430 from a fresh to a 10s view, queue cv 0.53 → 0.12, p99 6.9 → 5.8s (free sharing: 0.408 → 0.431, 0.52 → 0.12, 6.6 → 6.2s). shadow cascade: longest prefix + shadow p99 1517s, queue cv 2.64, hit rate 0.337 below blind (free sharing: 1546s, 2.64, 0.329); hybrid + shadow 6.3s and dualmap + shadow 6.0s are fine either way
    - hit rates move by less than half a point at every period for every policy, so free in-flight sharing was not inflating the trace numbers: at 6 req/s over 8 workers, same-iteration followers of the same prefix are rare enough not to register in the total, and the herding effect is about *where* a burst lands, not about whether its first two members share a prefill
    - the caveat on findings 4 and 7 is closed; both settings are reported and agree

-  toolagent, prefill_done         P=0     0.5      1       2       5      10      30   shadow
  longest prefix hit rate       0.407   0.408   0.411   0.412   0.427   0.430   0.426   0.337
  longest prefix queue cv       0.525   0.523   0.525   0.520   0.242   0.116   0.090   2.644
  longest prefix ttft p99 (s)   6.91    6.85    6.52    6.66    6.26    5.80    6.33   1517
  hybrid hit rate               0.419   0.419   0.422   0.423   0.421   0.421   0.420   0.417
  hybrid ttft p99 (s)           6.34    6.34    6.46    6.50    6.25    6.22    6.41    6.31
  dualmap hit rate              0.414   0.413   0.413   0.413   0.414   0.413   0.411   0.414
  dualmap ttft p99 (s)          6.09    6.11    6.11    6.11    6.05    6.09    6.09    5.96
  p2c / round robin hit rate    0.336 / 0.344, p99 6.76 / 7.06


- stale view treatments, corrected survival view (2/9/26, E12 addendum): the survival treatment above was measured with the product-form expected match and fleet-wide rescue rates. re-run after the two corrections from the theory check (nested losses, rescue at the chosen worker), still with che's step lifetime at each policy's own turnover. same setup, cardinal scorers reading `--overlap-source expected`. out/treatment_survival_sweep.png/.csv now hold this run
    - the treatment is no longer harmful on snapshots. hybrid at 15s mean age 0.537 (raw 0.539, product form 0.510); lmetric 0.600 (raw 0.599, was 0.563). the product form was throwing away most of a long match on the strength of one uncertain block; summing nested survivals keeps it
    - dynamo cost, the scorer with the most false positives (0.13 of prompt tokens at 15s), is the one it helps: 0.475 vs 0.469 raw at 2.5s mean age, 0.470 vs 0.450 at 5s, 0.457 vs 0.447 at 15s, recovering about 40% of what the stale view cost it. hybrid and lmetric, with fewer false positives, are unchanged within noise
    - the model's own predicted false positives are still 3-20x below the measured ones here (hybrid 0.041 predicted vs 0.076 measured at 15s, 0.003 vs 0.051 at 2.5s), because the step lifetime says nothing is gone before one turnover of idleness and most evictions happen well before that (p50 8.7s against 14.4s). the treatment therefore barely discounts at the ages where a snapshot actually lies. the residence-cdf lifetime, which the theory check validated, is the next run
    - rankers unchanged, as before

-  hit rate, corrected survival     P=0     0.5      1       2       5      10      30
  hybrid raw                      0.597   0.596   0.595   0.593   0.566   0.549   0.539
  hybrid survival (step)          0.597   0.595   0.590   0.591   0.562   0.546   0.537
  lmetric raw                     0.673   0.678   0.671   0.666   0.641   0.623   0.599
  lmetric survival (step)         0.673   0.676   0.666   0.666   0.646   0.621   0.600
  dynamo cost raw                 0.499   0.497   0.487   0.490   0.469   0.450   0.447
  dynamo cost survival (step)     0.499   0.497   0.495   0.483   0.475   0.470   0.457


- stale view treatments, survival with the residence-cdf lifetime (2/9/26, E12 final): same sweep as the addendum above, the survival view now fed the residence-time cdf measured on each policy's own perfect run (`--survival-lifetime cdf`) instead of che's step, with nested losses and per-worker rescue. out/treatment_survival_cdf_sweep.png/.csv
    - the model is calibrated in the loop, not just in the theory check: the view's own predicted false positives track the measured ones within ~1.5x at every age (hybrid 0.057 predicted vs 0.037 measured at 2.5s mean age, 0.072 vs 0.058 at 5s, 0.059 vs 0.071 at 15s; dynamo cost 0.077 / 0.052, 0.108 / 0.091, 0.092 / 0.099)
    - and a calibrated discount still does not raise hit rate on snapshots. hybrid 0.567 / 0.547 / 0.536 at 2.5 / 5 / 15s against raw 0.566 / 0.549 / 0.539; lmetric a point lower than raw (0.633 / 0.606 / 0.590 vs 0.641 / 0.623 / 0.599); dynamo cost, the scorer with the most false positives, gains 0.3 / 1.2 / 1.3 points (0.472 / 0.462 / 0.460 vs 0.469 / 0.450 / 0.447), about what the step gave it
    - the error accounting says why. the discount does what it should to the promises: hybrid's measured false positives fall by a third (0.100 → 0.071 at 15s). but they are traded for false negatives (0.031 → 0.046) and the routing regret, the share of tokens a warmer worker held at the instant of the decision, does not move (0.132 raw, 0.130 discounted, 0.087 with a fresh view). a discount can only lower the router's trust in what the scrape says; the loss under a scrape is what the scrape does not say
    - what a scrape does not say is mostly the router's own doing: every request it dispatched since the scrape is invisible to it, and those are exactly the warm spots it created. the router has that information for free. so the treatment for a snapshot is the opposite of a discount: overlay the dispatches since the last refresh (llm-d's speculative entries, made principled), which attacks the blind spot rather than the stale promises. that run is next
    - the two view models now have two different diseases and two different cures: a record-insert index that outgrew the cache is false-positive dominated and a survival discount or ttl at the turnover repairs it (80-90% recovered); a periodic scrape is blind-spot dominated and only fresher information repairs it

-  hit rate, cdf survival          P=0     0.5      1       2       5      10      30
  hybrid raw                      0.597   0.596   0.595   0.593   0.566   0.549   0.539
  hybrid survival (cdf)           0.597   0.596   0.595   0.585   0.567   0.547   0.536
  lmetric raw                     0.673   0.678   0.671   0.666   0.641   0.623   0.599
  lmetric survival (cdf)          0.673   0.672   0.665   0.654   0.633   0.606   0.590
  dynamo cost raw                 0.499   0.497   0.487   0.490   0.469   0.450   0.447
  dynamo cost survival (cdf)      0.499   0.493   0.487   0.482   0.472   0.462   0.460
  hybrid view fp, raw / cdf       0 / 0   .004/.004 .008/.006 .021/.014 .055/.037 .086/.058 .100/.071
  hybrid regret, raw / cdf        .087    .085/.091 .085/.092 .088/.096 .112/.111 .124/.128 .132/.130


- the overlay: a scrape plus the router's own dispatches since it (2/9/26, E13): same sweep, the snapshot view now keeps every dispatch since its last refresh in a small router-side prefix cache and reads the longer of the two matches; the refresh drops the overlay, since the copy then shows what the worker really kept (`--treatment overlay`, SnapshotView(overlay=True)). the router learns nothing new: it only stops forgetting what it did. out/treatment_overlay_sweep.png/.csv
    - for the rankers staleness disappears. longest prefix holds 0.723-0.726 at every period, where the raw scrape fell 0.723 → 0.645 at 15s; regret 0.000 and view fn 0.000 throughout. its false positives still grow with age (0.026 at 15s) and, ordinal, it does not care
    - the cardinal scorers recover to fresh-view quality up to about half a turnover. lmetric 0.673 at 2.5s mean age, equal to its perfect view (raw 0.641), 0.662 at 5s (raw 0.623), 0.639 at 15s (raw 0.599); hybrid 0.589 / 0.584 / 0.550 at 2.5 / 5 / 15s (raw 0.566 / 0.549 / 0.539, fresh 0.597); dynamo cost 0.490 / 0.474 / 0.456 (raw 0.469 / 0.450 / 0.447). hybrid's regret is 0.082-0.099 up to 5s of age against 0.087 fresh: the decisions are as good as with a fresh view
    - beyond half a turnover the other disease takes over. hybrid's false positives reach 0.151 at 15s (raw 0.100: the overlay keeps promising dispatches the worker has since evicted, on top of the scrape's) and its regret climbs back to 0.128. that is the regime the survival discount was built for, so the complete treatment is overlay plus survival, run next
    - the accounting is now clean. a scrape's loss is its blind spot (fn and regret, both zero once the router remembers its own dispatches); a record-insert index's loss is its stale promises (fp, repaired by a discount or a ttl at the turnover). every production router does one or the other: dynamo's fixed expiry and llm-d's speculative entries are the two halves of the same fix

-  hit rate, overlay               P=0     0.5      1       2       5      10      30
  longest prefix raw              0.723   0.716   0.715   0.714   0.697   0.692   0.645
  longest prefix overlay          0.723   0.725   0.726   0.724   0.725   0.724   0.725
  hybrid raw                      0.597   0.596   0.595   0.593   0.566   0.549   0.539
  hybrid overlay                  0.597   0.602   0.601   0.595   0.589   0.584   0.550
  lmetric raw                     0.673   0.678   0.671   0.666   0.641   0.623   0.599
  lmetric overlay                 0.673   0.677   0.678   0.676   0.673   0.662   0.639
  dynamo cost raw                 0.499   0.497   0.487   0.490   0.469   0.450   0.447
  dynamo cost overlay             0.499   0.497   0.499   0.493   0.490   0.474   0.456
  hybrid regret raw / overlay     .087    .085/.082 .085/.083 .088/.087 .112/.094 .124/.099 .132/.128
  hybrid view fp raw / overlay    0       .004/.004 .008/.007 .021/.019 .055/.049 .086/.078 .100/.151


- capacity vs routing map on the traces (2/9/26, E16): first 6000 requests of toolagent and conversation at their own timestamps, 8 batched workers with the uncalibrated trace costs (0.05ms/prompt token, 0.25ms/decode step, max batch 64, chunked prefill 512), session depth 16, one seed, blocks per worker 256 / 1024 / 4096 / 16384. two bounds per trace: the reuse ceiling (share of prompt tokens whose blocks appeared in an earlier request, the hit rate of an infinite cache with perfect placement) and a global pool (one worker holding the whole fleet's blocks and batch slots; its latency is meaningless, its hit rate is what placement can never beat). free in-flight sharing (kv at admission), which the prefill-done rerun showed is worth under half a point here. scripts/capacity_map.py, out/capacity_map.png/.csv
    - the prediction was wrong, and usefully so. i expected the routing gain to peak where the cache lifetime is about one repeat gap and to shrink at both ends. it does not shrink at the top: it grows with capacity and saturates at the ceiling. conversation, longest prefix minus round robin: 0.015 / 0.115 / 0.229 / 0.229 at 256 / 1024 / 4096 / 16384 blocks; toolagent 0.015 / 0.064 / 0.146 / 0.156. a blind router spreads each session's turns over eight workers, so turn j hits only if an earlier turn happened to land on the same worker, and no amount of capacity fixes that: round robin at 16384 blocks, which never evicts, is 0.124 on conversation against a ceiling of 0.353. the ceiling is an ordering fact, and partitioning loses it at any capacity
    - so "reuse on the traces is capacity-limited, not routing-limited", which the earlier trace entries say, was half right. at 1024 blocks both bind. going to 4096 raises longest prefix 0.408 → 0.514 on toolagent and 0.180 → 0.329 on conversation, and round robin 0.344 → 0.368 and 0.065 → 0.100. capacity pays almost entirely through cache-aware routing; a router is what converts memory into hits. that is the cloud statement: the extra memory is only worth buying with a router that can use it
    - the perfect-view ranker is within 1-2 points of the global pool at every capacity on both traces (toolagent 0.408 vs 0.433 at 1024, 0.514 vs 0.525 at 4096, 0.527 vs 0.528 at 16384; conversation 0.180 vs 0.183, 0.329 vs 0.330, 0.353 vs 0.353). the fleet's partitioning costs almost nothing once placement is right. hybrid and dualmap pay 1-3 points for balance (queue cv 0.1-0.2 against longest prefix's 0.5), the same trade as on the synthetic workload
    - what capacity buys, in units that transfer: the share of the ceiling a router reaches against cache lifetime over median repeat gap. conversation (gap 123s, 10% of repeats same-instant): 16% of the ceiling at 0.22 gaps, 51% at 1.1, 93% at 7, all at never-evict. toolagent (gap 81s among the 48% of repeats that are not same-instant): 63% at 0.34, 77% at 1.7, 97% at 14. a cache lifetime of one median gap gets half to three quarters of the reachable reuse; several gaps get nearly all of it
    - turnover grows faster than linearly with capacity under cache-aware routing (toolagent longest prefix 27 → 136 → 1168s for 4x steps) because a warm cache evicts less: the same prefix stops being re-inserted on a cold worker. blind routing's turnover grows slower (27 → 122 → 825s), it keeps re-fetching
    - absolute hit rates here are relative until the mac calibration; the shape and the ordering are the result

-  hit rate by blocks per worker        256     1024    4096    16384
  toolagent  reuse ceiling 0.528
    global pool                       0.334   0.433   0.525   0.528
    longest prefix                    0.331   0.408   0.514   0.527
    hybrid                            0.334   0.418   0.507   0.510
    dualmap                           0.325   0.412   0.502   0.507
    round robin                       0.316   0.344   0.368   0.371
    cache lifetime / median gap       0.34    1.7     14      never evicts
  conversation  reuse ceiling 0.353
    global pool                       0.056   0.183   0.330   0.353
    longest prefix                    0.056   0.180   0.329   0.353
    hybrid                            0.052   0.167   0.302   0.319
    dualmap                           0.055   0.165   0.297   0.321
    round robin                       0.041   0.065   0.100   0.124
    cache lifetime / median gap       0.22    1.1     7       never evicts


- overlay plus survival: the complete repair of a scrape (2/9/26, E13 final): same sweep, `--treatment overlay_survival --survival-lifetime cdf --overlap-source expected`. the overlay closes the blind spot, the survival weight (nested losses, per-worker rescue, residence-cdf lifetime) discounts the promises that went stale since. out/treatment_overlay_survival_sweep.png/.csv
    - every cardinal scorer is flat across view age up to 15s, 1-1.5 turnovers, within about a point of its fresh view. hybrid 0.589-0.594 at every period (fresh 0.597; raw 0.539 at 15s; overlay alone 0.550); lmetric 0.652-0.673 (fresh 0.673; raw 0.599; overlay alone 0.639); dynamo cost 0.486-0.494 (fresh 0.499; raw 0.447; overlay alone 0.456). regret sits at the fresh view's level at every age: hybrid 0.090-0.098 against 0.087, lmetric 0.030-0.045 against 0.026, dynamo 0.177-0.181 against 0.164
    - the two halves do what the accounting said they would. false negatives are zero at every age (the overlay); hybrid's false positives at 15s are 0.084 with the discount against 0.151 with the overlay alone (the discount). the model's own prediction tracks the measured false positives within 1.5x throughout (hybrid 0.047 vs 0.030 at 2.5s, 0.061 vs 0.084 at 15s)
    - the rankers are untouched (longest prefix 0.724-0.726, dualmap 0.720-0.725), as at every step: they were never sick
    - what the router needs for this: the snapshot's last-access ages (already in any scrape), its own dispatch log (already in its memory), per-(worker, block) dispatch counts over a window (a dict), and one residence-time distribution per worker measured once from the eviction log, or che's step if the log is not available (the step version is never worse than raw and recovers part of dynamo's loss; the cdf version is the one that goes flat). nothing new from the engine
    - so the answer to the project's question, on the synthetic workload: a router's view of the caches can be a full cache turnover old and the routing is as good as fresh, provided the router remembers what it did since the scrape and discounts what the cache has probably dropped since. the two fixes are complements and each alone is partial. next: the same treatment on the traces, and a shadow index with the same survival weight (the record-insert counterpart)

-  hit rate, overlay + survival    P=0     0.5      1       2       5      10      30
  hybrid raw                      0.597   0.596   0.595   0.593   0.566   0.549   0.539
  hybrid overlay                  0.597   0.602   0.601   0.595   0.589   0.584   0.550
  hybrid overlay + survival       0.597   0.592   0.594   0.593   0.591   0.589   0.591
  lmetric raw                     0.673   0.678   0.671   0.666   0.641   0.623   0.599
  lmetric overlay                 0.673   0.677   0.678   0.676   0.673   0.662   0.639
  lmetric overlay + survival      0.673   0.671   0.673   0.668   0.656   0.655   0.652
  dynamo cost raw                 0.499   0.497   0.487   0.490   0.469   0.450   0.447
  dynamo cost overlay             0.499   0.497   0.499   0.493   0.490   0.474   0.456
  dynamo cost overlay + survival  0.499   0.489   0.486   0.489   0.486   0.494   0.488
  hybrid regret, overlay + surv.  .087    .093    .090    .092    .097    .098    .098
