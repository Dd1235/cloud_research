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


