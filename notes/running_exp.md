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


