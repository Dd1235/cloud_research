- 4 workers, 2 request/sec, 4000 req, 1ms/prompt token, 10ms/output tokens, 256 cached blocks per worker, and zipf alpha = 1.0
    - we see lpm routes requests to cache-warm workers so avoids too much prompt refill
    - concetrated workers may have worse p95, p99 at ttft

    