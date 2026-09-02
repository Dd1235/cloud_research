# Prefix-Aware LLM Serving Plan

Build a trace-driven simulator and router that study how KV-cache-aware routing behaves when the router's cache view is stale.

## Steps

### 1. Build the simulator

**Tools:** Python, `heapq`, NumPy, pandas, pytest

**Build:** Event loop, request/worker model, prefix cache with LRU eviction, sequential and continuous-batching workers.

**Policies:** Round-robin, least-outstanding, power-of-two choices, session hash, longest-prefix match, and hybrid cache/load scoring.

**Goal:** Establish a reproducible simulator with TTFT, TPOT/TBT, cache-hit rate, goodput, and load-balance metrics.

**Resources:** [Vidur](https://github.com/microsoft/vidur), [PagedAttention](https://arxiv.org/abs/2309.06180), [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

### 2. Validate the simulator on workloads

**Tools:** Trace replayers, synthetic Zipf workload generator, seeded experiments, pandas/matplotlib

**Build:** Replay available traces and generate workloads with controllable prefix sharing, request lengths, and arrival rates.

**Goal:** Confirm that prefix affinity improves cache reuse and median latency, while identifying its hotspot and tail-latency cost.

**Resources:** Mooncake traces, Qwen-Bailian traces, [RadixAttention](https://arxiv.org/abs/2312.07104)

### 3. Model stale and approximate cache views

**Tools:** Simulator view modules and experiment scripts

**Build:** Compare periodic snapshots, delayed/lost KV events, and router-maintained shadow indexes. Sweep view delay, load, prefix skew, and cache capacity.

**Goal:** Measure when stale information removes the benefit of prefix-aware routing.

**Metrics:** False-hit rate, false-miss opportunity, overlap error, herd/hotspot index, TTFT, goodput, and load balance.

**Resources:** [llm-d KV-cache-aware routing](https://llm-d.ai/docs/getting-started/architecture), [Dynamo](https://github.com/ai-dynamo/dynamo)

### 4. Add robust routing policies

**Tools:** Python policy interface, NumPy, offline parameter sweeps

**Build:** Implement and compare hybrid scoring, DualMap-style two-choice routing, shadow-index routing, and telemetry-age-aware scoring.

**Goal:** Identify policies that retain cache benefits without creating severe hotspots under stale views.

**Resources:** [DualMap](https://arxiv.org/abs/2602.06502), [Preble](https://arxiv.org/abs/2407.00023), [LMETRIC](https://arxiv.org/abs/2603.15202)

### 5. Evaluate online learning

**Tools:** NumPy, logged request contexts/actions/rewards, pandas

**Build:** Start with UCB policy selection and hybrid-weight tuning, then evaluate LinUCB over workers. Compare greedy and learning-based policies across staleness levels.

**Goal:** Determine whether exploration can recover performance lost to stale cache metadata.

**Resources:** [LinUCB](https://arxiv.org/abs/1003.0146), [Lodestar](https://arxiv.org/abs/2606.00946)

### 6. Build the live router and mock workers

**Tools:** Go, `net/http`, goroutines, gRPC or Redis streams, Docker Compose, kind/k3d

**Build:** Router with a filter/score/pick policy interface, OpenAI-compatible mock workers, per-worker prefix caches, KV-event reporting, and injected staleness.

**Goal:** Demonstrate cache-blind, cache-aware, and stale-view routing on the same trace.

### 7. Add observability and hardware validation

**Tools:** OpenTelemetry, Prometheus, Grafana, Python analysis notebooks, vLLM on available hardware

**Build:** Trace routing, queueing, prefill, decode, cache usage, and request outcomes. Calibrate simulator service times and cache behavior against heterogeneous workers.

**Goal:** Produce an end-to-end latency decomposition and quantify the difference between simulated and observed routing behavior.

**Resources:** [OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/), [Prometheus](https://prometheus.io/docs/), [vllm-metal](https://github.com/vllm-project/vllm-metal)
