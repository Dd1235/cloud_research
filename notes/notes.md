- M/M/1 - standard shorthand from Kendall's queue notation
    - Markovian service times, and Markovian arrival, one server
    - Markovian means memoryless
    - poisson, independent arrivals at rate lambda, independent exponential service times at rate mu, one server, fifo queue, stable system (lam < mu)
    - effectively steady-state operation

- For an ideal M/M/1 queue
    - mean latency = 1 / (mu - lam)
    - p99 = ln(100) / (mu - lam)

- ttft - time to first token = queue wait + prefill time
- tpot - time per output token = (finish - first_token) / (output_tokens - 1)
- prefill - per prompt token
- decode - per output token

- at the same target per-worker utilization, two workers will have lower ttft than one. round robin alternates requests, making each workers arrival stream more regular than the original poisson stream. that smoothing reduces queue bursts. property of the simple deterministic round robin.


- python built-in string hash is intentionally randomized between processes so routing would not be stable across runs
- a consistent-hash ring also, on adding/removing worker, remaps roughly 1/n keys

- Dual Map : session_hash gets a great hit rate, but can get hotspots. also zipf skew. hash the same session key to two rings with different slats, theres a power-of-two-choices trick applied to affinity. it can improve hitrate as well, two warm workers per prefix evict less. power of two trick is Instead of sending a request to one random worker, pick two candidate workers and choose the better one.. 
- p2c, implicitly discovers its lower capacity from its queue/load, in the heterogenous case.


- in our 2/9/26 sim, p2c is better over random, but not round robin in homogeneous case, that is because round-robin splits a poisson stream deterministically, there is lower variance, and there is lower queueing. 

- tbt - time between tokens. long gaps can get averaged away in tpot


* **Chunked prefill** splits a long prompt across multiple decode iterations instead of processing it all at once.
* This greatly reduces long pauses between generated tokens. Example: worst gap went **10s → 0.52s**.
* **TPOT stays unchanged** because it’s an average; it hides occasional huge stalls. So `tbt_p99` is a better user-experience metric.
* Chunking slightly worsens the long request’s own **TTFT**, because its prompt takes more iterations to finish.
* For short ~600-token prompts, chunking is actually worse because there wasn’t much stall to fix. For ~8k prompts, it helps massively (**16.8s → 0.29s TBT p99**).
* `NO_TOKEN` means: during a partial prefill iteration, that request did computation but **did not generate a token**. So you must not record `token_time` or decrement `tokens_left`.

Core takeaway: **chunking trades a bit of TTFT for much smoother token streaming, but only becomes worthwhile for sufficiently long prompts.**

- cache views (2/9/26): a production router never reads a worker's real cache. what it reads is a *view*, and the view is stale in one of three ways
    - A. snapshot: copy the cache every P seconds (a metrics scrape). age is uniform in [0, P), mean P/2. this is what the mac cluster will physically do
    - C. shadow: the router keeps its own prefix cache and inserts whatever it routed (record-insert). it never sees worker evictions. llm-d "approximate" mode
    - B. delayed events: the true cache as of t - d (kv events over a bus). not built yet, needs insert/evict hooks on the cache
    - policies read `worker.view.match(...)`; the worker itself always reads `worker.cache`. with the perfect view they are the same object
    - the knob is the refresh *period*; what matters is the *age* the router actually saw at each decision, so the router records `view_age_at_dispatch` and plots use the measured mean age, not P

- what a wrong view costs, split by where the error lives (all as a fraction of prompt tokens)
    - view fp: the view promised tokens the chosen worker did not hold (metadata error, stale-positive)
    - view fn: the chosen worker held tokens the view did not know about (metadata error, stale-negative)
    - routing regret: a warmer worker existed at the instant of the decision (decision error; hybrid has this even with a perfect view, by design, because it trades locality for balance)
    - execution fp: promised tokens gone by the time the worker admitted the request. equals view fp plus the drift between dispatch and admission; at rate 6 the drift is ~0.001, so the error really is in the view
    - the perfect view is *admission* truth, not routing knowledge: two requests arriving at the same instant are both routed before the worker admits either, so the second sees nothing cached even though it will reuse the first's prefill. the shadow view does see the first dispatch. that is why shadow can beat perfect on bursts

- normalising staleness: two candidates for the x axis
    - arrivals per refresh, rate x P (balls-into-bins theory: the batch size)
    - age / cache turnover time, where turnover = capacity / eviction rate measured from the perfect-view run (~15s here). a snapshot older than one turnover describes a cache that has been fully replaced since
    - measured: view fp is about 5% of prompt tokens per turnover of age, and hybrid churns its caches faster (turnover 10s) so it is more staleness-sensitive at the same age. that points at turnover as the right normalisation, the scaling figure checks it

- daemon events: the engine now takes `schedule(..., daemon=True)`. a daemon (view refresh, load sampler) reschedules itself forever, so it must not count as work; `run()` stops when no live events remain, `run(until=T)` executes everything up to T including daemons. without this a snapshot view would keep the simulation alive forever

- `queue_cv` vs `load_cv`: load_cv is the CoV of total busy time per worker, i.e. whether the work was shared. queue_cv is the CoV of the time-averaged outstanding count from a 100ms sampler, i.e. whether the *queueing* was shared. a worker that is swamped half the time and idle the other half has a fine load_cv and a bad queue_cv

- session keys on real traces (2/9/26): hashing policies (session hash, dualmap) need a key that names the *session*, and "first block" is the wrong key on real traces. the conversation trace has one distinct first block across 6000 requests and toolagent has four: a system / tool prompt everyone shares. keyed on it, the entire trace hashes onto one or two workers, which looks like burstiness but is not. the synthetic workload hid this because its block 0 *is* the prefix id
    - a deeper fixed key works (16 blocks = 8k tokens: thousands of distinct keys, hottest 0.3%). dualmap's paper does it adaptively, growing the hashed prefix for keys whose traffic share exceeds 2/n. that adaptive version is the next step; the fixed depth is its floor
    - with a sane key dualmap is the best-balanced policy on both traces and its hit rate is flat across staleness, because the view only ever ranks two hash-pinned candidates. hashing bounds how wrong a stale view can make you
    - record-insert (shadow view) plus pure longest-prefix ranking is the one combination that explodes on real traces: the shadow sees each dispatch immediately, every follower of a burst matches on that worker, and a universal prefix means every request is a follower. a load term (hybrid) or a hash cap (dualmap) stops the cascade
