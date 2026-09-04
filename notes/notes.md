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

- why the view ages at the cache's characteristic time (2/9/26, derived, to be checked in E11): che's approximation from web caching says an lru cache of C blocks behaves like an infinite cache where each block lives a fixed characteristic time T_C after its last access (T_C = time for C distinct blocks to be requested; a block with request rate λ hits with probability 1 - e^{-λ T_C}). our "cache turnover" = capacity / eviction rate is T_C in steady state
    - a block the view saw present at age a is still there iff it was accessed in the last T_C. either it was re-referenced during a (probability 1 - e^{-λ a}), or it was not and then it survives only if its residual life at snapshot time exceeded a: with the residual uniform on [0, T_C) that is 1 - a/T_C. so P[false positive | age a, rate λ] ≈ e^{-λ a} * a / T_C and survival S(a) = 1 - e^{-λ a} * a / T_C
    - what that predicts: view fp is linear in a / T_C with slope = the cold share of the promised blocks (measured ~5% per turnover on zipf 0.9); hybrid churns its caches faster (T_C 10s vs 15s) so the same age is older in turnover units; curves at different cache sizes should collapse on a / T_C and not on rate x period, because a capacity sweep at fixed rate changes T_C but not λP
    - the fix is a clock, not a weight: the snapshot already carries every block's last-access time, so "a block counts as present only while its last-access age < T_C" is one rule that works for snapshots, shadow indexes and event-fed indexes alike. dynamo's fixed 120s expiry and llm-d's speculative entries are ad-hoc versions of it. the refinement is the survival-weighted overlap Σ_j Π_{i≤j} S_i(age_i), which only helps if re-reference rates are worth tracking; plain ttl first
    - which policies need it: the ones that *add* overlap to a load term (hybrid, lmetric's product, dynamo's cost). the rankers (longest prefix, dualmap) are ordinal and shrug off false positives, as the shadow ablation showed. ordinal use of the view is robust, cardinal use is not
    - caveat: lru-on-leaves in a radix tree is not object lru (a parent block cannot be evicted while a child is touched), so measure the per-block residence-time distribution before trusting the uniform-residual step
    - in age-of-information terms the view age at dispatch is the AoI and a / T_C is its natural normalisation

- in-flight sharing (2/9/26): the batched worker does match then insert per admitted request, so a follower admitted in the same iteration as its leader hits on blocks nobody has computed yet. real engines make it wait or recompute. the "staleness helps pure affinity on bursts" and "shadow + longest prefix herds" numbers were measured under that free sharing and get re-measured with `kv_available_at = prefill_done` before they go anywhere

- cold replicas (2/9/26, planned): a new worker starts with an empty cache, so scaling out *lowers* fleet hit rate until it warms. che says warm-up takes about T_C. the serverless cold-start literature (keep-alive windows from inter-arrival histograms) is the same shape: cache retention from the repeat-gap histogram, and either shield a cold replica (only unseen prefixes for one T_C) or pre-warm it with the zipf head. cache warmth is a resource the autoscaler has to account for

- what the theory check taught (2/9/26): che's "every block lives exactly T_C after its last use" is the right *unit* and the wrong *shape* for a radix cache. leaf blocks (a prompt's unique tail) die at 5-13s, interior blocks (the shared beginning) live much longer, so residence is a spread, not a step
    - two ways to turn a residence distribution into "is this block still there": F(idle age) answers "what fraction of blocks get evicted by this idle age", which is the right question for a ledger that never observed survival; for a photo that *saw the block present* the right question is the hazard over the photo's age, (F(x) - F(x - a)) / (1 - F(x - a)), because the block had already survived to x - a. mixing these up over-predicts false positives 10-40x for fresh photos
    - even the hazard over-predicts by 10-20x, and the reason is survivorship: the eviction log is populated by the blocks that die young, and the blocks a router promises are the ones that did not. the residence distribution has to be measured on the promised population (by depth in the tree, or on blocks that were re-referenced at least once) before the formula gives numbers rather than shapes
    - what survives all of this: false positives grow linearly in age / turnover with slope = the cold share of promised blocks. that explains the 5% per turnover, why hybrid (shorter turnover) is more sensitive at the same age, and why bigger caches sit lower on the same axis

- what made the survival model quantitative (2/9/26, night): two structural facts about *where* the model is applied, not a better residence distribution
    - binning every promised block by how old the view believed it to be showed the per-block hazard from the residence cdf was only ~2x off (view age 0.25s: blocks believed 5-8s old were false 1.3% of the time, hazard said 3.1%; 8-12s: 4.3% vs 7.7%), while the view's aggregate prediction was 15x off. the error was in how per-block survivals were combined
    - losses along a matched path are nested, not independent. a radix cache evicts leaves first, so a block is only ever evicted after every block below it, and "block j is present" already implies its ancestors are. expected surviving depth is Σ_j S_j (running-min clamped), not Σ_j Π_{i≤j} S_i. the product treats each block as its own coin flip and turns a 3% risk per block into a 27% loss over a 20-block path
    - the rescue must be counted at the chosen worker. a block is refreshed only by a dispatch to *that* worker, so the rate that rescues a stale entry is the per-(worker, block) rate, not the fleet rate. with the fleet rate, hybrid (which spreads a prefix over the fleet) was rescued far too often and its false positives were under-predicted 2.6x at one turnover of age; per-worker rates bring it to 0.9x
    - after both: prediction within ~1.3x wherever false positives exceed 1%, ~2x over at the smallest ages. the remaining 2x is the leaf-heavy eviction log (survivorship), and it sits where it does not matter
    - the general lesson: an estimator that is right per element can be badly wrong per decision if the elements are combined with the wrong dependence structure. check the per-element error before touching the distribution

- two diseases, two cures (2/9/26, late): a stale view can be wrong in two directions, and the error accounting tells them apart
    - stale promises (false positives): the view says a block is there and it is not. a record-insert index that never sees evictions is all this. cure: stop trusting an entry as its idle age grows, a ttl at the turnover or the survival weight. repairs 80-90% of what cardinal scorers lost to an oversized shadow
    - the blind spot (false negatives and regret): the view does not know what happened since it was taken, and most of what happened is the router's own dispatches, the warm spots it created itself. a periodic scrape is mostly this: a calibrated discount lowers its false positives by a third and leaves regret exactly where it was. cure: overlay the dispatches since the last refresh, wiped at each refresh so it cannot outgrow the cache. rankers lose nothing at any age; cardinal scorers are at fresh-view quality to about half a turnover, after which stale promises take over and the discount is the complement
    - a discount cannot cure a blind spot because it only lowers trust in what the view says; an overlay cannot cure stale promises because it adds promises. dynamo's fixed expiry is the first cure with the wrong constant, llm-d's speculative entries are the second sold as a race fix
    - how a router knows which it has: the engine reports per request how many tokens were actually reused (vllm's usage stats), which is the execution false-positive rate. high → discount; low but hit rate below the fresh run → blind spot
    - the general shape: before treating an estimate, split its error into "said something false" and "did not know something true". the treatments for the two are opposites, and the wrong one makes things worse
    - the ranker clause (from both traces): an overlay is record-insert with a short memory, and record-insert plus a pure ranker locks onto one worker on shared-prefix traffic no matter how short the memory. the mechanism is a symmetry break at the first instant: with a live view the first same-timestamp batch ties (nothing admitted yet), spreads by load, and every worker learns the shared prefix; with record-insert the second request already sees it on worker 0, a one-block match that pure ranking treats as decisive, and no other worker ever gets a request to learn it from. cures: a match threshold (sglang routes by load below half the prompt), a load term (hybrid), a hash cap (dualmap), or a key that skips the shared prefix. so the blind-spot cure is for scorers; a pure ranker needs a threshold or a cap before it is given any memory of its own decisions
    - correction from the controlled sweep (4/9/26, out/lockin.*): the lock-in does not need record-insert at all. on spaced arrivals even the *perfect view* locks in identically (the first dispatch is admitted before the second arrival, so one worker is strictly warmest forever); the traces' perfect view escaped only because same-instant ties spread the first burst by load before anything was admitted. record-insert merely closes that one escape hatch. so the clause tightens: a pure ranker needs a threshold or a cap on shared-prefix traffic *whatever the view*, and the threshold is free when traffic is not shared (hit rate even rises with the universal prefix, a free hit wherever a request lands)

- generations in an lru radix cache (4/9/26): the generational-gc picture (young die fast, survivors live long) is real here but shows up in *counts*, not clocks: 97% of evictions are deep leaves, yet idle-age-at-death is the same distribution at every depth, and measured false positives conditioned on believed idle age are depth-flat
    - why: gc treats generations differently, so "collections survived" carries information. lru applies one rule, evict the longest-idle leaf, so idle time screens off depth: once you know a block's last-access age, its depth tells you nothing more about survival. the snapshot already carries per-block last-access ages, so the survival model was already generational without knowing it
    - the lesson for covariate-hunting: a covariate that predicts *who enters the risky state* (deep blocks are idle more often) is useless to a model that already conditions on *being in the risky state* (the known idle age). check whether the candidate is screened off by what you condition on before building the stratified estimator
    - corollary for "pin the system prompt in cache": unnecessary under lru. a block touched every turn never gets idle, which is the protection pinning would add. the failure mode is not eviction pressure but spreading: a policy that scatters a prefix over workers divides its touch rate per copy (hybrid's shorter turnover)
