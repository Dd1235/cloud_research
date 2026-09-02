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
