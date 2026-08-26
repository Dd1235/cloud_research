- M/M/1 - standard shorthand from Kendall's queue notation
    - Markovian service times, and Markovian arrival, one server
    - Markovian means memoryless
    - poisson, independent arrivals at rate lambda, independent exponential service times at rate mu, one server, fifo queue, stable system (lam < mu)
    - effectively steady-state operation

- For an ideal M/M/1 queue
    - mean latency = 1 / (mu - lam)
    - p99 = ln(100) / (mu - lam)

- ttft - time to first token
- tpot - time per output token = (finish - first_token) / (output_tokens - 1)
- prefill - per prompt token
- decode - per output token