from collections import deque
import numpy as np
from sim.engine import Engine
from .policies.round_robin import RoundRobin
from .workers import Worker
from .request import Request

# single-threaded discrete-event simulation
# lambda - rate at which requests arrive
# mu - rate at which server processes them
# model as poisson
# pho - utilization
# fifo, processes based on smaller time, smaller seq no (to break ties), "arrived" vs "depart", 

def simulate(lam : float, mu : float, n_arrivals: int = 20_000, seed: int = 42, warmup_frac: float = 0.1):
    engine = Engine(seed)
    busy = False
    queue = deque()
    sojourn = []
    arrived = 0

    def arrive() -> None:
        nonlocal busy, arrived

        arrived += 1

        if arrived < n_arrivals:
            arrival_gap = engine.rng.exponential(1 / lam)
            engine.schedule(arrival_gap, arrive)

        if busy:
            queue.append(engine.now)
        else:
            busy = True
            service_time = engine.rng.exponential(1 / mu)
            engine.schedule(service_time, depart, engine.now)


    def depart(arrival_time: float) -> None:

        nonlocal busy
        sojourn.append(engine.now - arrival_time)

        if queue:
            next_arrival = queue.popleft()
            service_time = engine.rng.exponential(1 / mu)
            engine.schedule(service_time, depart, next_arrival)
        else:
            busy = False
        

    first_arrival_gap = engine.rng.exponential(1 / lam)
    engine.schedule(first_arrival_gap, arrive)

    engine.run()

    warmup_count = int(len(sojourn) * warmup_frac)
    measured = np.asarray(sojourn[warmup_count:])

    return measured.mean(), np.percentile(measured, 99)


def simulate_with_workers(
    rate: float,
    c_prefill: float,
    c_decode: float,
    prompt_tokens: int,
    output_tokens: int,
    n_workers: int = 2,
    n_arrivals: int = 20_000,
    seed: int = 0,
    warmup_frac: float = 0.1,
):
    assert n_workers >= 1

    engine = Engine(seed)
    workers = [
        Worker(
            engine,
            wid=worker_id,
            c_prefill=c_prefill,
            c_decode=c_decode,
        )
        for worker_id in range(n_workers)
    ]
    policy = RoundRobin()
    requests = []
    arrived = 0

    def arrive() -> None:
        nonlocal arrived

        req = Request(
            id=arrived,
            arrival=engine.now,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        requests.append(req)
        arrived += 1

        if arrived < n_arrivals:
            arrival_gap = engine.rng.exponential(1 / rate)
            engine.schedule(arrival_gap, arrive)

        chosen_worker = policy.choose(req, workers)
        chosen_worker.submit(req)

    first_arrival_gap = engine.rng.exponential(1 / rate)
    engine.schedule(first_arrival_gap, arrive)

    engine.run()

    warmup_count = int(len(requests) * warmup_frac)
    measured = np.asarray([
        req.ttft
        for req in requests[warmup_count:]
    ])

    return measured.mean(), np.percentile(measured, 99)


if __name__ == "__main__":

    mu = 1.0

    print(f"{'rho':>5} {'sim mean':>9} {'analytic':>9} {'sim p99':>8}")

    for rho in (0.5,0.8,0.9,0.95):
        lam = rho * mu
        mean, p99 = simulate(lam, mu)

        analytic_mean = 1 / (mu - lam)
        print(f"{rho:>5.2f} {mean:>9.2f} {analytic_mean:>9.2f} {p99:>8.2f}")

