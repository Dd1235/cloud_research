import heapq
import itertools
from collections import deque
import numpy as np


# single-threaded discrete-event simulation
# lambda - rate at which requests arrive
# mu - rate at which server processes them
# model as poisson
# pho - utilization
# fifo, processes based on smaller time, smaller seq no (to break ties), "arrived" vs "depart", 

def simulate(lam : float, mu : float, n_arrivals: int = 20_000, seed: int = 42, warmup_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    heap = []
    seq = itertools.count()

    def push(time, kind, payload=None):
        heapq.heappush(heap, (time, next(seq), kind, payload))

    now = 0.0
    busy = False
    queue = deque()
    sojourn = []
    arrived = 0

    push(rng.exponential(1 / lam), "arrive")

    while heap:
        now, _, kind, payload = heapq.heappop(heap)

        if kind == "arrive":
            arrived += 1

            if arrived < n_arrivals:
                next_gap = rng.exponential(1 / lam)
                push(now + next_gap, "arrive")

            if busy:
                queue.append(now)
            else:
                busy = True
                service_time = rng.exponential(1 / mu)
                push(now + service_time, "depart", now)

        else:
            sojourn.append(now - payload)
            if queue:
                next_arrival = queue.popleft()
                service_time = rng.exponential(1/mu)
                push(now + service_time, "depart", next_arrival)

            else:
                busy = False

    warmup_count = int(len(sojourn) * warmup_frac)
    measured = np.asarray(sojourn[warmup_count:])

    return measured.mean(), np.percentile(measured, 99)


if __name__ == "__main__":

    mu = 1.0

    print(f"{'rho':>5} {'sim mean':>9} {'analytic':>9} {'sim p99':>8}")

    for rho in (0.5,0.8,0.9,0.95):
        lam = rho * mu
        mean, p99 = simulate(lam, mu)

        analytic_mean = 1 / (mu - lam)
        print(f"{rho:>5.2f} {mean:>9.2f} {analytic_mean:>9.2f} {p99:>8.2f}")

