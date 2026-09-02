import math


def turnover_from_evictions(cache_blocks: int, evictions: int, duration: float) -> float:
    """Seconds to replace one cache's worth of blocks: capacity / eviction rate.

    Measured from a worker's eviction counter over a run. In steady state this
    is the LRU characteristic time: a block that is not touched again lives
    about this long before it is evicted.
    """
    if evictions == 0 or duration <= 0:
        return float("inf")

    return cache_blocks / (evictions / duration)


def che_characteristic_time(rates, capacity: int, tolerance: float = 1e-9) -> float:
    """Che's fixed point: the T with sum over blocks of 1 - exp(-rate * T) = capacity.

    Each block with request rate `rate` is present in an LRU cache with
    probability 1 - exp(-rate * T_C), and the presences must add up to the
    capacity. Solved by bisection. A router can compute this from its own
    dispatch stream alone, without any worker telemetry. If every observed
    block fits, nothing is ever evicted and the characteristic time is infinite.
    """
    rates = [rate for rate in rates if rate > 0]

    if len(rates) <= capacity:
        return float("inf")

    def expected_resident(time_constant: float) -> float:
        return sum(1.0 - math.exp(-rate * time_constant) for rate in rates)

    low, high = 0.0, 1.0
    while expected_resident(high) < capacity:
        high *= 2.0

    while high - low > tolerance * max(high, 1.0):
        middle = (low + high) / 2.0
        if expected_resident(middle) < capacity:
            low = middle
        else:
            high = middle

    return (low + high) / 2.0
