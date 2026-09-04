from .dualmap import DualMap
from .dynamo_cost import DynamoCost
from .hybrid import Hybrid
from .least_outstanding import LeastOutstanding
from .lmetric import Lmetric
from .longest_prefix import LongestPrefix
from .power_of_two import PowerOfTwo
from .round_robin import RoundRobin
from .session_hash import SessionHash


# Every factory takes the policy rng, even the policies that do not need one, so
# a caller can build any policy by name without knowing its constructor. Adding a
# policy means adding one line here.
#
# Order is deliberate: cache-blind baselines first, then affinity, then the
# policies that use both signals. Scripts print them in this order.
# a factory also takes an options dict; policies that have no knobs ignore it,
# so one --session-depth flag can reach the two hashing policies and nothing else
POLICIES = {
    RoundRobin.name: lambda rng, options: RoundRobin(),
    LeastOutstanding.name: lambda rng, options: LeastOutstanding(),
    PowerOfTwo.name: lambda rng, options: PowerOfTwo(rng),
    SessionHash.name: lambda rng, options: SessionHash(
        key_blocks=options.get("key_blocks", 1),
    ),
    LongestPrefix.name: lambda rng, options: LongestPrefix(
        match_threshold=options.get("match_threshold", 0.0),
    ),
    Hybrid.name: lambda rng, options: Hybrid(
        alpha=1.0,
        beta=1.0,
        overlap_source=options.get("overlap_source", "raw"),
    ),
    DualMap.name: lambda rng, options: DualMap(
        key_blocks=options.get("key_blocks", 1),
    ),
    Lmetric.name: lambda rng, options: Lmetric(
        overlap_source=options.get("overlap_source", "raw"),
    ),
    DynamoCost.name: lambda rng, options: DynamoCost(
        overlap_source=options.get("overlap_source", "raw"),
    ),
}


def make_policy(name: str, rng, options: dict | None = None):
    if name not in POLICIES:
        known_names = ", ".join(POLICIES)
        raise KeyError(
            f"unknown policy {name!r}; known policies: {known_names}"
        )

    return POLICIES[name](rng, options or {})


__all__ = [
    "POLICIES",
    "make_policy",
    "DualMap",
    "DynamoCost",
    "Hybrid",
    "LeastOutstanding",
    "Lmetric",
    "LongestPrefix",
    "PowerOfTwo",
    "RoundRobin",
    "SessionHash",
]
