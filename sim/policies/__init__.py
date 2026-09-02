from .dualmap import DualMap
from .hybrid import Hybrid
from .least_outstanding import LeastOutstanding
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
    LongestPrefix.name: lambda rng, options: LongestPrefix(),
    Hybrid.name: lambda rng, options: Hybrid(alpha=1.0, beta=1.0),
    DualMap.name: lambda rng, options: DualMap(
        key_blocks=options.get("key_blocks", 1),
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
    "Hybrid",
    "LeastOutstanding",
    "LongestPrefix",
    "PowerOfTwo",
    "RoundRobin",
    "SessionHash",
]
