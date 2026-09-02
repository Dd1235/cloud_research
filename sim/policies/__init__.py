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
POLICIES = {
    RoundRobin.name: lambda rng: RoundRobin(),
    LeastOutstanding.name: lambda rng: LeastOutstanding(),
    PowerOfTwo.name: lambda rng: PowerOfTwo(rng),
    SessionHash.name: lambda rng: SessionHash(),
    LongestPrefix.name: lambda rng: LongestPrefix(),
    Hybrid.name: lambda rng: Hybrid(alpha=1.0, beta=1.0),
    DualMap.name: lambda rng: DualMap(),
}


def make_policy(name: str, rng):
    if name not in POLICIES:
        known_names = ", ".join(POLICIES)
        raise KeyError(
            f"unknown policy {name!r}; known policies: {known_names}"
        )

    return POLICIES[name](rng)


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
