import math

from .hashring import HashRing
from .session_hash import session_key


class Chwbl:
    """Consistent hashing with bounded loads (Mirrokni, Thorup & Zadimoghaddam,
    SODA'18), in the form KubeAI ships as its LLM load balancer.

    Session hash gives perfect affinity and no back pressure: a hot prefix owns
    one worker no matter how deep that worker's queue gets. CHWBL keeps the ring
    but caps what any worker may accept at (1 + eps) times the fleet's mean load.
    A request whose owner is over the cap walks the ring to the next worker, so
    affinity holds right up to the point where it would cost queueing, and no
    further.

    The policy never reads worker.view. Affinity comes from the key alone, so it
    cannot be fooled by a stale or optimistic cache view the way the additive
    scorers can - and that is exactly why it matters here. The scale-out
    experiment needs a shield that keeps a hot prefix off an overloaded veteran
    without needing to believe anything about a cold worker's shelf: the bound is
    computed from load, which is always observed exactly.
    """

    name = "chwbl"

    def __init__(
        self,
        key_blocks: int = 1,
        load_bound: float = 1.25,
        vnodes: int = 64,
    ):
        self.key_blocks = key_blocks
        self.load_bound = load_bound
        self.vnodes = vnodes
        self._ring = None
        self._worker_count = None

    def choose(self, req, workers):
        if (
            self._ring is None
            or self._worker_count != len(workers)
        ):
            self._worker_count = len(workers)
            self._ring = HashRing(
                self._worker_count,
                self.vnodes,
            )

        key = session_key(req, self.key_blocks)

        # the paper's bound counts the item being placed, so a fleet sitting at
        # mean m admits a worker up to ceil((1 + eps) * (m + 1)). the +1 is what
        # keeps the bound from being unsatisfiable on an idle fleet: at mean 0
        # every worker would otherwise be "over" its bound of 0
        mean_outstanding = sum(
            worker.outstanding
            for worker in workers
        ) / len(workers)
        bound = math.ceil(self.load_bound * (mean_outstanding + 1))

        # this is the rehash variant of the bounded-load walk, not the neighbour
        # walk. the paper walks clockwise to the next distinct owner on the ring;
        # hashring.py exposes only lookup(), which returns one owner and no way
        # to ask for the successor, so we get successive owners by rehashing the
        # key with an incrementing probe suffix instead. the two are equivalent
        # in what matters: a deterministic, key-dependent order over the workers
        # that different keys shuffle differently, so an overloaded worker's
        # overflow does not all pile onto one neighbour
        examined_ids = set()
        max_probes = 2 * len(workers)

        for probe in range(max_probes + 1):
            probe_key = key if probe == 0 else f"{key}:{probe}"
            worker_id = self._ring.lookup(probe_key)

            # a rehash can land on a worker we already rejected, so the cap is on
            # probes rather than on distinct owners - it is what guarantees the
            # walk terminates when the ring keeps returning the same few owners
            if worker_id in examined_ids:
                continue

            examined_ids.add(worker_id)
            candidate = workers[worker_id]

            if candidate.outstanding + 1 <= bound:
                return candidate

        # every worker the walk reached is over the bound. that is reachable
        # whenever loads are equal and high, and the paper's answer is that the
        # bound is advisory: something has to take the request, so give it to
        # whoever is carrying least
        return min(
            workers,
            key=lambda worker: (
                worker.outstanding,
                worker.id,
            ),
        )
