from .hashring import HashRing
from .session_hash import session_key


class DualMap:
    """Power-of-two-choices over two independent hash rings.

    Session hash sends every request for a prefix to exactly one worker, so a hot
    prefix creates a hotspot. DualMap hashes the same key on two rings with
    different salts, which gives two candidate workers, and then picks between
    them using current state. A hot prefix is therefore spread over two workers
    instead of one, and both of them stay warm for it.

    The hashing finds the candidates without consulting any cache view; the cache
    view is only used to rank the two. Follows Yuan et al., arXiv 2602.06502.
    """

    name = "dualmap"

    def __init__(self, vnodes: int = 64, key_blocks: int = 1):
        self.vnodes = vnodes
        self.key_blocks = key_blocks
        self._rings = None
        self._worker_count = None

    def _build_rings(self, worker_count: int):
        # different salts give two independent placements of the same key
        return (
            HashRing(worker_count, self.vnodes, salt="ring-a"),
            HashRing(worker_count, self.vnodes, salt="ring-b"),
        )

    def choose(self, req, workers):
        if (
            self._rings is None
            or self._worker_count != len(workers)
        ):
            self._worker_count = len(workers)
            self._rings = self._build_rings(self._worker_count)

        key = session_key(req, self.key_blocks)

        candidate_ids = {
            ring.lookup(key)
            for ring in self._rings
        }

        # the two rings can land on the same worker, in which case dualmap
        # degrades to plain session hashing for this key
        candidates = [
            workers[worker_id]
            for worker_id in sorted(candidate_ids)
        ]

        return max(
            candidates,
            key=lambda worker: (
                worker.view.match(req.blocks),
                -worker.outstanding,
                -worker.id,
            ),
        )
