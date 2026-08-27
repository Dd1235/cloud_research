import bisect
import hashlib


def stable_hash(value: str) -> int:
    digest = hashlib.blake2b(
        value.encode(),
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, "big")


class HashRing:
    def __init__(
        self,
        n_workers: int,
        vnodes: int = 64,
        salt: str = "",
    ):
        points = sorted(
            (
                stable_hash(f"{salt}:{worker_id}:{virtual_id}"),
                worker_id,
            )
            for worker_id in range(n_workers)
            for virtual_id in range(vnodes)
        )

        self._keys = [point for point, _ in points]
        self._owners = [owner for _, owner in points]

    def lookup(self, key: str) -> int:
        position = bisect.bisect(
            self._keys,
            stable_hash(key),
        ) % len(self._keys)

        return self._owners[position]