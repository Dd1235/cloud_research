from .hashring import HashRing

def session_key(req) -> str:

    if req.blocks:
        return repr(req.blocks[0])

    return f"req:{req.id}"


class SessionHash:

    name = "session_hash"

    def __init__(self, vnodes: int = 64):
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
                self.vnodes
            )
        worker_id = self._ring.lookup(session_key(req))
        return workers[worker_id]
