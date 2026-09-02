from .hashring import HashRing

# how many leading blocks identify a session. one block is right for the
# synthetic workload, where block 0 names the shared prefix. real traces put a
# system prompt everyone shares in the first blocks (every conversation-trace
# request starts with the same block), so a one-block key maps the whole trace
# to one worker. dualmap's paper grows the hashed prefix when a key gets hot;
# a fixed deeper key is the simple version of that
def session_key(req, key_blocks: int = 1) -> str:

    if req.blocks:
        return repr(req.blocks[:key_blocks])

    return f"req:{req.id}"


class SessionHash:

    name = "session_hash"

    def __init__(self, vnodes: int = 64, key_blocks: int = 1):
        self.vnodes = vnodes
        self.key_blocks = key_blocks
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
        worker_id = self._ring.lookup(session_key(req, self.key_blocks))
        return workers[worker_id]
