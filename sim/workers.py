from collections import deque
from .request import Request


class Worker:

    def __init__(self, engine, wid: int, c_prefill: float, c_decode: float):
        self.engine = engine
        self.id = wid
        self.c_prefill = c_prefill
        self.c_decode = c_decode

        self.queue = deque()
        self.busy = False
        self.completed = 0


    def _first_token(self, req: Request) -> None:
        req.first_token = self.engine.now

    def _finish(self, req: Request) -> None:
        req.finish = self.engine.now
        self.completed += 1
        self._start_next()

    @property
    def outstanding(self) -> int:
        return len(self.queue) + (1 if self.busy else 0)

    def submit(self, req: Request) -> None:
        req.worker_id = self.id
        self.queue.append(req)

        if not self.busy:
            self._start_next()

    def _start_next(self) -> None:

        if not self.queue:
            self.busy = False
            return

        self.busy = True
        req = self.queue.popleft()
        req.start = self.engine.now

        prefill_time = self.c_prefill * req.prompt_tokens
        decode_time = self.c_decode * req.output_tokens


        self.engine.schedule(prefill_time, self._first_token, req)
        self.engine.schedule(prefill_time + decode_time, self._finish, req)