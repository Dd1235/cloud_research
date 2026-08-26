from collections import deque


class Worker:

    def __init__(self, engine, wid: int, service_rate: float):
        self.engine = engine
        self.id = wid
        self.service_rate = service_rate

        self.queue = deque()
        self.busy = False
        self.completed = 0

        self.sojourn = []   

    def _finish(self, arrival_time: float) -> None:
        sojourn_time = self.engine.now - arrival_time
        self.sojourn.append(sojourn_time)

        self.completed += 1
        self._start_next()   

    @property
    def outstanding(self) -> int:
        return len(self.queue) + (1 if self.busy else 0)

    def submit(self, arrival_time: float) -> None:
        self.queue.append(arrival_time)

        if not self.busy:
            self._start_next()

    def _start_next(self) -> None:
        if not self.queue:
            self.busy = False
            return

        self.busy = True
        arrival_time = self.queue.popleft()

        service_time = self.engine.rng.exponential(1 / self.service_rate)
        self.engine.schedule(service_time, self._finish, arrival_time)