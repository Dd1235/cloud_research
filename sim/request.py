from dataclasses import dataclass


@dataclass
class Request:
    id: int
    arrival: float
    prompt_tokens: int
    output_tokens: int

    worker_id: int | None = None
    start: float | None = None # prefill starts
    first_token: float | None = None
    finish: float | None = None

    @property
    def done(self) -> bool:
        return self.finish is not None

    @property
    def queue_wait(self) -> float:
        return self.start - self.arrival

    # queue wait + prefill time, so first_token_time - arrival_time
    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival 

    @property
    def tpot(self) -> float:
        return (self.finish - self.first_token) / max(self.output_tokens - 1, 1,)

    