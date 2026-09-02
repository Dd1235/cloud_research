from dataclasses import dataclass, field


@dataclass
class Request:
    id: int
    arrival: float
    prompt_tokens: int
    output_tokens: int
    blocks: tuple = ()
    cached_tokens: int = 0

    worker_id: int | None = None
    start: float | None = None # prefill starts
    first_token: float | None = None
    finish: float | None = None

    # when each output token was emitted. only the batching worker fills this in,
    # because only there can a token be delayed by work done for other requests
    token_times: list = field(default_factory=list)

    # filled by the router at dispatch, in tokens like cached_tokens. None when
    # a request reached a worker without going through a router
    estimated_cached_tokens: int | None = None        # what the view promised on the chosen worker
    true_cached_tokens_at_dispatch: int | None = None # what that worker really held at that instant
    best_cached_tokens_at_dispatch: int | None = None # the most any worker really held at that instant

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

    # tpot is a mean, so a single long stall averages away. these are the actual
    # gaps between consecutive tokens, which is what a reader would notice
    @property
    def tbt_gaps(self) -> list[float]:
        return [
            later - earlier
            for earlier, later in zip(self.token_times, self.token_times[1:])
        ]

    