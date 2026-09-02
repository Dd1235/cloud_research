from collections import deque

from .radix import PrefixCache
from .request import Request


# which output token a resident sequence produced during one iteration
FIRST_TOKEN = "first_token"
DECODE_STEP = "decode_step"
NO_TOKEN = "no_token"  # still working through its prompt, nothing emitted yet


class _Sequence:
    """A request while it is resident on the worker.

    The worker holds several of these at once and advances all of them by one
    step per iteration, which is what continuous batching means.
    """

    __slots__ = ("req", "prefill_left", "tokens_left")

    def __init__(self, req: Request, prefill_left: int):
        self.req = req
        # uncached prompt tokens still to be prefilled
        self.prefill_left = prefill_left
        # output tokens still to be emitted, the first one included
        self.tokens_left = req.output_tokens


class BatchedWorker:
    """A worker that runs iterations instead of whole requests.

    The sequential Worker serves one request from start to finish before looking
    at the next. A real engine does not: every forward pass advances *all*
    resident sequences by one step, and new arrivals join at the next iteration
    boundary rather than waiting for the current request to finish. That is
    iteration-level scheduling from Orca, and it is why throughput scales with
    batch size.

    Cost of one iteration:

        t_iter = c_iter + c_prefill * prefill_tokens + c_decode * decode_steps

    c_iter is the fixed cost of a forward pass, dominated by reading every weight
    from memory once. It is paid whether one sequence or sixteen are resident,
    so batching amortises it, and that amortisation is the entire throughput win.

    To stay comparable with the sequential Worker, note that a single stream
    costs c_iter + c_decode per output token here, so pick the two constants such
    that c_iter + c_decode equals the sequential worker's c_decode.

    The public surface is deliberately identical to Worker (submit, outstanding,
    busy, cache, and the same counters), so policies and metrics do not change.
    """

    def __init__(
        self,
        engine,
        wid: int,
        c_prefill: float,
        c_decode: float,
        c_iter: float = 0.008,
        cache_blocks: int = 1,
        block_size: int = 16,
        max_batch: int = 16,
        prefill_budget: int | None = None,
    ):
        self.engine = engine
        self.id = wid
        self.c_prefill = c_prefill
        self.c_decode = c_decode
        self.c_iter = c_iter

        self.block_size = block_size
        self.max_batch = max_batch
        # most prompt tokens one iteration may process. None means a prompt is
        # prefilled in a single iteration, however long it is
        self.prefill_budget = prefill_budget
        self.cache = PrefixCache(cache_blocks)

        # what routing policies read. the worker's own cache by default, i.e. a
        # perfect and instantaneous view; the router swaps in a stale or
        # approximate one. admission below always consults self.cache
        self.view = self.cache

        self.queue = deque()
        self.running: list[_Sequence] = []

        self._iteration_in_flight = False
        self._step_scheduled = False

        # same counter names as the sequential worker so metrics.summarize works
        self.busy_time = 0.0
        self.completed = 0
        self.tokens_processed = 0
        self.tokens_reused = 0
        self.iterations = 0

    @property
    def outstanding(self) -> int:
        return len(self.queue) + len(self.running)

    @property
    def busy(self) -> bool:
        return self._iteration_in_flight

    def submit(self, req: Request) -> None:
        req.worker_id = self.id
        self.queue.append(req)
        self._kick()

    def _kick(self) -> None:
        """Ask for an iteration to start, without starting one right now.

        Scheduling the step at delay 0 rather than calling _step directly is the
        important part. Every other event already queued for this same timestamp
        runs first, so a burst of requests that all arrive at the same instant is
        admitted into the same iteration. Calling _step directly from submit
        would open an iteration containing only the first of them and make the
        rest wait a full iteration for no reason.
        """
        if self._iteration_in_flight or self._step_scheduled:
            return

        self._step_scheduled = True
        self.engine.schedule(0.0, self._step)

    def _admit_from_queue(self) -> None:
        """Move waiting requests into the running batch, up to max_batch.

        The prefix cache is consulted here, at admission, because that is when
        the engine would look up which blocks it already holds.
        """
        while self.queue and len(self.running) < self.max_batch:
            req = self.queue.popleft()
            req.start = self.engine.now

            cached_blocks = self.cache.match(req.blocks)
            req.cached_tokens = min(
                cached_blocks * self.block_size,
                req.prompt_tokens,
            )
            uncached_tokens = req.prompt_tokens - req.cached_tokens

            self.tokens_processed += req.prompt_tokens
            self.tokens_reused += req.cached_tokens
            self.cache.insert(req.blocks, self.engine.now)

            self.running.append(_Sequence(req, uncached_tokens))

    def _step(self) -> None:
        self._step_scheduled = False
        self._admit_from_queue()

        if not self.running:
            self._iteration_in_flight = False
            return

        self._iteration_in_flight = True

        prefill_tokens = 0
        decode_steps = 0
        actions = []

        budget_left = (
            self.prefill_budget
            if self.prefill_budget is not None
            else float("inf")
        )

        for sequence in self.running:
            if sequence.prefill_left > 0:
                # take as much of the remaining prompt as the budget allows. with
                # no budget this is the whole prompt and the iteration becomes as
                # long as the prompt, which is what stalls everyone else
                chunk = min(sequence.prefill_left, budget_left)
                sequence.prefill_left -= chunk
                prefill_tokens += chunk
                budget_left -= chunk

                if sequence.prefill_left > 0:
                    # prompt not finished, so no token comes out this iteration
                    actions.append((sequence, NO_TOKEN))
                    continue
            else:
                decode_steps += 1

            # a sequence emits its first token in the iteration that finishes its
            # prefill. a fully cached prompt has no prefill work left at all, but
            # still needs this one forward pass to produce that first token
            emits_first_token = sequence.req.first_token is None

            actions.append(
                (
                    sequence,
                    FIRST_TOKEN if emits_first_token else DECODE_STEP,
                )
            )

        iteration_time = (
            self.c_iter
            + self.c_prefill * prefill_tokens
            + self.c_decode * decode_steps
        )

        self.busy_time += iteration_time
        self.iterations += 1

        self.engine.schedule(
            iteration_time,
            self._end_iteration,
            actions,
        )

    def _end_iteration(self, actions) -> None:
        now = self.engine.now

        for sequence, action in actions:
            if action is NO_TOKEN:
                continue

            if action is FIRST_TOKEN:
                sequence.req.first_token = now

            sequence.req.token_times.append(now)
            sequence.tokens_left -= 1

            if sequence.tokens_left <= 0:
                sequence.req.finish = now
                self.completed += 1

        self.running = [
            sequence
            for sequence in self.running
            if sequence.req.finish is None
        ]

        self._iteration_in_flight = False
        self._kick()
