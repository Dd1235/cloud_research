import math

import numpy as np

from .request import Request


# frequency is inverse proportional to its rank

def zipf_probs(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    probabilities = ranks ** (-alpha)

    return probabilities / probabilities.sum()


# a generated request - arrival time is poisson process, prompt token count - shared prefix blocks
# + unique suffix blocks, output token count - random configured range, blocks exact cache-key path


def generate(
    rng: np.random.Generator,
    n_requests: int,
    rate: float,
    *,
    n_prefixes: int = 16,
    zipf_alpha: float = 1.0,
    prefix_blocks: int = 32,
    suffix_blocks: tuple[int, int] = (4, 16),
    output_tokens: tuple[int, int] = (16, 64),
    block_size: int = 16,
    universal_blocks: int = 0,
) -> list[Request]:
    """universal_blocks prepends the same blocks to every request, the way a
    shared system prompt begins every real conversation. The Mooncake traces
    have exactly this shape (conversation: one distinct first block in 6000
    requests) and it is what broke first-block session keys and let
    record-insert views lock a pure ranker onto one worker; the knob makes
    those pathologies reproducible and sweepable instead of trace anecdotes."""
    probabilities = zipf_probs(n_prefixes, zipf_alpha)

    universal_prefix = tuple(
        ("u", block_index)
        for block_index in range(universal_blocks)
    )

    time = 0.0
    requests = []

    for request_id in range(n_requests):
        arrival_gap = rng.exponential(1 / rate)
        time += arrival_gap

        prefix_id = int(
            rng.choice(n_prefixes, p=probabilities)
        )
        suffix_length = int(
            rng.integers(
                suffix_blocks[0],
                suffix_blocks[1] + 1,
            )
        )


        shared_prefix = tuple(
            ("p", prefix_id, block_index)
            for block_index in range(prefix_blocks)
        )
        unique_suffix = tuple(
            ("s", request_id, block_index)
            for block_index in range(suffix_length)
        )
        blocks = universal_prefix + shared_prefix + unique_suffix

        output_length = int(
            rng.integers(
                output_tokens[0],
                output_tokens[1] + 1,
            )
        )

        requests.append(
            Request(
                id=request_id,
                arrival=time,
                prompt_tokens=len(blocks) * block_size,
                output_tokens=output_length,
                blocks=blocks,
            )
        )

    return requests


def generate_sessions(
    rng: np.random.Generator,
    n_requests: int,
    session_rate: float,
    *,
    universal_blocks: int = 2,
    mean_turns: float = 8,
    think_time_p50: float = 20.0,
    think_time_sigma: float = 1.0,
    first_prompt_blocks: tuple[int, int] = (4, 16),
    reply_blocks: tuple[int, int] = (1, 4),
    output_tokens: tuple[int, int] = (16, 64),
    block_size: int = 16,
) -> list[Request]:
    """Multi-turn chat sessions: deep serial reuse within a session, shallow
    universal reuse across sessions.

    The reuse taxonomy this models has exactly two layers:

    * cross-session, shallow: every request from every session starts with the
      same ("u", i) blocks, the shared system prompt. That is all any two
      sessions ever have in common, and it is only universal_blocks long.
    * within-session, deep and serial: turn t re-sends the whole conversation
      so far (the trunk: every earlier turn's user text plus the assistant's
      replies) and appends fresh user blocks. So turn t's block path is a
      strict prefix of turn t+1's, prompts grow monotonically, and the reuse is
      causally ordered - the blocks a turn can hit were produced by that same
      session's earlier turns, never concurrently by someone else.

    Why bother when generate() already has a hot set: generate() is stationary.
    Its Zipf hot prefixes are hot at second 0 and equally hot at second 1000, so
    a view that has not been refreshed in a while is still roughly right about
    where the popular prefixes live, and stale-view policies come out looking
    better than they are. Sessions are nonstationary by construction: a session's
    trunk is worthless before its first turn, is the single most valuable thing
    in the cluster for a few minutes, then is dead forever once the user stops
    typing. Locality has to be tracked, not learned once.
    """
    universal_prefix = tuple(
        ("u", block_index)
        for block_index in range(universal_blocks)
    )

    session_start = 0.0
    session_id = 0
    requests = []

    # sessions arrive as their own Poisson process; each contributes a burst of
    # turns spread over minutes, so request arrivals are far from Poisson
    while len(requests) < n_requests:
        session_start += rng.exponential(1 / session_rate)

        n_turns = int(
            rng.geometric(p=1 / mean_turns)
        )

        # the trunk is the conversation history every later turn re-sends, and
        # the running index keeps every block this session ever mints distinct
        trunk = ()
        running_index = 0
        turn_arrival = session_start

        for turn_index in range(n_turns):
            # the opening prompt carries the task; follow-ups are short asides,
            # so they are drawn on the reply scale rather than the opening one
            fresh_range = first_prompt_blocks if turn_index == 0 else reply_blocks
            fresh_length = int(
                rng.integers(
                    fresh_range[0],
                    fresh_range[1] + 1,
                )
            )
            fresh_user_blocks = tuple(
                ("s", session_id, running_index + block_index)
                for block_index in range(fresh_length)
            )
            running_index += fresh_length

            blocks = universal_prefix + trunk + fresh_user_blocks

            output_length = int(
                rng.integers(
                    output_tokens[0],
                    output_tokens[1] + 1,
                )
            )

            requests.append(
                Request(
                    id=len(requests),
                    arrival=turn_arrival,
                    prompt_tokens=len(blocks) * block_size,
                    output_tokens=output_length,
                    blocks=blocks,
                ),
            )

            # the assistant's answer is prompt for every later turn, so it joins
            # the trunk. Its length is drawn rather than derived from
            # output_tokens: what matters downstream is that the trunk grows by
            # a plausible amount, not that the two agree token for token
            reply_length = int(
                rng.integers(
                    reply_blocks[0],
                    reply_blocks[1] + 1,
                )
            )
            reply_block_ids = tuple(
                ("s", session_id, running_index + block_index)
                for block_index in range(reply_length)
            )
            running_index += reply_length

            trunk = trunk + fresh_user_blocks + reply_block_ids

            # think time dwarfs service time at these scales, so a turn is not
            # made to wait for its own predecessor's response to finish; the
            # arrival order within a session is causal either way
            turn_arrival += rng.lognormal(
                mean=math.log(think_time_p50),
                sigma=think_time_sigma,
            )

        session_id += 1

    # merge the sessions into one arrival stream. Cutting the sorted stream at
    # n_requests biases the tail: sessions that would have started after the
    # last one spawned are missing, so the final few seconds are thinner in new
    # sessions than steady state. Harmless as long as n_requests is well above
    # a single session's turn count
    requests.sort(key=lambda request: request.arrival)
    requests = requests[:n_requests]

    for request_id, request in enumerate(requests):
        request.id = request_id

    return requests