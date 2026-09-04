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