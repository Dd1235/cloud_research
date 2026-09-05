"""Turning a trace's block ids into token ids a real engine can prefill.

A mooncake row carries hash ids, one per 512-token block, and no text. To
replay it on a real engine every id becomes a fixed pseudo-random chunk of
token ids, seeded by the id, so two requests that share a leading run of ids
share a leading run of tokens, which is exactly what the engine's block
hashing turns back into a shared prefix. The tokens are nonsense to the model
and that is fine: the experiment is about the cache, not the answers.

The seed comes from blake2b of the id's repr, never from python's hash(),
which is salted per process: the replayer and the offline twin must agree on
every token without sharing a process.
"""
import hashlib
from functools import lru_cache

import numpy as np

# tokens per trace block (sim.traces.MOONCAKE_BLOCK_SIZE) and per engine
# block (vllm's default); a trace block must be a whole number of engine
# blocks so shared trace ids become shared engine blocks with nothing left over
CHUNK_TOKENS = 512
ENGINE_BLOCK_TOKENS = 16
assert CHUNK_TOKENS % ENGINE_BLOCK_TOKENS == 0

# clear of the special tokens at the bottom of every vocabulary and inside the
# smallest vocabulary we might load (llama 3.2 has 128k ids, qwen 151k)
VOCAB_RANGE = (1000, 32000)


@lru_cache(maxsize=65536)
def chunk_tokens(block, *, chunk: int = CHUNK_TOKENS, vocab_range=VOCAB_RANGE) -> tuple[int, ...]:
    """The fixed token chunk that stands for one trace block id."""
    digest = hashlib.blake2b(repr(block).encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    low, high = vocab_range

    return tuple(int(token) for token in rng.integers(low, high, size=chunk))


def prompt_token_ids(blocks, prompt_tokens: int, *, chunk: int = CHUNK_TOKENS) -> list[int]:
    """The prompt for a request: its blocks' chunks in order, cut to its length.

    A request whose length is not a multiple of the chunk ends part way through
    its last chunk, as the trace's last hash id stands for a partial block. A
    request with fewer chunks than its length claims is as long as its chunks.
    """
    token_ids = []

    for block in blocks:
        if len(token_ids) >= prompt_tokens:
            break

        token_ids.extend(chunk_tokens(block, chunk=chunk))

    return token_ids[:prompt_tokens]


def engine_blocks(token_ids, block_size: int = ENGINE_BLOCK_TOKENS) -> tuple[tuple[int, ...], ...]:
    """The prompt as the engine's cache sees it: whole blocks only.

    vllm hashes and caches full blocks and never a partial tail, so the
    router's own block ids stop where the engine's do.
    """
    full = len(token_ids) - len(token_ids) % block_size

    return tuple(
        tuple(token_ids[start:start + block_size])
        for start in range(0, full, block_size)
    )
