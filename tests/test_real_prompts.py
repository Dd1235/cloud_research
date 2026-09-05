import subprocess
import sys

from real.prompts import VOCAB_RANGE, chunk_tokens, engine_blocks, prompt_token_ids


def test_a_block_id_always_yields_the_same_chunk_even_in_another_process():
    here = chunk_tokens(("m", 7))
    assert len(here) == 512

    # a fresh interpreter has a different hash() salt; the chunk must not care
    elsewhere = subprocess.run(
        [sys.executable, "-c", "from real.prompts import chunk_tokens; print(chunk_tokens(('m', 7))[:8])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert elsewhere == str(here[:8])


def test_different_block_ids_yield_different_chunks_inside_the_vocab_range():
    first = chunk_tokens(("m", 1))
    second = chunk_tokens(("m", 2))

    assert first != second
    assert all(VOCAB_RANGE[0] <= token < VOCAB_RANGE[1] for token in first + second)


def test_shared_leading_ids_give_a_shared_token_prefix_of_whole_chunks():
    left = prompt_token_ids((("m", 1), ("m", 2), ("m", 3)), 1536)
    right = prompt_token_ids((("m", 1), ("m", 2), ("m", 9)), 1536)

    # the first two chunks are identical, the third is not
    assert left[:1024] == right[:1024]
    assert left[1024:] != right[1024:]


def test_the_prompt_is_cut_to_the_requests_length_and_blocked_in_whole_engine_blocks():
    # 14 trace blocks hold 7168 tokens; the request is 6758 long, so it ends
    # 102 tokens into its last chunk. 6758 = 422 * 16 + 6: 422 engine blocks
    # and 6 tokens the engine will never cache
    token_ids = prompt_token_ids(tuple(("m", i) for i in range(14)), 6758)
    assert len(token_ids) == 6758

    blocks = engine_blocks(token_ids, 16)
    assert len(blocks) == 422
    assert blocks[0] == tuple(token_ids[:16])
    assert blocks[-1] == tuple(token_ids[6736:6752])


def test_a_request_claiming_more_tokens_than_its_chunks_hold_is_as_long_as_its_chunks():
    assert len(prompt_token_ids((("m", 1), ("m", 2)), 2000)) == 1024
    assert prompt_token_ids((), 300) == []
