import numpy as np
import pytest

from sim.workload import generate, generate_sessions


def session_ids_of(request) -> set:
    """The sessions a request's blocks belong to, recovered from the ids alone."""
    return {
        block[1]
        for block in request.blocks
        if block[0] == "s"
    }


def group_by_session(requests) -> dict:
    turns_by_session = {}

    for request in requests:
        (session_id,) = session_ids_of(request)
        turns_by_session.setdefault(session_id, []).append(request)

    return turns_by_session


def test_workload_is_seeded_and_has_shared_prefixes():
    kwargs = dict(
        n_requests=3,
        rate=1.0,
        n_prefixes=1,
        prefix_blocks=2,
        suffix_blocks=(1, 1),
        output_tokens=(5, 5),
        block_size=16,
    )

    first = generate(np.random.default_rng(7), **kwargs)
    second = generate(np.random.default_rng(7), **kwargs)

    assert first == second
    assert [req.arrival for req in first] == sorted(
        req.arrival for req in first
    )

    for req in first:
        assert req.blocks[:2] == (
            ("p", 0, 0),
            ("p", 0, 1),
        )
        assert req.prompt_tokens == 48
        assert req.output_tokens == 5

def test_universal_blocks_begin_every_request_like_a_shared_system_prompt():
    requests = generate(
        np.random.default_rng(0),
        20,
        rate=1.0,
        universal_blocks=3,
        prefix_blocks=2,
    )

    universal_prefix = (("u", 0), ("u", 1), ("u", 2))
    for req in requests:
        assert req.blocks[:3] == universal_prefix
        # the shared prefix follows the universal one, so every request has
        # exactly one distinct first block, the trace pathology on demand
        assert req.blocks[3][0] == "p"
        assert req.prompt_tokens == len(req.blocks) * 16


def test_each_session_turn_strictly_extends_the_previous_turns_block_path():
    requests = generate_sessions(
        np.random.default_rng(3),
        200,
        session_rate=0.5,
        universal_blocks=2,
        first_prompt_blocks=(2, 2),
        reply_blocks=(1, 1),
    )

    turns_by_session = group_by_session(requests)
    assert any(len(turns) >= 2 for turns in turns_by_session.values())

    universal_prefix = (("u", 0), ("u", 1))
    for turns in turns_by_session.values():
        for turn_index, turn in enumerate(turns):
            assert turn.blocks[:2] == universal_prefix
            # every turn re-sends the trunk (2 prompt + 1 reply blocks per
            # earlier turn) plus its own 2 or 1 fresh blocks, so with these
            # degenerate ranges turn k holds universal + 2k blocks exactly
            assert len(turn.blocks) == 2 + 2 * (turn_index + 1)

        for earlier, later in zip(turns, turns[1:]):
            assert later.arrival > earlier.arrival
            assert later.blocks[:len(earlier.blocks)] == earlier.blocks
            assert len(later.blocks) > len(earlier.blocks)


def test_session_requests_are_one_sorted_stream_with_consistent_ids_and_tokens():
    requests = generate_sessions(
        np.random.default_rng(11),
        150,
        session_rate=0.5,
        block_size=8,
    )

    assert len(requests) == 150
    assert [req.id for req in requests] == list(range(150))
    assert [req.arrival for req in requests] == sorted(
        req.arrival for req in requests
    )

    for req in requests:
        assert req.prompt_tokens == len(req.blocks) * 8
        assert 16 <= req.output_tokens <= 64

    # turns of one session are spaced by the think time alone, which is the
    # log-normal median when sigma is zero
    spaced = generate_sessions(
        np.random.default_rng(11),
        150,
        session_rate=0.5,
        think_time_p50=20.0,
        think_time_sigma=0.0,
    )
    for turns in group_by_session(spaced).values():
        for earlier, later in zip(turns, turns[1:]):
            assert later.arrival - earlier.arrival == pytest.approx(20.0)


def test_two_session_streams_from_the_same_seed_are_identical():
    kwargs = dict(
        n_requests=80,
        session_rate=0.75,
        mean_turns=4,
    )

    first = generate_sessions(np.random.default_rng(19), **kwargs)
    second = generate_sessions(np.random.default_rng(19), **kwargs)

    assert first == second


def test_sessions_share_the_universal_prefix_and_nothing_else():
    requests = generate_sessions(
        np.random.default_rng(5),
        200,
        session_rate=0.5,
        universal_blocks=2,
    )

    turns_by_session = group_by_session(requests)
    assert len(turns_by_session) >= 2

    owner_of_block = {}
    for session_id, turns in turns_by_session.items():
        for turn in turns:
            for block in turn.blocks:
                if block[0] == "s":
                    assert owner_of_block.setdefault(block, session_id) == session_id

    # the only cross-session reuse a router can ever exploit is the system
    # prompt, so any two sessions overlap in exactly the universal blocks
    universal_prefix = {("u", 0), ("u", 1)}
    block_sets = [
        {block for turn in turns for block in turn.blocks}
        for turns in turns_by_session.values()
    ]
    for left in range(len(block_sets)):
        for right in range(left + 1, len(block_sets)):
            assert block_sets[left] & block_sets[right] == universal_prefix
