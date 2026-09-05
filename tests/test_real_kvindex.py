import pytest

from real.kvindex import EventFedIndex

# two engine blocks of 16 tokens, hashed 11 and 22 by the engine, 22 under 11
FIRST = tuple(range(16))
SECOND = tuple(range(16, 32))
OTHER = tuple(range(100, 116))

STORED = {
    "type": "BlockStored",
    "block_hashes": [11, 22],
    "parent_block_hash": None,
    "token_ids": list(range(32)),
    "block_size": 16,
}


def test_a_stored_chain_matches_by_token_content_from_the_root():
    index = EventFedIndex(capacity_blocks=8)
    index.apply(STORED, now=1.0)

    assert index.size == 2
    assert index.match((FIRST, SECOND)) == 2
    # the second block's parent is 11, so it is not a root block
    assert index.match((SECOND,)) == 0
    # a different second block stops the walk after the first
    assert index.match((FIRST, OTHER)) == 1


def test_a_removed_block_is_gone_and_logged_with_its_residence():
    index = EventFedIndex(capacity_blocks=8, record_residence=True)
    index.apply(STORED, now=1.0)
    index.apply({"type": "BlockRemoved", "block_hashes": [22]}, now=5.0)

    assert index.match((FIRST, SECOND)) == 1
    assert index.evictions == 1
    # (hash, stored, last access, removed, depth): the leaf lived from 1 to 5
    assert index.residence_log == [(22, 1.0, 1.0, 5.0, 2)]

    # stored again later, it is a new residence
    index.apply({"type": "BlockStored", "block_hashes": [22], "parent_block_hash": 11,
                 "token_ids": list(SECOND), "block_size": 16}, now=6.0)
    assert index.match((FIRST, SECOND)) == 2
    assert index.match_with_ages((FIRST, SECOND), now=7.0) == pytest.approx([6.0, 1.0])


def test_a_touch_refreshes_the_ages_of_the_blocks_the_engine_holds():
    index = EventFedIndex(capacity_blocks=8)
    index.apply(STORED, now=1.0)

    # only the first block is held along this path, so only it is refreshed
    assert index.touch((FIRST, OTHER), now=8.0) == 1
    assert index.match_with_ages((FIRST, SECOND), now=10.0) == pytest.approx([2.0, 9.0])


def test_a_copy_is_a_snapshot_that_later_events_and_touches_leave_alone():
    index = EventFedIndex(capacity_blocks=8)
    index.apply(STORED, now=1.0)
    snapshot = index.copy()

    index.apply({"type": "BlockRemoved", "block_hashes": [22]}, now=2.0)
    index.touch((FIRST,), now=3.0)

    assert index.match((FIRST, SECOND)) == 1
    assert snapshot.match((FIRST, SECOND)) == 2
    assert snapshot.match_with_ages((FIRST,), now=3.0) == pytest.approx([2.0])


def test_clearing_all_blocks_empties_the_index_and_counts_every_eviction():
    index = EventFedIndex(capacity_blocks=8)
    index.apply(STORED, now=1.0)
    index.apply({"type": "AllBlocksCleared"}, now=4.0)

    assert index.size == 0
    assert index.evictions == 2
    assert index.match((FIRST, SECOND)) == 0
