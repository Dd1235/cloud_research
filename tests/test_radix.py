from sim.radix import PrefixCache


def test_empty_cache_matches_nothing():
    cache = PrefixCache(10)

    assert cache.match(("a", "b")) == 0

def test_insert_then_full_and_partial_match():
    cache = PrefixCache(10)

    assert cache.insert(("a", "b", "c"), now=1.0) == 3

    assert cache.match(("a", "b", "c")) == 3
    assert cache.match(("a", "b", "x")) == 2
    assert cache.match(("z",)) == 0
    assert cache.size == 3

def test_shared_prefix_is_stored_once():
    cache = PrefixCache(10)

    cache.insert(("a", "b", "c"), now=1.0)

    assert cache.insert(("a", "b", "d"), now=2.0) == 1
    assert cache.size == 4

def test_evicts_lru_leaf_not_shared_prefix():
    cache = PrefixCache(4)

    cache.insert(("a", "b", "c"), now=1.0)
    cache.insert(("a", "b", "d"), now=2.0)
    cache.insert(("a", "b", "e"), now=3.0)

    assert cache.size == 4
    assert cache.evictions == 1

    assert cache.match(("a", "b", "c")) == 2
    assert cache.match(("a", "b", "d")) == 3
    assert cache.match(("a", "b", "e")) == 3

def test_touching_a_path_refreshes_its_recency():
    cache = PrefixCache(4)

    cache.insert(("a", "b", "c"), now=1.0)
    cache.insert(("a", "b", "d"), now=2.0)

    cache.insert(("a", "b", "c"), now=3.0)
    cache.insert(("a", "b", "e"), now=4.0)

    assert cache.match(("a", "b", "c")) == 3
    assert cache.match(("a", "b", "d")) == 2

def test_never_evicts_just_inserted_path_even_if_over_capacity():
    cache = PrefixCache(2)

    cache.insert(("a", "b", "c", "d"), now=1.0)

    assert cache.match(("a", "b", "c", "d")) == 4

    cache.insert(("x",), now=2.0)

    assert cache.size <= 3
    assert cache.match(("x",)) == 1


def test_copy_matches_the_same_paths_and_keeps_size():
    cache = PrefixCache(10)
    cache.insert(("a", "b", "c"), now=1.0)
    cache.insert(("a", "b", "d"), now=2.0)

    snapshot = cache.copy()

    assert snapshot.match(("a", "b", "d")) == 3
    assert snapshot.match(("a", "b", "c")) == 3
    assert snapshot.match(("a", "x")) == 1
    assert snapshot.size == 4
    assert snapshot.capacity == 10
    assert snapshot.evictions == 0


def test_copy_is_independent_of_the_original():
    cache = PrefixCache(4)
    cache.insert(("a", "b"), now=1.0)

    snapshot = cache.copy()

    # the worker keeps inserting after the scrape: the snapshot must not see it
    cache.insert(("a", "c"), now=2.0)
    assert snapshot.match(("a", "c")) == 1
    assert snapshot.size == 2

    # and filling the snapshot past capacity must not evict from the worker
    snapshot.insert(("x", "y", "z"), now=3.0)
    assert cache.size == 3
    assert cache.evictions == 0


def test_copy_preserves_recency_so_it_evicts_the_same_victim():
    cache = PrefixCache(4)
    cache.insert(("a", "b", "c"), now=1.0)
    cache.insert(("a", "b", "d"), now=2.0)

    snapshot = cache.copy()

    # both are at capacity; inserting one more leaf must evict "c" (older) in each
    cache.insert(("a", "b", "e"), now=3.0)
    snapshot.insert(("a", "b", "e"), now=3.0)

    assert cache.match(("a", "b", "c")) == 2
    assert snapshot.match(("a", "b", "c")) == 2
    assert snapshot.match(("a", "b", "d")) == 3

def test_match_with_ages_reports_how_long_each_matched_block_sat_untouched():
    cache = PrefixCache(10)
    cache.insert(("a", "b"), now=1.0)
    cache.insert(("a", "c"), now=3.0)   # refreshes "a", not "b"

    # "a" was last touched at 3.0, "b" at 1.0
    assert cache.match_with_ages(("a", "b"), now=5.0) == [2.0, 4.0]
    # the walk stops at the first missing block, like match()
    assert cache.match_with_ages(("a", "c", "d"), now=5.0) == [2.0, 2.0]
    assert cache.match_with_ages(("z",), now=5.0) == []
    assert len(cache.match_with_ages(("a", "b"), now=5.0)) == cache.match(("a", "b"))


def test_residence_log_records_how_long_each_evicted_block_lived():
    cache = PrefixCache(2, record_residence=True)
    cache.insert(("a",), now=1.0)
    cache.insert(("b",), now=2.0)
    cache.insert(("c",), now=3.0)   # over capacity: the oldest leaf "a" goes

    assert cache.residence_log == [("a", 1.0, 1.0, 3.0, 1)]

    # a refreshed block reports its insertion and its last touch separately
    cache.insert(("b",), now=4.0)
    cache.insert(("d",), now=5.0)   # evicts "c" (last touched 3.0), not "b"
    assert cache.residence_log[-1] == ("c", 3.0, 3.0, 5.0, 1)


def test_residence_log_is_off_by_default():
    cache = PrefixCache(1)
    cache.insert(("a",), now=1.0)
    cache.insert(("b",), now=2.0)

    assert cache.residence_log is None
    assert cache.evictions == 1


def test_residence_log_records_the_depth_of_the_evicted_block():
    cache = PrefixCache(3, record_residence=True)

    # a three-deep path fills the cache; the next insert evicts its leaf,
    # which sits three edges from the root
    cache.insert(("a", "b", "c"), now=1.0)
    cache.insert(("x",), now=2.0)

    assert cache.residence_log == [("c", 1.0, 1.0, 2.0, 3)]
