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