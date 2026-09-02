from .radix import PrefixCache


class CacheView:
    """What the router believes a worker's cache holds.

    Policies only ever call match(). The router additionally reports every
    dispatch through record_dispatch(), which only views that build their own
    picture from routing decisions care about; the rest ignore it.
    """

    def match(self, blocks) -> int:
        raise NotImplementedError

    def record_dispatch(self, blocks, now: float) -> None:
        pass


class PerfectView(CacheView):
    """The worker's true cache, read live at the moment of the decision.

    This is what every policy did before views existed. It is the Δt = 0 point
    of the staleness sweep and the upper bound nothing real can reach.
    """

    def __init__(self, cache: PrefixCache):
        self._cache = cache

    def match(self, blocks) -> int:
        return self._cache.match(blocks)


VIEW_KINDS = ("perfect",)


def make_view_factory(kind: str, engine, *, period=None, shadow_blocks=None):
    """Returns a function worker -> CacheView for the requested view model.

    None means the perfect view; the router then wraps each worker's own cache.
    """
    if kind == "perfect":
        return None

    known_kinds = ", ".join(VIEW_KINDS)
    raise KeyError(f"unknown view {kind!r}; known views: {known_kinds}")
