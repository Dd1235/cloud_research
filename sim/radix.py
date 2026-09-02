

# one cached token block
class _Node:

    __slots__ = ("key", "parent", "children", "last_access")

    def __init__(self, key, parent):
        self.key = key
        self.parent = parent
        self.children = {}
        self.last_access = 0.0


class PrefixCache:

    def __init__(self, capacity_blocks: int):

        assert capacity_blocks >= 1

        self.capacity = capacity_blocks
        self.root = _Node(None, None)
        self.size = 0
        self.evictions = 0


    def match(self, blocks) -> int:

        node = self.root
        matched = 0
        for block in blocks:
            node = node.children.get(block)
            if node is None:
                break

            matched += 1


        return matched 

    def insert(self, blocks, now: float) -> int:
        node = self.root
        node.last_access = now

        created = 0
        protect = set()

        for block in blocks:
            child = node.children.get(block)

            if child is None:
                child = _Node(block, node)
                node.children[block] = child
                self.size += 1
                created += 1

            child.last_access = now
            protect.add(id(child))
            node = child

        self._evict(protect)
        return created

    def copy(self) -> "PrefixCache":
        """A snapshot: the same blocks with the same recency, in independent nodes.

        This is what a router holds when it reads a worker's cache through a
        periodic scrape, so it has to be a real copy that the worker's later
        inserts and evictions cannot reach. Iterative rather than recursive
        because a long prompt makes a deep path. evictions stays 0: a snapshot
        never evicted anything.
        """
        snapshot = PrefixCache(self.capacity)
        snapshot.size = self.size
        snapshot.root.last_access = self.root.last_access

        pending = [(self.root, snapshot.root)]

        while pending:
            source, target = pending.pop()

            for block, source_child in source.children.items():
                target_child = _Node(block, target)
                target_child.last_access = source_child.last_access
                target.children[block] = target_child
                pending.append((source_child, target_child))

        return snapshot

    def _leaves(self):
        stack = [self.root]

        while stack:
            node = stack.pop()

            if node is not self.root and not node.children:
                yield node

            stack.extend(node.children.values())


    def _evict(self, protect=frozenset()) -> None:
        while self.size > self.capacity:
            candidates = [
                leaf
                for leaf in self._leaves()
                if id(leaf) not in protect
            ]

            if not candidates:
                return

            victim = min(
                candidates,
                key=lambda node: node.last_access,
            )

            del victim.parent.children[victim.key]
            self.size -= 1
            self.evictions += 1