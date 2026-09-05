"""A worker's cache as the router can know it from the engine's kv events.

vllm publishes BlockStored (the hashes it assigned, their parent hash, and
the token ids each block holds) and BlockRemoved events. Applied as they
arrive this index is the true cache contents to within the event lag; copied
every P seconds it is the periodic snapshot the simulator studied. It speaks
PrefixCache's dialect (match, match_with_ages, copy, size, evictions,
residence_log) so the simulator's views wrap it unchanged.

Matching walks a prompt's engine blocks, tuples of token ids, from the root:
the first block's parent is None and each later block's parent is the hash
the engine assigned to the block before it. The router never recomputes
vllm's hash function; it replays the chain the engine reported, so hashes
need not even agree between engines.

last_access is what a router can know: the time the engine stored the block,
or the router's own latest dispatch of it to this worker (touch). A cache hit
inside the engine emits no event, so a block kept warm by traffic that did
not pass through this router looks older than it is; all traffic does.
"""
from dataclasses import dataclass, replace


@dataclass
class _Entry:
    parent: object
    tokens: tuple
    stored_at: float
    last_access: float
    depth: int


class EventFedIndex:

    def __init__(self, capacity_blocks: int, record_residence: bool = False):
        self.capacity = capacity_blocks
        self.evictions = 0
        self.residence_log = [] if record_residence else None

        self._entries: dict[object, _Entry] = {}
        self._by_path: dict[tuple, object] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    def apply(self, event: dict, now: float) -> None:
        kind = event["type"]

        if kind == "BlockStored":
            self._store(event, now)
        elif kind == "BlockRemoved":
            for block_hash in event["block_hashes"]:
                self._remove(block_hash, now)
        elif kind == "AllBlocksCleared":
            for block_hash in list(self._entries):
                self._remove(block_hash, now)
        else:
            raise ValueError(f"unknown kv event {kind!r}")

    def _store(self, event: dict, now: float) -> None:
        block_size = event["block_size"]
        token_ids = event["token_ids"]
        parent = event.get("parent_block_hash")

        for index, block_hash in enumerate(event["block_hashes"]):
            tokens = tuple(token_ids[index * block_size:(index + 1) * block_size])

            # the engine re-announces a block it already holds; that is a
            # touch, not a new residence
            if block_hash in self._entries:
                self._entries[block_hash].last_access = now
            else:
                parent_entry = self._entries.get(parent)
                depth = 1 if parent_entry is None else parent_entry.depth + 1
                self._entries[block_hash] = _Entry(parent, tokens, now, now, depth)
                self._by_path[(parent, tokens)] = block_hash

            parent = block_hash

    def _remove(self, block_hash, now: float) -> None:
        entry = self._entries.pop(block_hash, None)
        if entry is None:
            return

        del self._by_path[(entry.parent, entry.tokens)]
        self.evictions += 1

        if self.residence_log is not None:
            self.residence_log.append(
                (block_hash, entry.stored_at, entry.last_access, now, entry.depth),
            )

    def _walk(self, blocks):
        """The entries along a prompt's blocks, up to the first one the engine does not hold."""
        parent = None
        for tokens in blocks:
            block_hash = self._by_path.get((parent, tuple(tokens)))
            if block_hash is None:
                return

            yield self._entries[block_hash]
            parent = block_hash

    def match(self, blocks) -> int:
        return sum(1 for _ in self._walk(blocks))

    def match_with_ages(self, blocks, now: float) -> list[float]:
        return [now - entry.last_access for entry in self._walk(blocks)]

    def touch(self, blocks, now: float) -> int:
        """The router sent these blocks here: the ones the engine holds are about to be hit."""
        touched = 0
        for entry in self._walk(blocks):
            entry.last_access = now
            touched += 1

        return touched

    def copy(self) -> "EventFedIndex":
        """A snapshot that later events and touches leave alone."""
        snapshot = EventFedIndex(self.capacity)
        snapshot.evictions = self.evictions
        snapshot._entries = {
            block_hash: replace(entry)
            for block_hash, entry in self._entries.items()
        }
        snapshot._by_path = dict(self._by_path)

        return snapshot
