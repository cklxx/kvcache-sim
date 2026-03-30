"""
RadixTree — KV cache prefix tree with reference counting.

Models vLLM-style radix-tree block management:
  - Prefix sharing: sessions with common token prefixes share tree nodes
  - Reference counting: in-flight requests pin nodes, preventing eviction
  - Leaf-only eviction: only unreferenced leaf nodes can be removed,
    preserving shared prefix invariant

Tree structure
--------------
  root (virtual)
  ├── hash_A (system prompt block 0)
  │   ├── hash_B (system prompt block 1)
  │   │   ├── hash_C (session 1, block 2)  ← leaf, evictable
  │   │   └── hash_D (session 2, block 2)  ← leaf, evictable
  │   …
  └── hash_X (different system prompt)
      └── …

Block identity depends on parent hash (incremental chaining), so the
tree naturally mirrors the token-prefix DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class RadixTreeNode:
    """A single node in the KV cache radix tree."""

    block_hash: str
    size_bytes: int = 0
    prefix_depth: int = 0
    ref_count: int = 0
    last_access_time: float = 0.0
    access_count: int = 0
    parent: Optional["RadixTreeNode"] = field(default=None, repr=False)
    children: Dict[str, "RadixTreeNode"] = field(default_factory=dict)

    def touch(self, current_time: float) -> None:
        self.last_access_time = current_time
        self.access_count += 1

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_evictable(self) -> bool:
        return self.is_leaf and self.ref_count == 0

    def __repr__(self) -> str:
        return (
            f"RadixTreeNode(hash={self.block_hash[:8]}, depth={self.prefix_depth}, "
            f"ref={self.ref_count}, children={len(self.children)}, "
            f"hits={self.access_count})"
        )


class RadixTree:
    """
    KV cache radix tree with O(1) hash lookup and prefix-aware eviction.

    The ``_hash_index`` dict provides fast lookups (same perf as flat dict),
    while the tree structure enables reference counting and leaf-only eviction.
    """

    def __init__(self, capacity_bytes: int = 0, block_size: int = 4096) -> None:
        self.root = RadixTreeNode(block_hash="__ROOT__", size_bytes=0)
        self._hash_index: Dict[str, RadixTreeNode] = {}
        self._evictable_leaves: Set[str] = set()
        self.capacity_bytes = capacity_bytes
        self.block_size = block_size
        self.used_bytes: int = 0

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_sequence(
        self,
        block_hashes: List[str],
        size_bytes: int,
        current_time: float,
    ) -> int:
        """
        Insert a prefix sequence into the tree.

        Shares existing prefix nodes; creates new nodes only for novel
        suffixes.  Returns the number of *new* nodes created.
        """
        new_count = 0
        parent = self.root
        for depth, bh in enumerate(block_hashes):
            if bh in self._hash_index:
                # Node exists — just touch it
                node = self._hash_index[bh]
                node.touch(current_time)
                # It might have become a non-leaf if it was evictable
                self._evictable_leaves.discard(bh)
                parent = node
            else:
                # Need space?
                if self.capacity_bytes > 0:
                    while self.used_bytes + size_bytes > self.capacity_bytes:
                        evicted = self._evict_one_leaf()
                        if evicted is None:
                            break  # no evictable leaves
                node = RadixTreeNode(
                    block_hash=bh,
                    size_bytes=size_bytes,
                    prefix_depth=depth,
                    last_access_time=current_time,
                    access_count=1,
                    parent=parent,
                )
                parent.children[bh] = node
                self._hash_index[bh] = node
                self.used_bytes += size_bytes
                new_count += 1
                # Parent is no longer a leaf
                self._evictable_leaves.discard(parent.block_hash)
                parent = node
        # The last node is a potential leaf
        if block_hashes:
            last = block_hashes[-1]
            if last in self._hash_index:
                node = self._hash_index[last]
                if node.is_evictable:
                    self._evictable_leaves.add(last)
        return new_count

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_prefix(
        self, block_hashes: List[str]
    ) -> Tuple[int, List[RadixTreeNode]]:
        """
        Walk the tree matching block hashes.

        Returns (match_depth, list_of_matched_nodes).
        Stops at the first miss.
        """
        matched: List[RadixTreeNode] = []
        for bh in block_hashes:
            node = self._hash_index.get(bh)
            if node is None:
                break
            matched.append(node)
        return len(matched), matched

    def contains(self, block_hash: str) -> bool:
        return block_hash in self._hash_index

    def get(self, block_hash: str) -> Optional[RadixTreeNode]:
        return self._hash_index.get(block_hash)

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def acquire_sequence(self, block_hashes: List[str]) -> None:
        """Increment ref_count for all nodes in this prefix path."""
        for bh in block_hashes:
            node = self._hash_index.get(bh)
            if node is not None:
                node.ref_count += 1
                self._evictable_leaves.discard(bh)

    def release_sequence(self, block_hashes: List[str]) -> None:
        """Decrement ref_count. Called when a request completes."""
        for bh in block_hashes:
            node = self._hash_index.get(bh)
            if node is not None:
                node.ref_count = max(0, node.ref_count - 1)
                if node.is_evictable:
                    self._evictable_leaves.add(bh)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_leaf(self) -> Optional[str]:
        """
        Evict one leaf node with ref_count==0 (LRU among evictable leaves).

        Only leaves can be evicted — this preserves the prefix sharing
        invariant.  Returns the evicted block_hash, or None.
        """
        return self._evict_one_leaf()

    def _evict_one_leaf(self) -> Optional[str]:
        """Pick the LRU evictable leaf and remove it."""
        if not self._evictable_leaves:
            return None
        # LRU: pick the leaf with the oldest last_access_time
        best_hash: Optional[str] = None
        best_time = float("inf")
        for bh in self._evictable_leaves:
            node = self._hash_index.get(bh)
            if node is not None and node.last_access_time < best_time:
                best_time = node.last_access_time
                best_hash = bh
        if best_hash is None:
            return None
        self._remove_node(best_hash)
        return best_hash

    def _remove_node(self, block_hash: str) -> None:
        """Remove a leaf node from the tree."""
        node = self._hash_index.pop(block_hash, None)
        if node is None:
            return
        self._evictable_leaves.discard(block_hash)
        self.used_bytes -= node.size_bytes
        # Detach from parent
        if node.parent is not None:
            node.parent.children.pop(block_hash, None)
            # Parent may become an evictable leaf
            if node.parent.is_evictable and node.parent.block_hash != "__ROOT__":
                self._evictable_leaves.add(node.parent.block_hash)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_blocks(self) -> int:
        return len(self._hash_index)

    @property
    def total_bytes(self) -> int:
        return self.used_bytes

    @property
    def utilization(self) -> float:
        if self.capacity_bytes <= 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes

    def cached_hashes(self) -> set:
        """Return set of all block hashes in the tree (for routing)."""
        return set(self._hash_index.keys())

    def __repr__(self) -> str:
        return (
            f"RadixTree(blocks={self.total_blocks}, "
            f"used={self.used_bytes / 1e6:.2f}MB, "
            f"evictable={len(self._evictable_leaves)})"
        )
