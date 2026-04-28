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


@dataclass(eq=False)
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
    KV cache radix tree with path-aware prefix lookup and leaf eviction.

    The ``_hash_index`` dict answers membership queries, but prefix lookup
    always follows parent->child edges. This prevents a block cached on a
    different branch from being counted as a contiguous prefix hit.
    """

    def __init__(self, capacity_bytes: int = 0, block_size: int = 4096) -> None:
        self.root = RadixTreeNode(block_hash="__ROOT__", size_bytes=0)
        self._hash_index: Dict[str, List[RadixTreeNode]] = {}
        self._evictable_leaves: Set[RadixTreeNode] = set()
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
        if not block_hashes:
            return 0

        new_count, _ = self._insert_from_parent(
            self.root,
            block_hashes,
            size_bytes,
            current_time,
            start_depth=0,
        )
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
        parent = self.root
        for bh in block_hashes:
            node = parent.children.get(bh)
            if node is None:
                break
            matched.append(node)
            parent = node
        return len(matched), matched

    def insert_suffix_after_prefix(
        self,
        full_sequence: List[str],
        prefix_depth: int,
        suffix_hashes: List[str],
        size_bytes: int,
        current_time: float,
    ) -> int:
        """
        Insert suffix_hashes under an already matched prefix of full_sequence.

        This is the explicit form for the common prefill flow:
        lookup_prefix(full_sequence) followed by inserting only the missing
        suffix. Plain insert_sequence remains stateless and always inserts the
        given sequence from the root.
        """
        if not suffix_hashes:
            return 0
        if prefix_depth < 0 or prefix_depth > len(full_sequence):
            raise ValueError("prefix_depth is outside full_sequence")
        suffix_end = prefix_depth + len(suffix_hashes)
        if full_sequence[prefix_depth:suffix_end] != suffix_hashes:
            raise ValueError("suffix_hashes must be contiguous in full_sequence")

        parent = self.root
        for bh in full_sequence[:prefix_depth]:
            node = parent.children.get(bh)
            if node is None:
                raise ValueError("matched prefix is not present in the tree")
            parent = node

        new_count, _ = self._insert_from_parent(
            parent,
            suffix_hashes,
            size_bytes,
            current_time,
            start_depth=prefix_depth,
        )
        return new_count

    def contains(self, block_hash: str) -> bool:
        return block_hash in self._hash_index

    def get(self, block_hash: str) -> Optional[RadixTreeNode]:
        nodes = self._hash_index.get(block_hash)
        if not nodes:
            return None
        return nodes[0]

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def acquire_sequence(self, block_hashes: List[str]) -> None:
        """Increment ref_count for all contiguous nodes in this prefix path."""
        for node in self._walk_existing_path(block_hashes):
            node.ref_count += 1
            self._evictable_leaves.discard(node)

    def release_sequence(self, block_hashes: List[str]) -> None:
        """Decrement ref_count. Called when a request completes."""
        for node in self._walk_existing_path(block_hashes):
            node.ref_count = max(0, node.ref_count - 1)
            if node.is_evictable:
                self._evictable_leaves.add(node)

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
        best_node: Optional[RadixTreeNode] = None
        best_time = float("inf")
        stale: List[RadixTreeNode] = []
        for node in self._evictable_leaves:
            if not self._is_indexed(node) or not node.is_evictable:
                stale.append(node)
                continue
            if node.last_access_time < best_time:
                best_time = node.last_access_time
                best_node = node
        for node in stale:
            self._evictable_leaves.discard(node)
        if best_node is None:
            return None
        evicted_hash = best_node.block_hash
        self._remove_node(best_node)
        return evicted_hash

    def _remove_node(self, node: RadixTreeNode) -> None:
        """Remove a leaf node from the tree."""
        if not self._is_indexed(node):
            return
        self._evictable_leaves.discard(node)
        self.used_bytes -= node.size_bytes
        # Detach from parent
        if node.parent is not None:
            if node.parent.children.get(node.block_hash) is node:
                node.parent.children.pop(node.block_hash, None)
            # Parent may become an evictable leaf
            if node.parent.is_evictable and node.parent.block_hash != "__ROOT__":
                self._evictable_leaves.add(node.parent)
        self._unindex_node(node)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _insert_from_parent(
        self,
        parent: RadixTreeNode,
        block_hashes: List[str],
        size_bytes: int,
        current_time: float,
        start_depth: int,
    ) -> Tuple[int, RadixTreeNode]:
        """Insert a contiguous path under parent."""
        new_count = 0
        for offset, bh in enumerate(block_hashes):
            node, created = self._ensure_child(
                parent,
                bh,
                size_bytes,
                current_time,
                prefix_depth=start_depth + offset,
            )
            if created:
                new_count += 1
            parent = node
        if parent.block_hash != "__ROOT__" and parent.is_evictable:
            self._evictable_leaves.add(parent)
        return new_count, parent

    def _ensure_child(
        self,
        parent: RadixTreeNode,
        block_hash: str,
        size_bytes: int,
        current_time: float,
        prefix_depth: int,
    ) -> Tuple[RadixTreeNode, bool]:
        """Return parent.children[block_hash], creating it if missing."""
        node = parent.children.get(block_hash)
        if node is not None:
            node.touch(current_time)
            self._evictable_leaves.discard(node)
            return node, False

        self._evictable_leaves.discard(parent)
        if self.capacity_bytes > 0:
            while self.used_bytes + size_bytes > self.capacity_bytes:
                evicted = self._evict_one_leaf()
                if evicted is None:
                    break

        node = RadixTreeNode(
            block_hash=block_hash,
            size_bytes=size_bytes,
            prefix_depth=prefix_depth,
            last_access_time=current_time,
            access_count=1,
            parent=parent,
        )
        parent.children[block_hash] = node
        self._index_node(node)
        self.used_bytes += size_bytes
        return node, True

    def _walk_existing_path(self, block_hashes: List[str]) -> List[RadixTreeNode]:
        """Return contiguous existing nodes from root for block_hashes."""
        matched: List[RadixTreeNode] = []
        parent = self.root
        for bh in block_hashes:
            node = parent.children.get(bh)
            if node is None:
                break
            matched.append(node)
            parent = node
        return matched

    def _index_node(self, node: RadixTreeNode) -> None:
        self._hash_index.setdefault(node.block_hash, []).append(node)

    def _unindex_node(self, node: RadixTreeNode) -> None:
        nodes = self._hash_index.get(node.block_hash)
        if not nodes:
            return
        for idx, candidate in enumerate(nodes):
            if candidate is node:
                nodes.pop(idx)
                break
        if not nodes:
            self._hash_index.pop(node.block_hash, None)

    def _is_indexed(self, node: RadixTreeNode) -> bool:
        nodes = self._hash_index.get(node.block_hash, [])
        return any(candidate is node for candidate in nodes)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_blocks(self) -> int:
        return sum(len(nodes) for nodes in self._hash_index.values())

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
