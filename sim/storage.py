"""
StorageTier and KVBlock — core data structures.

No real memory is allocated; we only track metadata and simulate
latency based on bandwidth/latency parameters.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class KVBlock:
    """A single KV-cache block (one page of attention keys/values)."""

    block_hash: str          # SHA-like hash identifying the prefix
    size_bytes: int          # bytes occupied (constant per block in practice)
    prefix_depth: int        # number of completed prefix tokens this block covers
    ref_count: int = 0       # active references (in-flight requests)
    last_access_time: float = 0.0
    access_count: int = 0
    tier: str = "HBM"        # current tier name

    def touch(self, current_time: float) -> None:
        self.last_access_time = current_time
        self.access_count += 1

    def __repr__(self) -> str:
        return (
            f"KVBlock(hash={self.block_hash[:8]}, depth={self.prefix_depth}, "
            f"tier={self.tier}, hits={self.access_count})"
        )


class BlockStore(dict):
    """Dictionary with tier-local recency ordering for fast victim selection."""

    def __init__(self) -> None:
        super().__init__()
        self._order: OrderedDict[str, None] = OrderedDict()

    def __setitem__(self, key: str, value: KVBlock) -> None:
        super().__setitem__(key, value)
        self.touch(key)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._order.pop(key, None)

    def pop(self, key: str, default=None):
        value = super().pop(key, default)
        self._order.pop(key, None)
        return value

    def clear(self) -> None:
        super().clear()
        self._order.clear()

    def touch(self, key: str) -> None:
        if key not in self:
            return
        self._order[key] = None
        self._order.move_to_end(key)

    def oldest_key(
        self,
        membership: Optional[Dict] = None,
        max_scan: int = 64,
    ) -> Optional[str]:
        oldest: Optional[str] = None
        scanned = 0
        for key in self._order:
            if key not in self:
                continue
            if oldest is None:
                oldest = key
            if membership is None or key in membership:
                return key
            scanned += 1
            if scanned >= max_scan:
                break
        return oldest

    def oldest_keys(
        self,
        limit: int,
        membership: Optional[Dict] = None,
        max_scan: int = 512,
    ) -> list[str]:
        keys: list[str] = []
        scanned = 0
        for key in self._order:
            if key not in self:
                continue
            if membership is None or key in membership:
                keys.append(key)
                if len(keys) >= limit:
                    break
            scanned += 1
            if scanned >= max_scan:
                break
        return keys


class StorageTier:
    """
    Simulated storage tier (HBM / DRAM / SSD).

    Parameters
    ----------
    name              : human-readable label
    capacity_bytes    : total byte capacity
    read_bw_gbps      : read bandwidth in GB/s
    write_bw_gbps     : write bandwidth in GB/s
    read_latency_ms   : base read latency (queue + access) in ms
    """

    def __init__(
        self,
        name: str,
        capacity_bytes: int,
        read_bw_gbps: float,
        write_bw_gbps: float,
        read_latency_ms: float,
    ) -> None:
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.read_bw = read_bw_gbps      # GB/s
        self.write_bw = write_bw_gbps    # GB/s
        self.read_latency = read_latency_ms  # ms
        self.used: int = 0
        self.blocks: BlockStore = BlockStore()

    # ------------------------------------------------------------------
    # Capacity helpers
    # ------------------------------------------------------------------

    def has_space(self, size_bytes: int) -> bool:
        return self.used + size_bytes <= self.capacity_bytes

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used

    @property
    def utilization(self) -> float:
        return self.used / self.capacity_bytes if self.capacity_bytes > 0 else 0.0

    # ------------------------------------------------------------------
    # Block operations
    # ------------------------------------------------------------------

    def insert(self, block: KVBlock) -> bool:
        """Insert block. Returns False if no capacity."""
        if not self.has_space(block.size_bytes):
            return False
        self.blocks[block.block_hash] = block
        self.used += block.size_bytes
        block.tier = self.name
        return True

    def remove(self, block_hash: str) -> Optional[KVBlock]:
        """Remove and return block (or None)."""
        block = self.blocks.pop(block_hash, None)
        if block is not None:
            self.used -= block.size_bytes
        return block

    def get(self, block_hash: str) -> Optional[KVBlock]:
        block = self.blocks.get(block_hash)
        if block is not None:
            self.blocks.touch(block_hash)
        return block

    def contains(self, block_hash: str) -> bool:
        return block_hash in self.blocks

    # ------------------------------------------------------------------
    # Latency model
    # ------------------------------------------------------------------

    def transfer_latency_ms(self, size_bytes: int, is_read: bool = True) -> float:
        """
        Simulated transfer latency (ms) for size_bytes.
        latency = base_latency + transfer_time
        """
        bw_bps = (self.read_bw if is_read else self.write_bw) * 1e9
        transfer_ms = (size_bytes / bw_bps) * 1000.0
        return self.read_latency + transfer_ms

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StorageTier({self.name}, "
            f"{self.used / 1e9:.2f}/{self.capacity_bytes / 1e9:.2f} GB, "
            f"{self.utilization:.1%})"
        )
