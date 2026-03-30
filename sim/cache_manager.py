"""
CacheManager — orchestrates the multi-tier KV-cache hierarchy.

Tier order (hot → cold): HBM → DRAM → SSD

Public API
----------
  read(block_hash, size, depth, ts)  → (hit_tier | None, latency_ms)
  write(block_hash, size, depth, ts) → latency_ms
  tick(ts)                           → run prefetch + background eviction
  get_metrics()                      → Metrics snapshot
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .metrics import Metrics
from .policies import EvictionPolicy, LRUPolicy, NoPrefetch, PrefetchPolicy
from .storage import KVBlock, StorageTier


class CacheManager:
    def __init__(
        self,
        tiers: List[StorageTier],
        eviction_policy: Optional[EvictionPolicy] = None,
        prefetch_policy: Optional[PrefetchPolicy] = None,
        promote_on_hit: bool = True,
        selective_write: bool = False,
        selective_write_depth: int = 3,
    ) -> None:
        self.tiers = tiers                      # ordered hot→cold
        self.eviction = eviction_policy or LRUPolicy()
        self.prefetch = prefetch_policy or NoPrefetch()
        self.promote_on_hit = promote_on_hit
        self.selective_write = selective_write
        self.selective_write_depth = selective_write_depth
        self.metrics = Metrics(tier_names=[t.name for t in tiers])
        self._pending_writes: List[Tuple[str, KVBlock, int]] = []  # (tier_idx, block, eta_ts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        block_hash: str,
        size_bytes: int,
        prefix_depth: int,
        current_time: float,
        session_id: str = "",
    ) -> Tuple[Optional[str], float]:
        """
        Look up a block across tiers (HBM first).

        Returns
        -------
        (hit_tier_name, latency_ms)  — hit_tier_name is None on miss.
        """
        self.metrics.total_requests += 1

        for tier_idx, tier in enumerate(self.tiers):
            block = tier.get(block_hash)
            if block is not None:
                block.touch(current_time)
                self.eviction.record_access(block_hash, current_time)
                latency = tier.transfer_latency_ms(size_bytes, is_read=True)
                self.metrics.record_hit(tier.name, latency)

                # Promote to warmer tiers
                if self.promote_on_hit and tier_idx > 0:
                    self._promote(block, tier_idx, current_time)

                # Trigger session-aware prefetch
                for candidate in self.prefetch.candidates(block_hash, session_id):
                    self._prefetch_block(candidate, size_bytes, prefix_depth, current_time)

                return tier.name, latency

        self.metrics.record_miss()
        return None, 0.0

    def write(
        self,
        block_hash: str,
        size_bytes: int,
        prefix_depth: int,
        current_time: float,
        session_id: str = "",
    ) -> float:
        """
        Insert a newly computed block into the cache.

        Selective-write mode: skip deep prefix blocks (they're unlikely to
        be reused soon).
        """
        if self.selective_write and prefix_depth > self.selective_write_depth:
            return 0.0

        # Already cached somewhere — skip
        for tier in self.tiers:
            if tier.contains(block_hash):
                return 0.0

        block = KVBlock(
            block_hash=block_hash,
            size_bytes=size_bytes,
            prefix_depth=prefix_depth,
            last_access_time=current_time,
            access_count=1,
        )
        self.eviction.record_access(block_hash, current_time)

        # Always write to HBM (tier 0) first; back-fill lower tiers lazily
        latency = self._insert(block, tier_idx=0, current_time=current_time)

        # Async backup to tier 1 (DRAM) if available
        if len(self.tiers) > 1:
            self._pending_writes.append((1, block, current_time + 5.0))

        return latency

    def tick(self, current_time: float) -> None:
        """Periodic maintenance: flush pending writes, evict if needed."""
        self._flush_pending_writes(current_time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert(self, block: KVBlock, tier_idx: int, current_time: float) -> float:
        """Insert block into tier[tier_idx], evicting if necessary."""
        if tier_idx >= len(self.tiers):
            return 0.0
        tier = self.tiers[tier_idx]

        # Evict until space is available
        while not tier.has_space(block.size_bytes):
            victim = self.eviction.evict_candidate(tier.blocks)
            if victim is None:
                break
            evicted = tier.remove(victim)
            # Do NOT remove from eviction policy — demoted blocks must remain
            # tracked so they can be evicted from lower tiers later.
            if evicted is not None:
                self.metrics.evictions += 1
                # Demote to next tier if possible
                if tier_idx + 1 < len(self.tiers):
                    self._demote(evicted, tier_idx + 1, current_time)
                else:
                    # Evicted from the coldest tier — now truly remove from policy
                    self.eviction.remove(victim)

        if tier.insert(block):
            return tier.transfer_latency_ms(block.size_bytes, is_read=False)
        return 0.0

    def _promote(self, block: KVBlock, from_tier_idx: int, current_time: float) -> None:
        """Move block up to tier 0 (HBM), cascading evictions down."""
        source_tier = self.tiers[from_tier_idx]
        source_tier.remove(block.block_hash)

        # Re-insert at the top, potentially cascading demotions
        self._insert(block, tier_idx=0, current_time=current_time)
        self.metrics.promotions += 1

    def _demote(self, block: KVBlock, to_tier_idx: int, current_time: float) -> None:
        """Move block down to a colder tier."""
        if to_tier_idx >= len(self.tiers):
            return
        self._insert(block, tier_idx=to_tier_idx, current_time=current_time)
        self.metrics.demotions += 1

    def _prefetch_block(
        self,
        block_hash: str,
        size_bytes: int,
        prefix_depth: int,
        current_time: float,
    ) -> None:
        """Pre-load a block from lower tiers into HBM."""
        # Check if already in HBM
        if self.tiers[0].contains(block_hash):
            return
        # Check lower tiers
        for tier_idx in range(1, len(self.tiers)):
            block = self.tiers[tier_idx].get(block_hash)
            if block is not None:
                self.tiers[tier_idx].remove(block_hash)
                self._insert(block, tier_idx=0, current_time=current_time)
                self.metrics.prefetches += 1
                return

    def _flush_pending_writes(self, current_time: float) -> None:
        remaining = []
        for tier_idx, block, eta in self._pending_writes:
            if current_time >= eta:
                if not self.tiers[tier_idx].contains(block.block_hash):
                    self._insert(block, tier_idx=tier_idx, current_time=current_time)
            else:
                remaining.append((tier_idx, block, eta))
        self._pending_writes = remaining

    # ------------------------------------------------------------------

    def get_metrics(self) -> Metrics:
        return self.metrics

    def reset_metrics(self) -> None:
        self.metrics = Metrics(tier_names=[t.name for t in self.tiers])

    def tier_utilizations(self) -> Dict[str, float]:
        return {t.name: t.utilization for t in self.tiers}
