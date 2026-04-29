"""
Router — prefix-tree matching + worker selection.

For each incoming request (a list of block hashes representing the
prefix), the router scores each worker by how many of those hashes
are already cached in its HBM tier, then routes to the best match.

Classes
-------
  PrefixTrieNode  – trie node used for prefix matching
  PrefixTrie      – maps block-hash sequences to worker IDs
  Worker          – a single simulated inference worker with its own CacheManager
  Router          – distributes requests across workers
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .cache_manager import CacheManager
from .metrics import Metrics
from .policies import LRUPolicy, NoPrefetch
from .storage import StorageTier


# ======================================================================
# Prefix Trie
# ======================================================================


class PrefixTrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, PrefixTrieNode] = {}
        self.worker_ids: List[int] = []   # workers that have this prefix cached


class PrefixTrie:
    """
    Lightweight prefix trie over sequences of block-hash strings.
    Used to quickly find workers that have a long matching prefix.
    """

    def __init__(self) -> None:
        self.root = PrefixTrieNode()

    def insert(self, prefix: List[str], worker_id: int) -> None:
        node = self.root
        for h in prefix:
            node = node.children.setdefault(h, PrefixTrieNode())
            if worker_id not in node.worker_ids:
                node.worker_ids.append(worker_id)

    def best_match(self, query: List[str]) -> Tuple[int, int]:
        """
        Returns (best_worker_id, match_depth).
        If no match, returns (-1, 0).
        """
        node = self.root
        depth = 0
        best_worker = -1
        best_depth = 0

        for h in query:
            if h not in node.children:
                break
            node = node.children[h]
            depth += 1
            if node.worker_ids:
                best_worker = node.worker_ids[0]
                best_depth = depth

        return best_worker, best_depth


# ======================================================================
# Worker
# ======================================================================


class Worker:
    """A single simulated vLLM-style worker with its own KV-cache."""

    def __init__(self, worker_id: int, cache_manager: CacheManager) -> None:
        self.worker_id = worker_id
        self.cache = cache_manager
        self.cached_prefix_sequences: List[List[str]] = []

    def process(
        self,
        block_hashes: List[str],
        size_bytes: int,
        session_id: str,
        current_time: float,
    ) -> Tuple[int, float]:
        """
        Process a request: read all blocks, write misses.

        Returns (num_hits, total_latency_ms).
        """
        hits = 0
        total_latency = 0.0
        for depth, bh in enumerate(block_hashes):
            hit_tier, lat = self.cache.read(
                bh, size_bytes, depth, current_time, session_id
            )
            if hit_tier:
                hits += 1
                total_latency += lat
            else:
                wlat = self.cache.write(bh, size_bytes, depth, current_time, session_id)
                total_latency += wlat

        self.cached_prefix_sequences.append(block_hashes)
        if len(self.cached_prefix_sequences) > 50:
            self.cached_prefix_sequences.pop(0)

        self.cache.tick(current_time)
        return hits, total_latency

    def cached_hashes(self) -> set:
        """Return the set of block hashes in HBM."""
        return set(self.cache.tiers[0].blocks.keys())


# ======================================================================
# Router
# ======================================================================


class Router:
    """
    Routes each request to the worker with the best prefix cache hit.

    Strategy
    --------
    1. Score each worker: count how many of the request's block hashes
       are already in that worker's HBM.
    2. Route to highest-scoring worker; break ties by worker ID.
    """

    def __init__(self, workers: List[Worker]) -> None:
        self.workers = workers
        self.trie = PrefixTrie()
        self._round_robin: int = 0

    def route(self, block_hashes: List[str]) -> Worker:
        if not block_hashes:
            return self._next_rr()

        best_worker: Optional[Worker] = None
        best_score = -1

        for w in self.workers:
            cached = w.cached_hashes()
            score = sum(1 for h in block_hashes if h in cached)
            if score > best_score:
                best_score = score
                best_worker = w

        return best_worker if best_score > 0 else self._next_rr()

    def _next_rr(self) -> Worker:
        w = self.workers[self._round_robin % len(self.workers)]
        self._round_robin += 1
        return w

    def aggregate_metrics(self) -> Metrics:
        """Merge metrics from all workers."""
        combined = Metrics(tier_names=self.workers[0].cache.metrics.tier_names)
        for w in self.workers:
            m = w.cache.metrics
            combined.total_requests += m.total_requests
            combined.total_hits += m.total_hits
            combined.total_misses += m.total_misses
            combined.evictions += m.evictions
            combined.promotions += m.promotions
            combined.demotions += m.demotions
            combined.prefetches += m.prefetches
            combined.total_latency_ms += m.total_latency_ms
            for tier in m.tier_names:
                combined.tier_hits[tier] = combined.tier_hits.get(tier, 0) + m.tier_hits.get(tier, 0)
                combined.tier_latency_ms[tier] = combined.tier_latency_ms.get(tier, 0.0) + m.tier_latency_ms.get(tier, 0.0)
            for tier in w.cache.tiers:
                combined.record_storage(
                    tier.name,
                    tier.used,
                    tier.capacity_bytes,
                    len(tier.blocks),
                )
        return combined
