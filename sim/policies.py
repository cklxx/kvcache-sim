"""
Eviction and prefetch policies.

Eviction
--------
  LRUPolicy       – classic least-recently-used
  ARCPolicy       – adaptive replacement cache (T1/T2/B1/B2)
  LearnedPolicy   – LightGBM-predicted reuse distance (falls back to LRU)
  BeladyOracle    – offline optimal (requires future access map)

Prefetch
--------
  NoPrefetch           – do nothing
  SessionAwarePrefetch – replay session patterns to predict next blocks
"""
from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Eviction policies
# ======================================================================


class EvictionPolicy(ABC):
    """Base class for all eviction policies."""

    @abstractmethod
    def record_access(self, block_hash: str, current_time: float) -> None:
        """Called every time a block is accessed (read or write)."""

    @abstractmethod
    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        """Return the hash of the block to evict from *blocks*."""

    @abstractmethod
    def remove(self, block_hash: str) -> None:
        """Forget a block (e.g. after it has been evicted)."""

    def name(self) -> str:
        return self.__class__.__name__


# ----------------------------------------------------------------------
# LRU
# ----------------------------------------------------------------------


class LRUPolicy(EvictionPolicy):
    def __init__(self) -> None:
        self._order: OrderedDict[str, None] = OrderedDict()

    def record_access(self, block_hash: str, current_time: float) -> None:
        if block_hash in self._order:
            self._order.move_to_end(block_hash)
        else:
            self._order[block_hash] = None

    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        for h in self._order:          # LRU is at the front
            if h in blocks:
                return h
        return None

    def remove(self, block_hash: str) -> None:
        self._order.pop(block_hash, None)

    def name(self) -> str:
        return "LRU"


# ----------------------------------------------------------------------
# ARC
# ----------------------------------------------------------------------


class ARCPolicy(EvictionPolicy):
    """
    Adaptive Replacement Cache.

    Maintains four lists:
      T1 – recently seen once
      T2 – seen at least twice (frequent)
      B1 – ghost of T1 (metadata only, no data)
      B2 – ghost of T2 (metadata only, no data)
    """

    def __init__(self) -> None:
        self._p: int = 0
        self.T1: OrderedDict[str, None] = OrderedDict()
        self.T2: OrderedDict[str, None] = OrderedDict()
        self.B1: OrderedDict[str, None] = OrderedDict()
        self.B2: OrderedDict[str, None] = OrderedDict()

    def _len_T(self) -> int:
        return len(self.T1) + len(self.T2)

    def record_access(self, block_hash: str, current_time: float) -> None:
        if block_hash in self.T1:
            self.T1.pop(block_hash)
            self.T2[block_hash] = None
        elif block_hash in self.T2:
            self.T2.move_to_end(block_hash)
        elif block_hash in self.B1:
            delta = max(1, len(self.B2) // max(1, len(self.B1)))
            self._p = min(self._p + delta, 10_000)
            self.B1.pop(block_hash)
            self.T2[block_hash] = None
        elif block_hash in self.B2:
            delta = max(1, len(self.B1) // max(1, len(self.B2)))
            self._p = max(self._p - delta, 0)
            self.B2.pop(block_hash)
            self.T2[block_hash] = None
        else:
            self.T1[block_hash] = None

    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        # Try to evict from T1 if it's larger than target p
        if self.T1 and (len(self.T1) > self._p or not self.T2):
            for h in self.T1:
                if h in blocks:
                    return h
        # Try T2
        for h in self.T2:
            if h in blocks:
                return h
        # Fallback: T1
        for h in self.T1:
            if h in blocks:
                return h
        return None

    def _move_to_ghost(self, block_hash: str) -> None:
        if block_hash in self.T1:
            self.T1.pop(block_hash)
            self.B1[block_hash] = None
        elif block_hash in self.T2:
            self.T2.pop(block_hash)
            self.B2[block_hash] = None

    def remove(self, block_hash: str) -> None:
        """Called when a block is physically evicted from the cache tier.
        Move it to the appropriate ghost list so ARC can adapt p on future hits."""
        # Only move to ghost if it's currently in T1 or T2 (live entries).
        # Ghost entries (B1/B2) are left intact so future accesses can trigger
        # the p-adaptation logic in record_access.
        self._move_to_ghost(block_hash)
        # Trim ghost lists to prevent unbounded growth
        _MAX_GHOST = 20_000
        while len(self.B1) > _MAX_GHOST:
            self.B1.popitem(last=False)
        while len(self.B2) > _MAX_GHOST:
            self.B2.popitem(last=False)

    def name(self) -> str:
        return "ARC"


# ----------------------------------------------------------------------
# Learned (LightGBM)
# ----------------------------------------------------------------------


class LearnedPolicy(EvictionPolicy):
    """
    Uses a LightGBM model to predict per-block reuse distance.
    Falls back to LRU when the model is not available.
    """

    def __init__(self) -> None:
        self._access_times: Dict[str, List[float]] = {}
        self._model = None
        self._lru = LRUPolicy()   # fallback + ordering
        self._last_sim_time: float = 0.0

    def set_model(self, model) -> None:
        self._model = model

    def record_access(self, block_hash: str, current_time: float) -> None:
        self._last_sim_time = current_time
        self._access_times.setdefault(block_hash, []).append(current_time)
        self._lru.record_access(block_hash, current_time)

    def _predict_reuse_distance(self, block_hash: str, current_time: float) -> float:
        if self._model is not None:
            try:
                if hasattr(self._model, "predict_reuse_distance"):
                    return float(
                        self._model.predict_reuse_distance(
                            block_hash, self._access_times, current_time
                        )
                    )
                from learned.features import extract_features
                import numpy as np
                feats = extract_features(block_hash, self._access_times, current_time)
                log_dist = float(self._model.predict(np.array([feats]))[0])
                return math.expm1(max(log_dist, 0.0))
            except Exception:
                pass
        # Fallback: average inter-access interval (larger = less frequent = evict first)
        hist = self._access_times.get(block_hash, [])
        if len(hist) < 2:
            return 1e9
        intervals = [hist[i + 1] - hist[i] for i in range(len(hist) - 1)]
        return sum(intervals) / len(intervals)

    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        if not blocks:
            return None
        current_time = self._last_sim_time
        # Sample at most 64 candidates to keep eviction O(1) in practice
        import random as _random
        keys = list(blocks.keys())
        if len(keys) > 64:
            keys = _random.sample(keys, 64)
        # Batch predict for all sampled candidates
        if self._model is not None and keys:
            try:
                from learned.features import extract_features
                import numpy as np
                import pandas as pd
                feat_rows = [extract_features(h, self._access_times, current_time) for h in keys]
                X = pd.DataFrame(feat_rows, columns=[f"f{i}" for i in range(len(feat_rows[0]))])
                preds = self._model.predict(X)
                scores = np.expm1(np.maximum(preds, 0.0))
                return keys[int(np.argmax(scores))]
            except Exception:
                pass
        return max(keys, key=lambda h: self._predict_reuse_distance(h, current_time))

    def remove(self, block_hash: str) -> None:
        self._access_times.pop(block_hash, None)
        self._lru.remove(block_hash)

    def name(self) -> str:
        return "Learned"


# ----------------------------------------------------------------------
# Belady Oracle (offline optimal)
# ----------------------------------------------------------------------


class BeladyOracle(EvictionPolicy):
    """
    Offline optimal eviction: always evict the block whose next access
    is furthest in the future (or never accessed again → ∞).

    Requires *future_access_map*: dict[block_hash → sorted list of future
    simulated timestamps].
    """

    def __init__(self, future_access_map: Dict[str, List[float]]) -> None:
        self._future: Dict[str, List[float]] = future_access_map
        self._current_time: float = 0.0

    def set_current_time(self, t: float) -> None:
        self._current_time = t

    def record_access(self, block_hash: str, current_time: float) -> None:
        self._current_time = current_time

    def _next_access(self, block_hash: str) -> float:
        accesses = self._future.get(block_hash, [])
        if not accesses:
            return float("inf")
        idx = bisect.bisect_right(accesses, self._current_time)
        return accesses[idx] if idx < len(accesses) else float("inf")

    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        if not blocks:
            return None
        return max(blocks, key=self._next_access)

    def remove(self, block_hash: str) -> None:
        pass   # Oracle has no internal state to clean up

    def name(self) -> str:
        return "BeladyOracle"


# ----------------------------------------------------------------------
# Importance-Aware (for long-context / agent workloads)
# ----------------------------------------------------------------------


class ImportancePolicy(EvictionPolicy):
    """
    Importance-aware eviction for long-context and agent workloads.

    Block importance is a weighted combination of:
      - Recency (LRU component)
      - Frequency (access count)
      - Prefix depth (shallow = system prompt/tool defs = critical)
      - Pinned flag (system prompt blocks never evicted)

    Eviction priority (lower = evict first):
      score = recency_score + frequency_score + depth_bonus + pin_bonus

    This models production strategies like H2O (Heavy Hitter Oracle),
    StreamingLLM (keep attention sinks), and SnapKV (observation-based).
    """

    def __init__(
        self,
        pin_depth: int = 0,
        depth_weight: float = 2.0,
        frequency_weight: float = 1.0,
        recency_weight: float = 1.0,
    ) -> None:
        self._access_times: Dict[str, float] = {}  # last access
        self._access_counts: Dict[str, int] = {}
        self._depths: Dict[str, int] = {}
        self._pin_depth = pin_depth  # blocks with depth <= this are pinned
        self._depth_weight = depth_weight
        self._frequency_weight = frequency_weight
        self._recency_weight = recency_weight
        self._current_time: float = 0.0

    def record_access(self, block_hash: str, current_time: float) -> None:
        self._access_times[block_hash] = current_time
        self._access_counts[block_hash] = self._access_counts.get(block_hash, 0) + 1
        self._current_time = current_time

    def set_depth(self, block_hash: str, depth: int) -> None:
        """Set prefix depth for importance scoring."""
        self._depths[block_hash] = depth

    def _score(self, block_hash: str) -> float:
        """Higher score = more important = evict last."""
        # Pinned blocks (system prompt, tool defs) — never evict
        depth = self._depths.get(block_hash, 999)
        if depth <= self._pin_depth:
            return float("inf")

        # Recency: how recently accessed (log scale)
        last = self._access_times.get(block_hash, 0.0)
        recency = 1.0 / (1.0 + self._current_time - last)

        # Frequency: how often accessed
        count = self._access_counts.get(block_hash, 1)
        frequency = min(count, 20) / 20.0  # cap at 20

        # Depth bonus: shallow blocks (near prefix start) are more valuable
        # because they are shared by more requests
        depth_bonus = 1.0 / (1.0 + depth * 0.01)

        return (
            self._recency_weight * recency
            + self._frequency_weight * frequency
            + self._depth_weight * depth_bonus
        )

    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        if not blocks:
            return None
        # Evict the block with the LOWEST importance score
        return min(blocks, key=self._score)

    def remove(self, block_hash: str) -> None:
        self._access_times.pop(block_hash, None)
        self._access_counts.pop(block_hash, None)
        self._depths.pop(block_hash, None)

    def name(self) -> str:
        return "Importance"


# ----------------------------------------------------------------------
# Sliding Window + Landmark (for very long context / agent)
# ----------------------------------------------------------------------


class SlidingWindowPolicy(EvictionPolicy):
    """
    Sliding window eviction with landmark preservation.

    Strategy for agent / very long context:
      1. ALWAYS keep: first ``landmark_blocks`` blocks (system prompt,
         tool defs, initial instructions) — these are "attention sinks"
      2. ALWAYS keep: last ``window_blocks`` blocks (recent turns)
      3. Evict from the MIDDLE using LRU

    This mirrors StreamingLLM / attention-sink patterns where the first
    and last tokens carry disproportionate attention weight.

    ┌──────────┬──────────────────────────────────┬──────────────┐
    │ LANDMARK │      MIDDLE (evictable, LRU)     │    WINDOW    │
    │ (pinned) │                                   │   (pinned)   │
    │ sys+tool │  old turns, stale tool results    │ recent turns │
    └──────────┴──────────────────────────────────┴──────────────┘
    """

    def __init__(
        self,
        landmark_blocks: int = 200,
        window_blocks: int = 500,
    ) -> None:
        self._landmark_blocks = landmark_blocks
        self._window_blocks = window_blocks
        self._lru = LRUPolicy()
        self._depths: Dict[str, int] = {}
        self._max_depth_seen: int = 0

    def record_access(self, block_hash: str, current_time: float) -> None:
        self._lru.record_access(block_hash, current_time)

    def set_depth(self, block_hash: str, depth: int) -> None:
        self._depths[block_hash] = depth
        self._max_depth_seen = max(self._max_depth_seen, depth)

    def _is_protected(self, block_hash: str) -> bool:
        depth = self._depths.get(block_hash, 999)
        # Landmark: first N blocks (system prompt + tool defs)
        if depth < self._landmark_blocks:
            return True
        # Window: last N blocks (recent context)
        if depth > self._max_depth_seen - self._window_blocks:
            return True
        return False

    def evict_candidate(self, blocks: Dict) -> Optional[str]:
        if not blocks:
            return None
        # Only evict from the middle (non-protected) region
        for h in self._lru._order:
            if h in blocks and not self._is_protected(h):
                return h
        # If everything is protected, fall back to pure LRU
        return self._lru.evict_candidate(blocks)

    def remove(self, block_hash: str) -> None:
        self._lru.remove(block_hash)
        self._depths.pop(block_hash, None)

    def name(self) -> str:
        return "SlidingWindow"


# ======================================================================
# Prefetch policies
# ======================================================================


class PrefetchPolicy(ABC):
    @abstractmethod
    def record_sequence(self, session_id: str, block_hashes: List[str]) -> None:
        """Record a completed access sequence for learning."""

    @abstractmethod
    def candidates(self, block_hash: str, session_id: str) -> List[str]:
        """Return block hashes that should be prefetched after accessing *block_hash*."""

    def name(self) -> str:
        return self.__class__.__name__


class NoPrefetch(PrefetchPolicy):
    def record_sequence(self, session_id: str, block_hashes: List[str]) -> None:
        pass

    def candidates(self, block_hash: str, session_id: str) -> List[str]:
        return []

    def name(self) -> str:
        return "NoPrefetch"


class SessionAwarePrefetch(PrefetchPolicy):
    """
    Records completed per-session block sequences and predicts upcoming blocks
    by matching the current block against recent past patterns.
    """

    def __init__(self, window: int = 3, lookahead: int = 2) -> None:
        self._patterns: Dict[str, List[List[str]]] = {}
        self._window = window
        self._lookahead = lookahead

    def record_sequence(self, session_id: str, block_hashes: List[str]) -> None:
        if not block_hashes:
            return
        seqs = self._patterns.setdefault(session_id, [])
        seqs.append(list(block_hashes))
        if len(seqs) > 10:
            seqs.pop(0)

    def candidates(self, block_hash: str, session_id: str) -> List[str]:
        seqs = self._patterns.get(session_id, [])
        seen: Dict[str, int] = {}
        for seq in seqs[-self._window:]:
            if block_hash in seq:
                idx = seq.index(block_hash)
                for k in range(1, self._lookahead + 1):
                    if idx + k < len(seq):
                        h = seq[idx + k]
                        seen[h] = seen.get(h, 0) + 1
        # Return candidates sorted by frequency (most likely first)
        return [h for h, _ in sorted(seen.items(), key=lambda x: -x[1])][: self._lookahead]

    def name(self) -> str:
        return "SessionAwarePrefetch"
