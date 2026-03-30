"""
TraceReplayer — drives a CacheManager through a request trace.

Usage
-----
  replayer = TraceReplayer(cache_manager, prefetch_policy)
  metrics  = replayer.run(requests, warmup=200)
"""
from __future__ import annotations

from typing import List, Optional

from sim.cache_manager import CacheManager
from sim.metrics import Metrics
from sim.policies import BeladyOracle, PrefetchPolicy, SessionAwarePrefetch
from trace.generator import Request


class TraceReplayer:
    def __init__(
        self,
        cache_manager: CacheManager,
        warmup_count: int = 0,
        verbose: bool = False,
    ) -> None:
        self.cache = cache_manager
        self.warmup_count = warmup_count
        self.verbose = verbose

    def run(self, requests: List[Request]) -> Metrics:
        """
        Replay all requests.  Requests before *warmup_count* populate the
        cache but their metrics are discarded.

        Also feeds session sequences into the prefetch policy for learning.
        """
        self.cache.reset_metrics()
        prefetch = self.cache.prefetch

        # Pre-build session sequence map for SessionAwarePrefetch
        if isinstance(prefetch, SessionAwarePrefetch):
            _seed_session_patterns(prefetch, requests)

        # Pre-build future access map for BeladyOracle
        if isinstance(self.cache.eviction, BeladyOracle):
            _seed_belady(self.cache.eviction, requests)

        for i, req in enumerate(requests):
            is_warmup = i < self.warmup_count
            if is_warmup:
                # Run without recording metrics so the cache is warm
                self._replay_request(req, record=False)
                continue

            self._replay_request(req, record=True)

            if self.verbose and i % 500 == 0:
                m = self.cache.metrics
                print(f"  [{i}/{len(requests)}] hit_rate={m.hit_rate:.2%} "
                      f"evictions={m.evictions}")

        return self.cache.get_metrics()

    def _replay_request(self, req: Request, record: bool) -> None:
        if not record:
            # Temporarily suppress metric recording
            orig = self.cache.metrics.total_requests
            for depth, bh in enumerate(req.block_hashes):
                hit_tier, _ = self.cache.read(
                    bh, req.block_size, depth, req.timestamp, req.session_id
                )
                if not hit_tier:
                    self.cache.write(
                        bh, req.block_size, depth, req.timestamp, req.session_id
                    )
            self.cache.tick(req.timestamp)
            # Restore counters
            self.cache.reset_metrics()
        else:
            for depth, bh in enumerate(req.block_hashes):
                hit_tier, _ = self.cache.read(
                    bh, req.block_size, depth, req.timestamp, req.session_id
                )
                if not hit_tier:
                    self.cache.write(
                        bh, req.block_size, depth, req.timestamp, req.session_id
                    )
            self.cache.tick(req.timestamp)


# ======================================================================
# Helpers
# ======================================================================


def _seed_session_patterns(policy: SessionAwarePrefetch, requests: List[Request]) -> None:
    """Pre-feed all session block sequences so the policy can learn them."""
    from collections import defaultdict
    sessions: dict = defaultdict(list)
    for req in requests:
        sessions[req.session_id].append(req.block_hashes)
    for sid, seqs in sessions.items():
        for seq in seqs[:-1]:   # Feed all but the last turn as training
            policy.record_sequence(sid, seq)


def _seed_belady(policy: BeladyOracle, requests: List[Request]) -> None:
    """Populate future_access_map from the full trace."""
    from collections import defaultdict
    future: dict = defaultdict(list)
    for req in requests:
        for bh in req.block_hashes:
            future[bh].append(req.timestamp)
    # Sort each list and store in policy
    policy._future = {k: sorted(v) for k, v in future.items()}
