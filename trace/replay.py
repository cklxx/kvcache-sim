"""
TraceReplayer — drives a Router (with Workers) through a request trace.

Usage
-----
  replayer = TraceReplayer(router, warmup_count=200)
  metrics  = replayer.run(requests)
"""
from __future__ import annotations

from typing import List

from sim.metrics import Metrics
from sim.policies import BeladyOracle, SessionAwarePrefetch
from sim.router import Router
from trace.generator import Request


class TraceReplayer:
    def __init__(
        self,
        router: Router,
        warmup_count: int = 0,
        verbose: bool = False,
    ) -> None:
        self.router = router
        self.warmup_count = warmup_count
        self.verbose = verbose

    def run(self, requests: List[Request]) -> Metrics:
        """
        Replay all requests through the router.
        Requests before warmup_count populate caches but metrics are discarded.
        """
        # Seed policies on all workers
        for w in self.router.workers:
            prefetch = w.cache.prefetch
            if isinstance(prefetch, SessionAwarePrefetch):
                _seed_session_patterns(prefetch, requests)
            if isinstance(w.cache.eviction, BeladyOracle):
                _seed_belady(w.cache.eviction, requests)

        # Reset all worker metrics
        for w in self.router.workers:
            w.cache.reset_metrics()

        for i, req in enumerate(requests):
            is_warmup = i < self.warmup_count

            # Route to best worker
            worker = self.router.route(req.block_hashes)
            worker.process(
                req.block_hashes,
                req.block_size,
                req.session_id,
                req.timestamp,
            )

            if is_warmup and i == self.warmup_count - 1:
                # Reset metrics after warmup completes
                for w in self.router.workers:
                    w.cache.reset_metrics()

            if self.verbose and not is_warmup and i % 500 == 0:
                m = self.router.aggregate_metrics()
                print(f"  [{i}/{len(requests)}] hit_rate={m.hit_rate:.2%} "
                      f"evictions={m.evictions}")

        return self.router.aggregate_metrics()


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
        for seq in seqs[:-1]:
            policy.record_sequence(sid, seq)


def _seed_belady(policy: BeladyOracle, requests: List[Request]) -> None:
    """Populate future_access_map from the full trace."""
    from collections import defaultdict
    future: dict = defaultdict(list)
    for req in requests:
        for bh in req.block_hashes:
            future[bh].append(req.timestamp)
    policy._future = {k: sorted(v) for k, v in future.items()}
