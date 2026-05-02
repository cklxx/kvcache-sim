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
        warmup_count = min(max(self.warmup_count, 0), len(requests))

        # Seed only explicit offline oracles. SessionAwarePrefetch learns
        # online below after each request has completed.
        _seed_belady_policies(
            [w.cache.eviction for w in self.router.workers
             if isinstance(w.cache.eviction, BeladyOracle)],
            requests,
        )

        # Reset all worker metrics
        for w in self.router.workers:
            w.cache.reset_metrics()

        for i, req in enumerate(requests):
            is_warmup = i < warmup_count

            # Route to best worker
            worker = self.router.route(req.block_hashes)
            worker.process(
                req.block_hashes,
                req.block_size,
                req.session_id,
                req.timestamp,
            )

            prefetch = worker.cache.prefetch
            if isinstance(prefetch, SessionAwarePrefetch):
                prefetch.record_sequence(req.session_id, req.block_hashes)

            if is_warmup and i == warmup_count - 1:
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


def _build_belady_future(requests: List[Request]) -> dict:
    from collections import defaultdict
    future: dict = defaultdict(list)
    for req in requests:
        for bh in req.block_hashes:
            future[bh].append(req.timestamp)
    return {k: sorted(v) for k, v in future.items()}


def _seed_belady_policies(policies: List[BeladyOracle], requests: List[Request]) -> None:
    if not policies:
        return

    shared_future = next(
        (p._future for p in policies if getattr(p, "_future", None)),
        None,
    )
    if shared_future is None:
        shared_future = _build_belady_future(requests)

    for policy in policies:
        if not getattr(policy, "_future", None):
            policy._future = shared_future


def _seed_belady(policy: BeladyOracle, requests: List[Request]) -> None:
    """Populate future_access_map from the full trace."""
    policy._future = _build_belady_future(requests)
