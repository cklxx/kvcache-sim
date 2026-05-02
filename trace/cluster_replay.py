"""
ClusterReplayer — drives a ClusterRouter through a request trace.

Replays requests through the cluster router, routing each request to
the best GPU based on session affinity and prefix match.
"""
from __future__ import annotations

from typing import List

from sim.cluster import Cluster, ClusterRouter
from sim.metrics import Metrics
from sim.policies import BeladyOracle, SessionAwarePrefetch
from trace.generator import Request


class ClusterReplayer:
    def __init__(
        self,
        cluster: Cluster,
        warmup_count: int = 0,
        verbose: bool = False,
    ) -> None:
        self.cluster = cluster
        self.router = ClusterRouter(cluster)
        self.warmup_count = warmup_count
        self.verbose = verbose

    def run(self, requests: List[Request]) -> Metrics:
        warmup_count = min(max(self.warmup_count, 0), len(requests))

        # Seed only explicit offline oracles. Session-aware prefetchers learn
        # online after a GPU has processed each request.
        _seed_belady_policies(
            [gpu.eviction for gpu in self.cluster.all_gpus
             if isinstance(gpu.eviction, BeladyOracle)],
            requests,
        )

        self.cluster.reset_all()

        for i, req in enumerate(requests):
            gpu = self.router.route(req.block_hashes, req.session_id)
            gpu.process_request(
                req.block_hashes,
                req.block_size,
                req.session_id,
                req.timestamp,
            )

            if isinstance(gpu.prefetch, SessionAwarePrefetch):
                gpu.prefetch.record_sequence(req.session_id, req.block_hashes)

            if i == warmup_count - 1:
                self.cluster.reset_all()

            if self.verbose and i >= warmup_count and i % 2000 == 0:
                m = self.cluster.aggregate_metrics()
                print(
                    f"    [{i}/{len(requests)}] hit={m.hit_rate:.2%} "
                    f"evict={m.evictions} eic_xgpu={self.cluster.total_cross_gpu_eic_hits}"
                )

        return self.cluster.aggregate_metrics()


# ── helpers ───────────────────────────────────────────────────────────


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
    policy._future = _build_belady_future(requests)
