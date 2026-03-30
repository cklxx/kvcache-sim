"""
PDReplayer — drives the PD pipeline through a request trace.

Converts base Requests to PDRequests and runs them through the
PDOrchestrator (prefill → transfer → decode).  Models concurrency
via ``earliest_available_time`` per prefill node.
"""
from __future__ import annotations

from typing import List

from sim.kv_transfer import KVTransferModel
from sim.pd_cluster import PDCluster, PDConfig
from sim.pd_metrics import PDMetrics
from sim.pd_nodes import ComputeConfig, PDRequest
from sim.pd_router import DecodeRouter, PDOrchestrator, PrefillRouter
from trace.generator import Request


class PDReplayer:
    """Replay a request trace through a PD-separated cluster."""

    def __init__(
        self,
        pd_cluster: PDCluster,
        warmup_count: int = 0,
        verbose: bool = False,
    ) -> None:
        self.cluster = pd_cluster
        self.warmup_count = warmup_count
        self.verbose = verbose

        # Build the orchestration pipeline
        self.prefill_router = PrefillRouter(pd_cluster.prefill_nodes)
        self.decode_router = DecodeRouter(pd_cluster.decode_nodes)
        self.transfer_model = KVTransferModel(
            pd_cluster.pd_config.transfer,
            pd_cluster.network,
        )
        self.orchestrator = PDOrchestrator(
            prefill_router=self.prefill_router,
            decode_router=self.decode_router,
            transfer_model=self.transfer_model,
            compute_cfg=pd_cluster.pd_config.compute,
            max_output_tokens=pd_cluster.pd_config.max_output_tokens,
        )

    def run(self, requests: List[Request]) -> PDMetrics:
        """
        Replay all requests through the PD pipeline.

        Requests before ``warmup_count`` populate caches but metrics
        are discarded.
        """
        self.cluster.reset_all()
        metrics = PDMetrics()

        for i, req in enumerate(requests):
            pd_req = _to_pd_request(req, self.cluster.pd_config.max_output_tokens)
            result = self.orchestrator.process_request(pd_req, req.timestamp)

            if i >= self.warmup_count:
                metrics.record_request(result)

            if i == self.warmup_count - 1 and self.warmup_count > 0:
                # Reset metrics after warmup
                metrics = PDMetrics()

            if self.verbose and i >= self.warmup_count and i % 1000 == 0:
                print(
                    f"    [{i}/{len(requests)}] "
                    f"TTFT_p50={metrics.ttft_p50:.1f}ms "
                    f"prefix_hit={metrics.prefix_cache_hit_rate:.1%} "
                    f"reqs={metrics.total_requests}"
                )

        return metrics


def _to_pd_request(req: Request, max_output_tokens: int = 128) -> PDRequest:
    """Convert a base Request to a PDRequest."""
    return PDRequest(
        session_id=req.session_id,
        turn_id=req.turn_id,
        timestamp=req.timestamp,
        block_hashes=req.block_hashes,
        block_size=req.block_size,
        prompt_tokens=req.prompt_tokens if req.prompt_tokens > 0 else len(req.block_hashes) * 32,
        max_output_tokens=max_output_tokens,
    )
