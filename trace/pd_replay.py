"""
PDReplayer — drives the PD pipeline through a request trace.

Converts base Requests to PDRequests and runs them through the
PDOrchestrator (prefill → transfer → decode).  Models concurrency
via ``earliest_available_time`` per prefill node.
"""
from __future__ import annotations

import heapq
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
        self.prefill_router = PrefillRouter(
            pd_cluster.prefill_nodes,
            seed=pd_cluster.routing_seed,
        )
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
        self.orchestrator.reset()
        metrics = PDMetrics()
        prefill_events = []
        decode_events = []

        def record_completed(results) -> None:
            for result in results:
                if result.request_index >= self.warmup_count:
                    metrics.record_request(result)

        def flush_until(cutoff_time: float) -> None:
            while prefill_events or decode_events:
                next_prefill = (
                    prefill_events[0][0] if prefill_events else float("inf")
                )
                next_decode = (
                    decode_events[0][0] if decode_events else float("inf")
                )
                event_time = min(next_prefill, next_decode)
                if event_time > cutoff_time:
                    break

                record_completed(self.orchestrator.advance_decode_nodes(event_time))

                while prefill_events and prefill_events[0][0] == event_time:
                    _, _, event = heapq.heappop(prefill_events)
                    pending, completed = self.orchestrator.start_decode_transfer(event)
                    record_completed(completed)
                    heapq.heappush(
                        decode_events,
                        (pending.decode_ready_time, pending.request_index, pending),
                    )

                while decode_events and decode_events[0][0] == event_time:
                    _, _, pending = heapq.heappop(decode_events)
                    record_completed(self.orchestrator.admit_decode(pending))

        for i, req in enumerate(requests):
            flush_until(req.timestamp)
            pd_req = _to_pd_request(req, self.cluster.pd_config.max_output_tokens)
            event = self.orchestrator.prepare_prefill(
                pd_req,
                req.timestamp,
                request_index=i,
            )
            heapq.heappush(
                prefill_events,
                (event.prefill_done_time, i, event),
            )

            if self.verbose and i >= self.warmup_count and i % 1000 == 0:
                print(
                    f"    [{i}/{len(requests)}] "
                    f"TTFT_p50={metrics.ttft_p50:.1f}ms "
                    f"prefix_hit={metrics.prefix_cache_hit_rate:.1%} "
                    f"reqs={metrics.total_requests}"
                )

        while prefill_events or decode_events:
            next_prefill = prefill_events[0][0] if prefill_events else float("inf")
            next_decode = decode_events[0][0] if decode_events else float("inf")
            flush_until(min(next_prefill, next_decode))

        record_completed(self.orchestrator.drain_decode())
        metrics.prefill_cache = self.cluster.aggregate_prefill_metrics()
        metrics.decode_cache = self.cluster.aggregate_decode_metrics()
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
        max_output_tokens=req.output_tokens if req.output_tokens > 0 else max_output_tokens,
    )
