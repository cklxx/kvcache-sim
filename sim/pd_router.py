"""
PD Router — separate routing for Prefill and Decode node pools.

PrefillRouter
  Maximise prefix cache hit in RadixTree, penalise queue depth.

DecodeRouter
  Prefer same-rack as prefill node (fast KV transfer), capacity-aware.

PDOrchestrator
  Coordinates the full PD pipeline:
  prefill routing → compute → KV transfer → decode → metrics
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .kv_transfer import KVTransferModel
from .pd_nodes import (
    ComputeConfig,
    DecodeNode,
    KVTransferInfo,
    PDRequest,
    PrefillNode,
    PrefillResult,
)


# ======================================================================
# Prefill Router
# ======================================================================


class PrefillRouter:
    """
    Routes requests to prefill nodes.

    Scoring:
      prefix_match × 1.0  −  queue_depth × 0.1  +  eic_match × 0.3
    Session affinity for multi-turn conversations.
    """

    def __init__(self, prefill_nodes: List[PrefillNode]) -> None:
        self.prefill_nodes = prefill_nodes
        self._session_affinity: Dict[str, int] = {}
        self._rr: int = 0

    def route(self, request: PDRequest, current_time: float) -> PrefillNode:
        # Soft session affinity: prefer previous node but re-evaluate if busy
        affinity_node: Optional[PrefillNode] = None
        if request.session_id in self._session_affinity:
            idx = self._session_affinity[request.session_id]
            if idx < len(self.prefill_nodes):
                affinity_node = self.prefill_nodes[idx]
                # Use affinity node if it's not backed up
                if affinity_node.estimated_queue_time(current_time) < 50.0:
                    return affinity_node

        if not request.block_hashes:
            return self._next_rr(request.session_id)

        # Score candidates (sample for scalability)
        candidates = self.prefill_nodes
        if len(candidates) > 16:
            candidates = random.sample(candidates, 16)
            # Always include affinity node if it exists
            if affinity_node and affinity_node not in candidates:
                candidates[0] = affinity_node

        best: Optional[PrefillNode] = None
        best_score = -float("inf")
        for node in candidates:
            cached = node.cached_hashes()
            prefix_score = sum(1.0 for h in request.block_hashes if h in cached)
            eic_score = sum(
                0.3 for h in request.block_hashes if node.eic.contains(h)
            )
            # Queue penalty based on actual wait time (ms → points conversion)
            queue_wait = node.estimated_queue_time(current_time)
            queue_penalty = queue_wait * 0.05  # 20ms queue ≈ 1 block advantage
            score = prefix_score + eic_score - queue_penalty
            if score > best_score:
                best_score = score
                best = node

        if best is not None and best_score > 0:
            idx = self.prefill_nodes.index(best)
            self._session_affinity[request.session_id] = idx
            return best

        return self._next_rr(request.session_id)

    def _next_rr(self, session_id: str) -> PrefillNode:
        idx = self._rr % len(self.prefill_nodes)
        self._session_affinity[session_id] = idx
        self._rr += 1
        return self.prefill_nodes[idx]


# ======================================================================
# Decode Router
# ======================================================================


class DecodeRouter:
    """
    Routes decode tasks to decode nodes.

    Scoring:
      −active_sequences × 1.0  +  same_rack_bonus × 5.0
    Prefers co-located nodes for fast KV transfer.
    """

    def __init__(self, decode_nodes: List[DecodeNode]) -> None:
        self.decode_nodes = decode_nodes

    def route(
        self,
        prefill_result: PrefillResult,
    ) -> DecodeNode:
        prefill_rack = prefill_result.prefill_rack_id

        best: Optional[DecodeNode] = None
        best_score = -float("inf")
        for node in self.decode_nodes:
            if not node.has_capacity():
                continue
            score = -node.active_count * 1.0
            if node.rack_id == prefill_rack:
                score += 5.0
            # Bonus if this node already has KV from a previous turn
            cached = node.cached_hashes()
            kv_score = sum(
                0.5 for h in prefill_result.block_hashes if h in cached
            )
            score += kv_score
            if score > best_score:
                best_score = score
                best = node

        if best is not None:
            return best

        # Fallback: least-loaded node (ignore capacity limit)
        return min(self.decode_nodes, key=lambda n: n.active_count)


# ======================================================================
# PD Orchestrator
# ======================================================================


class PDOrchestrator:
    """
    Coordinates the full Prefill → Transfer → Decode pipeline.

    For each request:
    1. PrefillRouter selects a prefill node
    2. PrefillNode.prefill() computes KV (with radix tree cache)
    3. DecodeRouter selects a decode node (rack-aware)
    4. KVTransferModel computes transfer cost
    5. DecodeNode receives KV and runs decode steps
    6. Returns full timing breakdown
    """

    def __init__(
        self,
        prefill_router: PrefillRouter,
        decode_router: DecodeRouter,
        transfer_model: KVTransferModel,
        compute_cfg: ComputeConfig,
        max_output_tokens: int = 128,
    ) -> None:
        self.prefill_router = prefill_router
        self.decode_router = decode_router
        self.transfer_model = transfer_model
        self.compute_cfg = compute_cfg
        self.max_output_tokens = max_output_tokens

    def process_request(
        self,
        request: PDRequest,
        current_time: float,
    ) -> "PDStepResult":
        """Full PD pipeline for one request."""
        max_out = request.max_output_tokens or self.max_output_tokens

        # ── 1. Route to prefill node ──
        p_node = self.prefill_router.route(request, current_time)
        p_node.queue_depth += 1

        # ── 2. Prefill ──
        prefill_result = p_node.prefill(request, current_time)
        queue_wait_ms = prefill_result.queue_wait_ms
        prefill_done_time = current_time + queue_wait_ms + prefill_result.compute_time_ms
        p_node.queue_depth -= 1

        # ── 3. Route to decode node ──
        d_node = self.decode_router.route(prefill_result)
        same_rack = p_node.rack_id == d_node.rack_id

        # ── 4. KV Transfer ──
        transfer_info = KVTransferInfo(
            source_node_id=p_node.gpu_id,
            dest_node_id=d_node.gpu_id,
            source_rack_id=p_node.rack_id,
            dest_rack_id=d_node.rack_id,
            session_id=request.session_id,
            block_hashes=prefill_result.block_hashes,
            total_bytes=prefill_result.kv_bytes,
        )
        full_transfer_ms = self.transfer_model.transfer_latency_ms(
            len(prefill_result.block_hashes), request.block_size, same_rack,
            src_gpu=p_node.gpu_id, dst_gpu=d_node.gpu_id,
        )
        ttft_transfer_ms = self.transfer_model.effective_ttft_transfer_ms(
            len(prefill_result.block_hashes), request.block_size, same_rack,
            src_gpu=p_node.gpu_id, dst_gpu=d_node.gpu_id,
        )

        # ── 5. Decode receives KV ──
        decode_start_time = prefill_done_time + ttft_transfer_ms
        d_node.receive_kv(transfer_info, request.block_size, decode_start_time)

        # ── 6. Decode steps (continuous batching) ──
        # First token: step time depends on concurrent active sequences
        first_decode_ms = d_node.decode_step(request.session_id, decode_start_time)
        # Remaining tokens: same decode node, concurrent sequence count may vary
        # Use current active count as representative
        step_ms = self.compute_cfg.decode_step_ms(d_node.active_count)
        remaining_decode_ms = (max_out - 1) * step_ms
        decode_total_ms = first_decode_ms + remaining_decode_ms

        # Finish sequence and release refs
        d_node.finish_sequence(request.session_id)
        p_node.release_sequence(prefill_result.block_hashes)

        # ── 7. Compute TTFT and E2E ──
        ttft_ms = queue_wait_ms + prefill_result.compute_time_ms + ttft_transfer_ms + first_decode_ms
        e2e_ms = ttft_ms + remaining_decode_ms

        return PDStepResult(
            session_id=request.session_id,
            prefill_node_id=p_node.gpu_id,
            decode_node_id=d_node.gpu_id,
            same_rack=same_rack,
            prefill_queue_wait_ms=queue_wait_ms,
            prefill_compute_ms=prefill_result.compute_time_ms,
            kv_transfer_ms=full_transfer_ms,
            ttft_transfer_ms=ttft_transfer_ms,
            decode_first_token_ms=first_decode_ms,
            decode_total_ms=decode_total_ms,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            prefix_hit_blocks=prefill_result.cached_blocks,
            new_computed_blocks=prefill_result.new_blocks,
            kv_bytes_transferred=prefill_result.kv_bytes,
            output_tokens=max_out,
        )


@dataclass
class PDStepResult:
    """Per-request timing breakdown from the PD pipeline."""

    session_id: str
    prefill_node_id: int
    decode_node_id: int
    same_rack: bool

    # Timing breakdown (ms)
    prefill_queue_wait_ms: float
    prefill_compute_ms: float
    kv_transfer_ms: float
    ttft_transfer_ms: float
    decode_first_token_ms: float
    decode_total_ms: float
    ttft_ms: float
    e2e_ms: float

    # Cache / transfer stats
    prefix_hit_blocks: int
    new_computed_blocks: int
    kv_bytes_transferred: int
    output_tokens: int
