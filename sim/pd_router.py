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
    DecodeSequence,
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

    def __init__(self, prefill_nodes: List[PrefillNode], seed: int = 0) -> None:
        self.prefill_nodes = prefill_nodes
        self._session_affinity: Dict[str, int] = {}
        self._rr: int = 0
        self._rng = random.Random(seed)

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
            candidates = self._rng.sample(candidates, 16)
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


@dataclass
class PDPrefillEvent:
    """A request whose prefill phase has been scheduled/computed."""

    request_index: int
    request: PDRequest
    sequence_id: str
    prefill_node: PrefillNode
    prefill_result: PrefillResult
    arrival_time: float
    prefill_done_time: float
    max_output_tokens: int


@dataclass
class PDPendingDecode:
    """A prefilled request waiting for decode admission."""

    request_index: int
    request: PDRequest
    sequence_id: str
    prefill_node: PrefillNode
    decode_node: DecodeNode
    prefill_result: PrefillResult
    transfer_info: KVTransferInfo
    same_rack: bool
    prefill_done_time: float
    decode_ready_time: float
    kv_transfer_ms: float
    ttft_transfer_ms: float
    max_output_tokens: int


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
        self._active_decodes: Dict[str, PDPendingDecode] = {}

    def reset(self) -> None:
        """Clear orchestrator-owned decode bookkeeping."""
        self._active_decodes.clear()

    def _sequence_id(self, request: PDRequest) -> str:
        return f"{request.session_id}:{request.turn_id}"

    def _event_sequence_id(self, request: PDRequest, request_index: int) -> str:
        base = self._sequence_id(request)
        if request_index >= 0:
            return f"{base}:{request_index}"
        return base

    def prepare_prefill(
        self,
        request: PDRequest,
        current_time: float,
        request_index: int = -1,
    ) -> PDPrefillEvent:
        """Route and execute the prefill phase, returning a timed event."""
        max_out = request.max_output_tokens or self.max_output_tokens

        p_node = self.prefill_router.route(request, current_time)
        p_node.queue_depth += 1

        prefill_result = p_node.prefill(request, current_time)
        prefill_done_time = (
            current_time
            + prefill_result.queue_wait_ms
            + prefill_result.compute_time_ms
        )
        p_node.queue_depth -= 1

        return PDPrefillEvent(
            request_index=request_index,
            request=request,
            sequence_id=self._event_sequence_id(request, request_index),
            prefill_node=p_node,
            prefill_result=prefill_result,
            arrival_time=current_time,
            prefill_done_time=prefill_done_time,
            max_output_tokens=max_out,
        )

    def start_decode_transfer(
        self,
        event: PDPrefillEvent,
    ) -> Tuple[PDPendingDecode, List["PDStepResult"]]:
        """
        Route to a decode node at prefill completion time and schedule KV transfer.

        Decode nodes are advanced to ``prefill_done_time`` before routing so
        the router sees active sequences that are valid on the event timeline.
        """
        completed = self.advance_decode_nodes(event.prefill_done_time)

        p_node = event.prefill_node
        prefill_result = event.prefill_result
        request = event.request

        d_node = self.decode_router.route(prefill_result)
        same_rack = p_node.rack_id == d_node.rack_id

        transfer_info = KVTransferInfo(
            source_node_id=p_node.gpu_id,
            dest_node_id=d_node.gpu_id,
            source_rack_id=p_node.rack_id,
            dest_rack_id=d_node.rack_id,
            session_id=request.session_id,
            block_hashes=prefill_result.block_hashes,
            total_bytes=prefill_result.kv_bytes,
            sequence_id=event.sequence_id,
        )
        transfer_timing = self.transfer_model.transfer_timing_ms(
            len(prefill_result.block_hashes),
            request.block_size,
            same_rack,
            src_gpu=p_node.gpu_id,
            dst_gpu=d_node.gpu_id,
            start_time_ms=event.prefill_done_time,
        )
        full_transfer_ms = transfer_timing.full_ms
        ttft_transfer_ms = transfer_timing.ttft_ms

        return (
            PDPendingDecode(
                request_index=event.request_index,
                request=request,
                sequence_id=event.sequence_id,
                prefill_node=p_node,
                decode_node=d_node,
                prefill_result=prefill_result,
                transfer_info=transfer_info,
                same_rack=same_rack,
                prefill_done_time=event.prefill_done_time,
                decode_ready_time=event.prefill_done_time + ttft_transfer_ms,
                kv_transfer_ms=full_transfer_ms,
                ttft_transfer_ms=ttft_transfer_ms,
                max_output_tokens=event.max_output_tokens,
            ),
            completed,
        )

    def admit_decode(self, pending: PDPendingDecode) -> List["PDStepResult"]:
        """Admit a decode-ready request and return any completed results."""
        self._active_decodes[pending.sequence_id] = pending
        completed_sequences = pending.decode_node.admit_sequence(
            transfer=pending.transfer_info,
            block_size=pending.request.block_size,
            ready_time=pending.decode_ready_time,
            max_tokens=pending.max_output_tokens,
        )
        return self._complete_sequences(completed_sequences)

    def advance_decode_nodes(self, current_time: float) -> List["PDStepResult"]:
        """Advance every decode node to ``current_time`` and collect completions."""
        completed_sequences: List[DecodeSequence] = []
        for node in self.decode_router.decode_nodes:
            completed_sequences.extend(node.advance_time(current_time))
        return self._complete_sequences(completed_sequences)

    def drain_decode(self) -> List["PDStepResult"]:
        """Run all decode nodes until their active sequences complete."""
        completed_sequences: List[DecodeSequence] = []
        for node in self.decode_router.decode_nodes:
            completed_sequences.extend(node.drain())
        return self._complete_sequences(completed_sequences)

    def _complete_sequences(
        self,
        sequences: List[DecodeSequence],
    ) -> List["PDStepResult"]:
        results: List[PDStepResult] = []
        for seq in sequences:
            sequence_id = seq.sequence_id or seq.session_id
            pending = self._active_decodes.pop(sequence_id, None)
            if pending is None:
                continue
            pending.prefill_node.release_sequence(
                pending.prefill_result.block_hashes
            )
            results.append(self._result_from_completion(pending, seq))
        return results

    def _result_from_completion(
        self,
        pending: PDPendingDecode,
        seq: DecodeSequence,
    ) -> "PDStepResult":
        first_token_time = (
            seq.first_token_time
            if seq.first_token_time is not None
            else seq.finish_time
            if seq.finish_time is not None
            else seq.start_time
        )
        decode_finish_time = (
            seq.finish_time if seq.finish_time is not None else first_token_time
        )
        transfer_done_time = pending.prefill_done_time + pending.kv_transfer_ms
        request_finish_time = max(decode_finish_time, transfer_done_time)

        prefill_result = pending.prefill_result
        return PDStepResult(
            session_id=pending.request.session_id,
            prefill_node_id=pending.prefill_node.gpu_id,
            decode_node_id=pending.decode_node.gpu_id,
            same_rack=pending.same_rack,
            prefill_queue_wait_ms=prefill_result.queue_wait_ms,
            prefill_compute_ms=prefill_result.compute_time_ms,
            kv_transfer_ms=pending.kv_transfer_ms,
            ttft_transfer_ms=pending.ttft_transfer_ms,
            decode_first_token_ms=max(0.0, first_token_time - seq.start_time),
            decode_total_ms=max(0.0, decode_finish_time - seq.start_time),
            ttft_ms=max(0.0, first_token_time - pending.request.timestamp),
            e2e_ms=max(0.0, request_finish_time - pending.request.timestamp),
            prefix_hit_blocks=prefill_result.cached_blocks,
            new_computed_blocks=prefill_result.new_blocks,
            kv_bytes_transferred=prefill_result.kv_bytes,
            output_tokens=pending.max_output_tokens,
            request_index=pending.request_index,
            decode_queue_wait_ms=max(0.0, seq.start_time - pending.decode_ready_time),
            decode_start_time=seq.start_time,
            decode_first_token_time=first_token_time,
            decode_finish_time=decode_finish_time,
            decode_ready_time=pending.decode_ready_time,
        )

    def process_request(
        self,
        request: PDRequest,
        current_time: float,
    ) -> "PDStepResult":
        """
        Full PD pipeline for one request.

        The replay path uses the event-oriented methods above.  This helper is
        kept for compatibility and drains decode work until this request has a
        completed result.
        """
        event = self.prepare_prefill(request, current_time)
        pending, completed = self.start_decode_transfer(event)
        completed.extend(self.admit_decode(pending))
        completed.extend(self.drain_decode())

        for result in completed:
            if result.session_id == request.session_id:
                return result
        raise RuntimeError(f"PD request did not complete: {request.session_id}")


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

    # Event-driven decode timeline (optional for existing metrics)
    request_index: int = -1
    decode_queue_wait_ms: float = 0.0
    decode_start_time: float = 0.0
    decode_first_token_time: float = 0.0
    decode_finish_time: float = 0.0
    decode_ready_time: float = 0.0
