"""
PD Nodes — Prefill and Decode specialised GPU nodes.

PrefillNode
  - Uses RadixTree for KV block management (prefix sharing + ref counting)
  - Session-aware prefetch: after prefill, predict and pre-load next-turn blocks
  - Continuous batching: batch concurrent prefill requests for higher GPU util
  - Models prefill compute time: O(new_tokens) scaled by batch efficiency

DecodeNode
  - Receives KV blocks from prefill nodes
  - Continuous batching: decode all active sequences in one step
  - Models decode time: memory-bandwidth bound with marginal KV read overhead
  - Step time scales sub-linearly with concurrent sequences

Compute calibration (H100-class GPU, 70B profile from config.yaml)
-------------------------------------
  Prefill  : ~0.35 ms/token  (2 × 140 GB / 800 TFLOPS)
  Decode   : ~83.6 ms/token  (2 × 140 GB / 3350 GB/s)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cluster import EICPool, GPUNode
from .metrics import Metrics
from .network import NetworkModel
from .policies import (
    EvictionPolicy,
    LRUPolicy,
    NoPrefetch,
    PrefetchPolicy,
    SessionAwarePrefetch,
)
from .radix_tree import RadixTree
from .storage import KVBlock, StorageTier


# ======================================================================
# Compute Configuration
# ======================================================================


@dataclass
class ComputeConfig:
    """Hardware-calibrated compute model for prefill and decode phases."""

    prefill_tflops: float = 800.0        # H100 FP16 effective
    decode_memory_bw_gbps: float = 3200  # HBM bandwidth
    model_params_b: float = 70.0         # 70B parameter model
    kv_bytes_per_token: int = 327680     # 320 KiB per token for Llama-3-70B GQA
    tokens_per_block: int = 16
    overhead_factor: float = 1.0         # scheduling + framework overhead

    # Continuous batching parameters
    prefill_batch_efficiency: float = 0.85  # GPU utilisation at batch size > 1
    decode_kv_overhead_factor: float = 0.02  # marginal cost per active sequence

    @property
    def model_params_bytes(self) -> float:
        return self.model_params_b * 1e9 * 2  # FP16 = 2 bytes

    @property
    def prefill_ms_per_token(self) -> float:
        """Time to compute one token during prefill (ms), single request."""
        flops_per_token = 2 * self.model_params_bytes
        seconds = flops_per_token / (self.prefill_tflops * 1e12)
        return seconds * 1000.0 * self.overhead_factor

    @property
    def decode_ms_per_token(self) -> float:
        """Base time to generate one token during decode (ms), 1 sequence."""
        bytes_per_step = 2 * self.model_params_bytes
        bw_bytes_per_sec = self.decode_memory_bw_gbps * 1e9
        seconds = bytes_per_step / bw_bytes_per_sec
        return seconds * 1000.0 * self.overhead_factor

    def decode_step_ms(self, active_sequences: int) -> float:
        """
        Decode step time with continuous batching.

        All active sequences are decoded in one GPU step.  The model
        weight read dominates; KV cache reads add marginal overhead
        per sequence (sub-linear scaling).

        Returns per-step time (ms).  Each step advances ALL sequences
        by one token.
        """
        base = self.decode_ms_per_token
        # Marginal overhead: reading KV cache for each active sequence
        # Sub-linear because GPU can overlap reads
        overhead = base * self.decode_kv_overhead_factor * math.log2(max(2, active_sequences))
        return base + overhead

    def batched_prefill_ms(self, total_new_tokens: int, batch_size: int) -> float:
        """
        Prefill compute time for a batch of requests.

        With batching, GPU processes all requests' tokens simultaneously.
        Total FLOPs = batch_tokens × 2 × params.  GPU efficiency improves
        with larger batches (better SM utilisation).

        Returns total batch compute time (ms).  All requests in the
        batch finish together.
        """
        if total_new_tokens <= 0:
            return 0.0
        base_ms = total_new_tokens * self.prefill_ms_per_token
        if batch_size > 1:
            # Batching improves GPU utilisation
            efficiency = self.prefill_batch_efficiency
            base_ms *= efficiency
        return base_ms

    @staticmethod
    def from_config(cfg: dict) -> "ComputeConfig":
        cc = cfg.get("pd_separation", {}).get("compute", {})
        return ComputeConfig(
            prefill_tflops=cc.get("prefill_tflops", 800.0),
            decode_memory_bw_gbps=cc.get("decode_memory_bw_gbps", 3200.0),
            model_params_b=cc.get("model_params_b", 7.0),
            kv_bytes_per_token=cc.get("kv_bytes_per_token", 64),
            tokens_per_block=cc.get("tokens_per_block", 32),
            overhead_factor=cc.get("overhead_factor", 1.0),
            prefill_batch_efficiency=cc.get("prefill_batch_efficiency", 0.85),
            decode_kv_overhead_factor=cc.get("decode_kv_overhead_factor", 0.02),
        )


# ======================================================================
# Data classes for PD pipeline
# ======================================================================


@dataclass
class PDRequest:
    """Request with PD-specific fields."""

    session_id: str
    turn_id: int
    timestamp: float
    block_hashes: List[str]
    block_size: int = 4096
    prompt_tokens: int = 0
    max_output_tokens: int = 128


@dataclass
class PrefillResult:
    """Output of prefill phase."""

    prefill_node_id: int
    prefill_rack_id: int
    session_id: str
    block_hashes: List[str]
    cached_blocks: int
    new_blocks: int
    compute_time_ms: float
    queue_wait_ms: float
    kv_bytes: int
    prefetched_blocks: int = 0


@dataclass
class KVTransferInfo:
    """Describes a KV cache transfer between nodes."""

    source_node_id: int
    dest_node_id: int
    source_rack_id: int
    dest_rack_id: int
    session_id: str
    block_hashes: List[str]
    total_bytes: int
    sequence_id: str = ""


@dataclass
class DecodeSequence:
    """Tracks an active decode sequence on a decode node."""

    session_id: str
    kv_block_hashes: List[str]
    sequence_id: str = ""
    tokens_generated: int = 0
    ready_time: float = 0.0
    start_time: float = 0.0
    max_tokens: int = 128
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None


# ======================================================================
# Prefill Node
# ======================================================================


class PrefillNode:
    """
    GPU node specialised for prefill (prompt processing).

    Features:
    - RadixTree for prefix-sharing block management
    - Session-aware prefetch: predicts next-turn blocks and pre-loads them
    - Continuous batching: processes multiple requests in one GPU batch
    """

    def __init__(
        self,
        gpu_id: int,
        rack_id: int,
        hbm: StorageTier,
        eic: EICPool,
        eviction: EvictionPolicy,
        prefetch: PrefetchPolicy,
        network: NetworkModel,
        compute_cfg: ComputeConfig,
        max_batch_tokens: int = 32768,
    ) -> None:
        self.gpu_id = gpu_id
        self.rack_id = rack_id
        self.hbm = hbm
        self.eic = eic
        self.eviction = eviction
        self.prefetch = prefetch
        self.network = network
        self.compute_cfg = compute_cfg
        self.max_batch_tokens = max_batch_tokens

        self.radix_tree = RadixTree(
            capacity_bytes=hbm.capacity_bytes,
        )
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])

        # Queue / batching model
        self.earliest_available_time: float = 0.0
        self.queue_depth: int = 0
        # Pending batch: requests waiting to be processed together
        self._pending_batch: List[Tuple[PDRequest, float]] = []  # (req, arrival)
        self._batch_window_ms: float = 2.0  # collect requests for this window

    def prefill(
        self,
        request: PDRequest,
        current_time: float,
    ) -> PrefillResult:
        """
        Execute prefill with RDMA-optimised EIC prefetch.

        Pipeline:
          1. RadixTree lookup (local HBM) → instant
          2. Batch probe EIC for ALL remaining hashes → 1 RTT (5μs)
          3. Async batch fetch EIC hits via RDMA → overlaps with GPU compute
          4. GPU computes KV for true misses → overlaps with RDMA fetch
          5. Insert new blocks + prefetch next-turn
          6. Effective time = max(rdma_fetch, gpu_compute)
        """
        block_hashes = request.block_hashes
        block_size = request.block_size

        # Queue wait: GPU busy until earliest_available_time
        effective_time = max(current_time, self.earliest_available_time)
        queue_wait = effective_time - current_time

        # ── 1. RadixTree lookup (local HBM, instant) ────────────
        match_depth, matched_nodes = self.radix_tree.lookup_prefix(block_hashes)
        hbm_hits = match_depth
        remaining_hashes = block_hashes[match_depth:]

        # ── 2. Batch probe EIC for ALL remaining hashes ─────────
        # One RDMA round-trip with all hashes, not per-block
        # Probe cost: 1 RTT = intra_rack_latency + rdma_base ≈ 8μs
        eic_hit_hashes: List[str] = []
        first_miss_idx = 0
        for bh in remaining_hashes:
            if self.eic.contains(bh):
                eic_hit_hashes.append(bh)
                first_miss_idx += 1
            else:
                break
        eic_miss_hashes = remaining_hashes[first_miss_idx:]
        probe_latency_ms = self.network.intra_rack_ms()  # single RTT

        # ── 3. Async batch fetch EIC hits (RDMA one-sided READ) ─
        # GPUDirect RDMA: write directly to GPU HBM, bypass CPU
        # Transfer time = total_bytes / rdma_bw
        eic_fetch_bytes = len(eic_hit_hashes) * block_size
        rdma_bw_bytes_per_ms = self.network.p2p_rdma_bw_gbps * 1e9 / 1000.0
        eic_fetch_ms = 0.0
        if eic_hit_hashes:
            eic_fetch_ms = (
                probe_latency_ms
                + eic_fetch_bytes / rdma_bw_bytes_per_ms
            )
            # Actually read from EIC and promote to local tree
            for idx, bh in enumerate(eic_hit_hashes):
                blk = self.eic.read(bh, self.gpu_id, effective_time)
                if blk is not None:
                    self.radix_tree.insert_suffix_after_prefix(
                        block_hashes,
                        match_depth + idx,
                        [bh],
                        block_size,
                        effective_time + eic_fetch_ms,
                    )
                    self.metrics.total_requests += 1
                    self.metrics.record_hit("EIC", eic_fetch_ms)

        eic_hits = len(eic_hit_hashes)
        total_cached = hbm_hits + eic_hits
        new_blocks = len(eic_miss_hashes)

        # Record cache metrics
        for _ in range(hbm_hits):
            self.metrics.total_requests += 1
            lat = self.hbm.transfer_latency_ms(block_size, is_read=True)
            self.metrics.record_hit("HBM", lat)
        for _ in range(new_blocks):
            self.metrics.total_requests += 1
            self.metrics.record_miss()

        # ── 4. GPU computes KV for true misses ──────────────────
        # This OVERLAPS with the RDMA fetch above
        new_tokens = new_blocks * self.compute_cfg.tokens_per_block
        batch_size = max(1, self.queue_depth)
        gpu_compute_ms = self.compute_cfg.batched_prefill_ms(new_tokens, batch_size)

        # ── Effective time = max(rdma_fetch, gpu_compute) ───────
        # RDMA and GPU operate in parallel (async DMA engine)
        compute_ms = max(eic_fetch_ms, gpu_compute_ms)

        # ── 5. Insert new blocks + async EIC backup ─────────────
        if eic_miss_hashes:
            self.radix_tree.insert_suffix_after_prefix(
                block_hashes,
                hbm_hits + eic_hits,
                eic_miss_hashes,
                block_size,
                effective_time + compute_ms,
            )
        self.radix_tree.acquire_sequence(block_hashes)

        # Async EIC backup for newly computed blocks
        for i, bh in enumerate(eic_miss_hashes):
            blk = KVBlock(
                block_hash=bh,
                size_bytes=block_size,
                prefix_depth=hbm_hits + eic_hits + i,
                last_access_time=effective_time + compute_ms,
                access_count=1,
            )
            if not self.eic.contains(bh):
                self.eic.write(blk, self.gpu_id, effective_time + compute_ms)

        # ── 6. Prefetch next-turn blocks from EIC ───────────────
        prefetched = 0
        self.prefetch.record_sequence(request.session_id, block_hashes)
        if block_hashes:
            candidates = self.prefetch.candidates(block_hashes[-1], request.session_id)
            for cand_hash in candidates:
                if not self.radix_tree.contains(cand_hash):
                    blk = self.eic.read(cand_hash, self.gpu_id, effective_time + compute_ms)
                    if blk is not None:
                        self.radix_tree.insert_sequence(
                            [cand_hash], block_size, effective_time + compute_ms
                        )
                        prefetched += 1
                        self.metrics.prefetches += 1

        # Update queue model
        self.earliest_available_time = effective_time + compute_ms

        kv_bytes = len(block_hashes) * block_size

        return PrefillResult(
            prefill_node_id=self.gpu_id,
            prefill_rack_id=self.rack_id,
            session_id=request.session_id,
            block_hashes=block_hashes,
            cached_blocks=total_cached,
            new_blocks=new_blocks,
            compute_time_ms=compute_ms,
            queue_wait_ms=queue_wait,
            kv_bytes=kv_bytes,
            prefetched_blocks=prefetched,
        )

    def release_sequence(self, block_hashes: List[str]) -> None:
        """Release ref counts after decode completes."""
        self.radix_tree.release_sequence(block_hashes)

    def cached_hashes(self) -> set:
        return self.radix_tree.cached_hashes()

    def estimated_queue_time(self, current_time: float) -> float:
        return max(0.0, self.earliest_available_time - current_time)

    def reset_metrics(self) -> None:
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])


# ======================================================================
# Decode Node
# ======================================================================


class DecodeNode:
    """
    GPU node specialised for decode (autoregressive generation).

    Features:
    - Receives KV blocks from prefill nodes
    - Continuous batching: all active sequences decoded in one GPU step
    - Step time scales sub-linearly with concurrent sequences
    """

    def __init__(
        self,
        gpu_id: int,
        rack_id: int,
        hbm: StorageTier,
        eic: EICPool,
        eviction: EvictionPolicy,
        network: NetworkModel,
        compute_cfg: ComputeConfig,
        max_concurrent_sequences: int = 64,
    ) -> None:
        self.gpu_id = gpu_id
        self.rack_id = rack_id
        self.hbm = hbm
        self.eic = eic
        self.eviction = eviction
        self.network = network
        self.compute_cfg = compute_cfg
        self.max_concurrent_sequences = max_concurrent_sequences

        self.radix_tree = RadixTree(
            capacity_bytes=hbm.capacity_bytes,
        )
        self.active_sequences: Dict[str, DecodeSequence] = {}
        self.timeline_time: float = 0.0
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])

    def _sequence_key(self, session_id: str) -> str:
        """Return the active-sequence key for legacy session-only callers."""
        if session_id in self.active_sequences:
            return session_id
        for key, seq in self.active_sequences.items():
            if seq.session_id == session_id:
                return key
        return session_id

    def advance_time(self, target_time: float) -> List[DecodeSequence]:
        """
        Advance decode work to ``target_time`` at decode-step boundaries.

        A request that arrives in the middle of an in-flight decode step
        cannot join that step.  Therefore this method only completes steps
        whose end time is <= ``target_time``; admission uses
        ``_advance_to_admission_boundary`` to roll through the crossing step
        for the selected node.
        """
        completed: List[DecodeSequence] = []
        if target_time < self.timeline_time:
            return completed

        while self.active_sequences:
            step_ms = self.compute_cfg.decode_step_ms(len(self.active_sequences))
            if self.timeline_time + step_ms > target_time:
                break
            completed.extend(self._run_one_decode_step())

        if not self.active_sequences and self.timeline_time < target_time:
            self.timeline_time = target_time
        return completed

    def _advance_to_admission_boundary(self, ready_time: float) -> List[DecodeSequence]:
        """Advance until a new sequence can join the next decode step."""
        completed = self.advance_time(ready_time)

        # If ready_time lands inside an in-flight step, finish that step first.
        if self.active_sequences and self.timeline_time < ready_time:
            completed.extend(self._run_one_decode_step())

        if not self.active_sequences and self.timeline_time < ready_time:
            self.timeline_time = ready_time
        return completed

    def _run_one_decode_step(self) -> List[DecodeSequence]:
        """Run one continuous-batching decode step for all active sequences."""
        if not self.active_sequences:
            return []

        active = len(self.active_sequences)
        step_ms = self.compute_cfg.decode_step_ms(active)
        step_end = self.timeline_time + step_ms
        completed: List[DecodeSequence] = []

        for key, seq in list(self.active_sequences.items()):
            seq.tokens_generated += 1
            if seq.tokens_generated == 1:
                seq.first_token_time = step_end
            if seq.tokens_generated >= seq.max_tokens:
                seq.finish_time = step_end
                completed.append(seq)
                self.active_sequences.pop(key, None)
                self.radix_tree.release_sequence(seq.kv_block_hashes)

        self.timeline_time = step_end
        return completed

    def admit_sequence(
        self,
        transfer: KVTransferInfo,
        block_size: int,
        ready_time: float,
        max_tokens: int,
    ) -> List[DecodeSequence]:
        """
        Admit a transferred KV sequence into the decode timeline.

        Returns sequences that completed while advancing this node to the
        admission point.  The newly admitted sequence completes later via
        ``advance_time``/``drain`` unless ``max_tokens`` is zero.
        """
        completed = self._advance_to_admission_boundary(ready_time)

        while self.active_count >= self.max_concurrent_sequences:
            completed.extend(self._run_one_decode_step())

        start_time = max(ready_time, self.timeline_time)
        sequence_id = transfer.sequence_id or transfer.session_id

        if max_tokens <= 0:
            completed.append(
                DecodeSequence(
                    session_id=transfer.session_id,
                    sequence_id=sequence_id,
                    kv_block_hashes=list(transfer.block_hashes),
                    ready_time=ready_time,
                    start_time=start_time,
                    max_tokens=0,
                    first_token_time=start_time,
                    finish_time=start_time,
                )
            )
            return completed

        self.radix_tree.insert_sequence(
            transfer.block_hashes, block_size, start_time
        )
        self.radix_tree.acquire_sequence(transfer.block_hashes)

        self.active_sequences[sequence_id] = DecodeSequence(
            session_id=transfer.session_id,
            sequence_id=sequence_id,
            kv_block_hashes=list(transfer.block_hashes),
            tokens_generated=0,
            ready_time=ready_time,
            start_time=start_time,
            max_tokens=max_tokens,
        )
        return completed

    def drain(self) -> List[DecodeSequence]:
        """Run decode until all active sequences complete."""
        completed: List[DecodeSequence] = []
        while self.active_sequences:
            completed.extend(self._run_one_decode_step())
        return completed

    def receive_kv(
        self,
        transfer: KVTransferInfo,
        block_size: int,
        current_time: float,
    ) -> None:
        """Receive KV blocks from prefill node and insert into local storage."""
        self.admit_sequence(
            transfer=transfer,
            block_size=block_size,
            ready_time=current_time,
            max_tokens=128,
        )

    def decode_step(self, session_id: str, current_time: float) -> float:
        """
        One autoregressive decode step with continuous batching.

        The GPU processes ALL active sequences in one step.  Step time
        is dominated by model weight reads; KV cache reads add sub-linear
        marginal overhead per sequence.

        Returns the per-step time (ms).
        """
        self._advance_to_admission_boundary(current_time)
        key = self._sequence_key(session_id)
        seq = self.active_sequences.get(key)
        if seq is None:
            return 0.0

        # Continuous batching: step time depends on concurrent sequences
        step_ms = self.compute_cfg.decode_step_ms(len(self.active_sequences))
        self._run_one_decode_step()
        return step_ms

    def finish_sequence(self, session_id: str) -> None:
        """Release resources for a completed sequence."""
        key = self._sequence_key(session_id)
        seq = self.active_sequences.pop(key, None)
        if seq is not None:
            self.radix_tree.release_sequence(seq.kv_block_hashes)

    @property
    def active_count(self) -> int:
        return len(self.active_sequences)

    def has_capacity(self) -> bool:
        return self.active_count < self.max_concurrent_sequences

    def cached_hashes(self) -> set:
        return self.radix_tree.cached_hashes()

    def reset_metrics(self) -> None:
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])
        for seq in list(self.active_sequences.values()):
            self.radix_tree.release_sequence(seq.kv_block_hashes)
        self.active_sequences.clear()
        self.timeline_time = 0.0
