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

Compute calibration (H100, 7B model)
-------------------------------------
  Prefill  : ~0.035 ms/token  (2 × 14 GB / 800 TFLOPS)
  Decode   : ~8.75 ms/token   (2 × 14 GB / 3200 GB/s)
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
    model_params_b: float = 7.0          # 7B parameter model
    kv_bytes_per_token: int = 64         # per token KV cache size
    tokens_per_block: int = 32
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


@dataclass
class DecodeSequence:
    """Tracks an active decode sequence on a decode node."""

    session_id: str
    kv_block_hashes: List[str]
    tokens_generated: int = 0
    start_time: float = 0.0
    max_tokens: int = 128


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
        Execute prefill phase for a request.

        1. Look up prefix in radix tree (cache hit → skip compute)
        2. Check EIC for additional hits
        3. Compute KV for remaining (new) tokens
        4. Insert new blocks into radix tree
        5. Run prefetch: predict next-turn blocks and pre-load
        6. Return result with timing
        """
        block_hashes = request.block_hashes
        block_size = request.block_size

        # Queue wait: GPU busy until earliest_available_time
        effective_time = max(current_time, self.earliest_available_time)
        queue_wait = effective_time - current_time

        # 1. Radix tree lookup — find cached prefix
        match_depth, matched_nodes = self.radix_tree.lookup_prefix(block_hashes)
        cached_blocks = match_depth

        # 2. Check EIC for blocks beyond radix tree match
        eic_hits = 0
        for bh in block_hashes[match_depth:]:
            blk = self.eic.read(bh, self.gpu_id, effective_time)
            if blk is not None:
                eic_hits += 1
                self.metrics.record_hit("EIC", 0.0)
            else:
                break  # prefix chain broken

        total_cached = cached_blocks + eic_hits
        new_blocks = len(block_hashes) - total_cached

        # Record cache metrics
        for _ in range(cached_blocks):
            self.metrics.total_requests += 1
            lat = self.hbm.transfer_latency_ms(block_size, is_read=True)
            self.metrics.record_hit("HBM", lat)
        for _ in range(new_blocks):
            self.metrics.total_requests += 1
            self.metrics.record_miss()

        # 3. Compute time for new tokens (continuous batching aware)
        new_tokens = new_blocks * self.compute_cfg.tokens_per_block
        # Check if there are other requests in flight (batch > 1)
        batch_size = max(1, self.queue_depth)
        compute_ms = self.compute_cfg.batched_prefill_ms(new_tokens, batch_size)

        # 4. Insert new blocks into radix tree + async EIC backup
        new_hashes = block_hashes[total_cached:]
        self.radix_tree.insert_sequence(
            new_hashes, block_size, effective_time + compute_ms
        )
        self.radix_tree.acquire_sequence(block_hashes)

        # Async EIC backup for new blocks
        for bh in new_hashes:
            blk = KVBlock(
                block_hash=bh,
                size_bytes=block_size,
                prefix_depth=block_hashes.index(bh) if bh in block_hashes else 0,
                last_access_time=effective_time + compute_ms,
                access_count=1,
            )
            if not self.eic.contains(bh):
                self.eic.write(blk, self.gpu_id, effective_time + compute_ms)

        # 5. Prefetch: predict and pre-load next-turn blocks
        prefetched = 0
        self.prefetch.record_sequence(request.session_id, block_hashes)
        if block_hashes:
            candidates = self.prefetch.candidates(block_hashes[-1], request.session_id)
            for cand_hash in candidates:
                if not self.radix_tree.contains(cand_hash):
                    # Check if available in EIC
                    blk = self.eic.read(cand_hash, self.gpu_id, effective_time + compute_ms)
                    if blk is not None:
                        self.radix_tree.insert_sequence(
                            [cand_hash], block_size, effective_time + compute_ms
                        )
                        prefetched += 1
                        self.metrics.prefetches += 1

        # Update queue model — prefill GPU is free after compute
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
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])

    def receive_kv(
        self,
        transfer: KVTransferInfo,
        block_size: int,
        current_time: float,
    ) -> None:
        """Receive KV blocks from prefill node and insert into local storage."""
        self.radix_tree.insert_sequence(
            transfer.block_hashes, block_size, current_time
        )
        self.radix_tree.acquire_sequence(transfer.block_hashes)

        self.active_sequences[transfer.session_id] = DecodeSequence(
            session_id=transfer.session_id,
            kv_block_hashes=list(transfer.block_hashes),
            start_time=current_time,
        )

    def decode_step(self, session_id: str, current_time: float) -> float:
        """
        One autoregressive decode step with continuous batching.

        The GPU processes ALL active sequences in one step.  Step time
        is dominated by model weight reads; KV cache reads add sub-linear
        marginal overhead per sequence.

        Returns the per-step time (ms).
        """
        seq = self.active_sequences.get(session_id)
        if seq is None:
            return 0.0

        # Continuous batching: step time depends on concurrent sequences
        step_ms = self.compute_cfg.decode_step_ms(len(self.active_sequences))
        seq.tokens_generated += 1
        return step_ms

    def finish_sequence(self, session_id: str) -> None:
        """Release resources for a completed sequence."""
        seq = self.active_sequences.pop(session_id, None)
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
        self.active_sequences.clear()
