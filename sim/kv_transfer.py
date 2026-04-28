"""
KVTransferModel — models the cost of moving KV cache between P and D nodes.

Topology-aware transfer:
  Same node (NVLink):    ~900 GB/s,  1μs base  → 1.4ms for 1.25 GB
  Same rack (RDMA):      ~12.5 GB/s, 8μs base  → 100ms for 1.25 GB
  Cross rack (spine):    ~12.5 GB/s, 20μs base → 100ms for 1.25 GB

In production, P and D GPUs are co-located on the same node (8-GPU NVLink
mesh), so KV transfer uses NVLink — 72× faster than RDMA.

Strategies:
  push          : prefill sends KV immediately (default, lowest TTFT)
  pull          : decode requests KV when ready (+1 RTT)
  pull_on_demand: decode fetches blocks lazily during generation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .network import NetworkModel


@dataclass
class TransferConfig:
    strategy: str = "push"           # "push" | "pull" | "pull_on_demand"
    rdma_bw_gbps: float = 12.5      # Effective GB/s; 100 Gbps NIC = 12.5 GB/s
    rdma_latency_us: float = 5.0    # base RDMA round-trip
    pipelining: bool = True          # overlap transfer with decode start
    pipeline_chunk_blocks: int = 16  # send this many blocks per chunk
    compression_ratio: float = 1.0   # 1.0 = none, 0.5 = 2× compression

    @staticmethod
    def from_config(cfg: dict) -> "TransferConfig":
        tc = cfg.get("pd_separation", {}).get("transfer", {})
        return TransferConfig(
            strategy=tc.get("strategy", "push"),
            rdma_bw_gbps=tc.get("rdma_bw_gbps", 12.5),
            rdma_latency_us=tc.get("rdma_latency_us", 5.0),
            pipelining=tc.get("pipelining", True),
            pipeline_chunk_blocks=tc.get("pipeline_chunk_blocks", 16),
            compression_ratio=tc.get("compression_ratio", 1.0),
        )


@dataclass(frozen=True)
class KVTransferTiming:
    """Full-transfer and TTFT-visible transfer timing."""

    full_ms: float
    ttft_ms: float
    queue_wait_ms: float = 0.0
    path: str = ""


class KVTransferModel:
    """Topology-aware KV transfer latency model."""

    def __init__(self, config: TransferConfig, network: NetworkModel) -> None:
        self.config = config
        self.network = network

    def transfer_latency_ms(
        self,
        num_blocks: int,
        block_size: int,
        same_rack: bool,
        src_gpu: int = -1,
        dst_gpu: int = -1,
    ) -> float:
        """
        Total transfer time (ms) for KV blocks.

        Picks NVLink (same node) or RDMA (cross-node) based on GPU IDs.
        """
        return self.transfer_timing_ms(
            num_blocks,
            block_size,
            same_rack,
            src_gpu=src_gpu,
            dst_gpu=dst_gpu,
        ).full_ms

    def transfer_timing_ms(
        self,
        num_blocks: int,
        block_size: int,
        same_rack: bool,
        src_gpu: int = -1,
        dst_gpu: int = -1,
        start_time_ms: float | None = None,
    ) -> KVTransferTiming:
        """
        Full and TTFT-visible transfer times for one KV movement.

        When the NetworkModel has contention enabled, this method reserves the
        selected interconnect once and returns both full-transfer and first
        pipeline chunk timing from the same queued transfer.
        """
        if num_blocks <= 0:
            return KVTransferTiming(0.0, 0.0)

        total_bytes = int(num_blocks * block_size * self.config.compression_ratio)
        chunk_blocks = (
            self.config.pipeline_chunk_blocks
            if self.config.pipelining
            else num_blocks
        )
        first_bytes = int(
            min(num_blocks, chunk_blocks) * block_size * self.config.compression_ratio
        )
        path, bandwidth_gbps, base_us, link_key = self._path_params(
            num_blocks,
            same_rack,
            src_gpu,
            dst_gpu,
        )
        timing = self.network.schedule_transfer(
            total_bytes=total_bytes,
            first_chunk_bytes=first_bytes,
            bandwidth_gbps=bandwidth_gbps,
            base_latency_us=base_us,
            link_key=link_key,
            start_time_ms=start_time_ms,
        )
        return KVTransferTiming(
            full_ms=timing.full_ms,
            ttft_ms=timing.first_chunk_ms,
            queue_wait_ms=timing.queue_wait_ms,
            path=path,
        )

    def _path_params(
        self,
        num_blocks: int,
        same_rack: bool,
        src_gpu: int,
        dst_gpu: int,
    ) -> tuple[str, float, float, Hashable]:
        same_node = (
            same_rack
            and src_gpu >= 0
            and dst_gpu >= 0
            and self.network._same_node(src_gpu, dst_gpu)
        )
        if same_node:
            node_id = src_gpu // self.network.gpus_per_node
            return (
                "nvlink",
                self.network.nvlink_bw_gbps,
                self.network.nvlink_latency_us,
                ("nvlink", node_id),
            )

        path = "rdma_intra_rack" if same_rack else "rdma_cross_rack"
        base_us = (
            self.network.intra_rack_us if same_rack else self.network.cross_rack_us
        )
        base_us += self.config.rdma_latency_us

        if self.config.strategy == "pull":
            base_us += base_us
        elif self.config.strategy == "pull_on_demand":
            n_chunks = max(1, num_blocks // self.config.pipeline_chunk_blocks)
            base_us += base_us * n_chunks * 0.1

        src_node = src_gpu // self.network.gpus_per_node if src_gpu >= 0 else -1
        dst_node = dst_gpu // self.network.gpus_per_node if dst_gpu >= 0 else -1
        link_key = (path, min(src_node, dst_node), max(src_node, dst_node))
        return path, self.config.rdma_bw_gbps, base_us, link_key

    def pipelined_first_chunk_ms(
        self,
        block_size: int,
        same_rack: bool,
        src_gpu: int = -1,
        dst_gpu: int = -1,
    ) -> float:
        """Time until first pipeline chunk arrives (for TTFT with pipelining)."""
        chunk_blocks = self.config.pipeline_chunk_blocks
        return self.transfer_latency_ms(
            chunk_blocks, block_size, same_rack, src_gpu, dst_gpu
        )

    def effective_ttft_transfer_ms(
        self,
        num_blocks: int,
        block_size: int,
        same_rack: bool,
        src_gpu: int = -1,
        dst_gpu: int = -1,
    ) -> float:
        """
        Transfer contribution to TTFT.

        With pipelining: first-chunk latency (decode starts early).
        Without:         full transfer latency.
        """
        if self.config.pipelining and num_blocks > self.config.pipeline_chunk_blocks:
            return self.transfer_timing_ms(
                num_blocks, block_size, same_rack, src_gpu, dst_gpu
            ).ttft_ms
        return self.transfer_latency_ms(
            num_blocks, block_size, same_rack, src_gpu, dst_gpu
        )
