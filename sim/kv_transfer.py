"""
KVTransferModel — models the cost of moving KV cache between P and D nodes.

Strategies
----------
  push          : prefill node sends KV immediately after computation (default)
  pull          : decode node requests KV when it has capacity
  pull_on_demand: decode fetches blocks lazily during generation

Push is the production default (DistServe, Mooncake) as it minimises TTFT.
"""
from __future__ import annotations

from dataclasses import dataclass

from .network import NetworkModel


@dataclass
class TransferConfig:
    strategy: str = "push"           # "push" | "pull" | "pull_on_demand"
    rdma_bw_gbps: float = 100.0     # NIC bandwidth for KV transfer
    rdma_latency_us: float = 5.0    # base RDMA round-trip
    pipelining: bool = True          # overlap transfer with decode start
    pipeline_chunk_blocks: int = 16  # send this many blocks per chunk
    compression_ratio: float = 1.0   # 1.0 = none, 0.5 = 2× compression

    @staticmethod
    def from_config(cfg: dict) -> "TransferConfig":
        tc = cfg.get("pd_separation", {}).get("transfer", {})
        return TransferConfig(
            strategy=tc.get("strategy", "push"),
            rdma_bw_gbps=tc.get("rdma_bw_gbps", 100.0),
            rdma_latency_us=tc.get("rdma_latency_us", 5.0),
            pipelining=tc.get("pipelining", True),
            pipeline_chunk_blocks=tc.get("pipeline_chunk_blocks", 16),
            compression_ratio=tc.get("compression_ratio", 1.0),
        )


class KVTransferModel:
    """Computes transfer latencies for KV cache movement between nodes."""

    def __init__(self, config: TransferConfig, network: NetworkModel) -> None:
        self.config = config
        self.network = network

    def transfer_latency_ms(
        self,
        num_blocks: int,
        block_size: int,
        same_rack: bool,
    ) -> float:
        """
        Total transfer time (ms) for moving ``num_blocks`` KV blocks.

        Includes network base latency + bandwidth-limited transfer time.
        If pipelining is enabled, the effective latency is the time for the
        *last* chunk to arrive (earlier chunks overlap with decode startup).
        """
        if num_blocks <= 0:
            return 0.0
        total_bytes = num_blocks * block_size * self.config.compression_ratio
        bw_bytes_per_sec = self.config.rdma_bw_gbps * 1e9
        transfer_us = (total_bytes / bw_bytes_per_sec) * 1e6

        base_us = (
            self.network.intra_rack_us if same_rack else self.network.cross_rack_us
        )
        base_us += self.config.rdma_latency_us

        # Pull adds one extra RTT for the request
        if self.config.strategy == "pull":
            base_us += base_us  # request + response
        elif self.config.strategy == "pull_on_demand":
            # Per-block RTT overhead (amortised over pipeline chunks)
            n_chunks = max(1, num_blocks // self.config.pipeline_chunk_blocks)
            base_us += base_us * n_chunks * 0.1  # amortised overhead

        return (base_us + transfer_us) / 1000.0

    def pipelined_first_chunk_ms(
        self,
        block_size: int,
        same_rack: bool,
    ) -> float:
        """
        Time (ms) until the first pipeline chunk arrives at the decode node.

        With pipelining, decode can issue its first forward pass after
        receiving just ``pipeline_chunk_blocks`` blocks.
        """
        chunk_blocks = self.config.pipeline_chunk_blocks
        return self.transfer_latency_ms(chunk_blocks, block_size, same_rack)

    def effective_ttft_transfer_ms(
        self,
        num_blocks: int,
        block_size: int,
        same_rack: bool,
    ) -> float:
        """
        Transfer contribution to TTFT.

        With pipelining: first-chunk latency (decode starts early).
        Without:         full transfer latency.
        """
        if self.config.pipelining and num_blocks > self.config.pipeline_chunk_blocks:
            return self.pipelined_first_chunk_ms(block_size, same_rack)
        return self.transfer_latency_ms(num_blocks, block_size, same_rack)
