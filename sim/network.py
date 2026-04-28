"""
NetworkModel — latency and bandwidth model for cluster interconnects.

Models three network paths:
  1. Intra-rack (GPU ↔ EIC via CXL/RDMA): ~3 μs
  2. Cross-rack (spine fabric):             ~15 μs
  3. Remote SSD (disaggregated NVMe):       ~200 μs
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable, Optional


@dataclass(frozen=True)
class TransferTiming:
    """Network transfer timing with queueing separated from service time."""

    full_ms: float
    first_chunk_ms: float
    queue_wait_ms: float = 0.0
    service_ms: float = 0.0


class NetworkModel:
    def __init__(
        self,
        intra_rack_us: float = 3.0,
        cross_rack_us: float = 15.0,
        remote_ssd_us: float = 200.0,
        p2p_rdma_bw_gbps: float = 100.0,  # Effective GB/s for rack/EIC paths
        p2p_rdma_latency_us: float = 5.0,
        nvlink_bw_gbps: float = 900.0,
        nvlink_latency_us: float = 1.0,
        gpus_per_node: int = 8,
        jitter_cv: float = 0.0,
        tail_jitter_prob: float = 0.0,
        tail_jitter_multiplier: float = 1.0,
        contention_enabled: bool = False,
        seed: int = 0,
    ) -> None:
        self.intra_rack_us = intra_rack_us
        self.cross_rack_us = cross_rack_us
        self.remote_ssd_us = remote_ssd_us
        self.p2p_rdma_bw_gbps = p2p_rdma_bw_gbps
        self.p2p_rdma_latency_us = p2p_rdma_latency_us
        # NVLink: intra-node GPU-to-GPU
        self.nvlink_bw_gbps = nvlink_bw_gbps
        self.nvlink_latency_us = nvlink_latency_us
        self.gpus_per_node = gpus_per_node  # GPUs sharing NVLink mesh
        self.jitter_cv = max(0.0, jitter_cv)
        self.tail_jitter_prob = min(max(tail_jitter_prob, 0.0), 1.0)
        self.tail_jitter_multiplier = max(1.0, tail_jitter_multiplier)
        self.contention_enabled = contention_enabled
        self._rng = random.Random(seed)
        self._link_available_ms: dict[Hashable, float] = {}

    def intra_rack_ms(self) -> float:
        return self._apply_jitter(self.intra_rack_us / 1000.0)

    def cross_rack_ms(self) -> float:
        return self._apply_jitter(self.cross_rack_us / 1000.0)

    def remote_ssd_ms(self) -> float:
        return self._apply_jitter(self.remote_ssd_us / 1000.0)

    def p2p_transfer_ms(self, size_bytes: int, same_rack: bool) -> float:
        """GPU-to-GPU RDMA transfer latency (ms), bandwidth expressed as GB/s."""
        base_us = self.intra_rack_us if same_rack else self.cross_rack_us
        base_us += self.p2p_rdma_latency_us
        return self.transfer_duration_ms(
            size_bytes=size_bytes,
            bandwidth_gbps=self.p2p_rdma_bw_gbps,
            base_latency_us=base_us,
        )

    def nvlink_transfer_ms(self, size_bytes: int) -> float:
        """Intra-node NVLink transfer latency (ms). 72x faster than RDMA."""
        return self.transfer_duration_ms(
            size_bytes=size_bytes,
            bandwidth_gbps=self.nvlink_bw_gbps,
            base_latency_us=self.nvlink_latency_us,
        )

    def transfer_duration_ms(
        self,
        size_bytes: int,
        bandwidth_gbps: float,
        base_latency_us: float,
        jitter_multiplier: Optional[float] = None,
    ) -> float:
        """Return service time for one transfer without link queueing."""
        transfer_ms = (size_bytes / (bandwidth_gbps * 1e9)) * 1000.0
        duration_ms = (base_latency_us / 1000.0) + transfer_ms
        if jitter_multiplier is None:
            jitter_multiplier = self._jitter_multiplier()
        return duration_ms * jitter_multiplier

    def schedule_transfer(
        self,
        total_bytes: int,
        first_chunk_bytes: int,
        bandwidth_gbps: float,
        base_latency_us: float,
        link_key: Hashable,
        start_time_ms: Optional[float] = None,
    ) -> TransferTiming:
        """
        Return transfer timing, optionally reserving a shared link.

        If contention is enabled and start_time_ms is provided, transfers using
        the same link_key serialize on that link. Jitter is sampled once per
        transfer and applied consistently to full and first-chunk timings.
        """
        multiplier = self._jitter_multiplier()
        service_ms = self.transfer_duration_ms(
            total_bytes,
            bandwidth_gbps,
            base_latency_us,
            jitter_multiplier=multiplier,
        )
        first_ms = self.transfer_duration_ms(
            min(first_chunk_bytes, total_bytes),
            bandwidth_gbps,
            base_latency_us,
            jitter_multiplier=multiplier,
        )

        queue_wait_ms = 0.0
        if self.contention_enabled and start_time_ms is not None:
            available_ms = self._link_available_ms.get(link_key, 0.0)
            queue_wait_ms = max(0.0, available_ms - start_time_ms)
            self._link_available_ms[link_key] = (
                start_time_ms + queue_wait_ms + service_ms
            )

        return TransferTiming(
            full_ms=queue_wait_ms + service_ms,
            first_chunk_ms=queue_wait_ms + min(first_ms, service_ms),
            queue_wait_ms=queue_wait_ms,
            service_ms=service_ms,
        )

    def reset(self) -> None:
        """Clear contention state while preserving jitter RNG sequence."""
        self._link_available_ms.clear()

    def _jitter_multiplier(self) -> float:
        if self.jitter_cv <= 0.0 and self.tail_jitter_prob <= 0.0:
            return 1.0

        multiplier = 1.0
        if self.jitter_cv > 0.0:
            sigma = math.sqrt(math.log1p(self.jitter_cv * self.jitter_cv))
            mu = -0.5 * sigma * sigma
            multiplier = self._rng.lognormvariate(mu, sigma)

        if (
            self.tail_jitter_prob > 0.0
            and self._rng.random() < self.tail_jitter_prob
        ):
            multiplier *= self.tail_jitter_multiplier
        return multiplier

    def _apply_jitter(self, latency_ms: float) -> float:
        return latency_ms * self._jitter_multiplier()

    def kv_transfer_ms(self, size_bytes: int, src_gpu: int, dst_gpu: int) -> float:
        """
        Pick the right interconnect for KV transfer based on topology.

        Same node (NVLink mesh):   ~900 GB/s, 1μs base
        Same rack (RDMA):          ~12.5 GB/s, 8μs base
        Cross rack (spine fabric): ~12.5 GB/s, 20μs base
        """
        if self._same_node(src_gpu, dst_gpu):
            return self.nvlink_transfer_ms(size_bytes)
        elif self._same_rack(src_gpu, dst_gpu):
            return self.p2p_transfer_ms(size_bytes, same_rack=True)
        else:
            return self.p2p_transfer_ms(size_bytes, same_rack=False)

    def _same_node(self, gpu_a: int, gpu_b: int) -> bool:
        return gpu_a // self.gpus_per_node == gpu_b // self.gpus_per_node

    def _same_rack(self, gpu_a: int, gpu_b: int) -> bool:
        # Rack assignment is handled externally; this is a fallback
        # that assumes gpus are numbered sequentially per rack
        return True  # caller should use rack_id comparison instead

    @staticmethod
    def from_config(cfg: dict) -> "NetworkModel":
        net = cfg.get("cluster", {}).get("network", {})
        return NetworkModel(
            intra_rack_us=net.get("intra_rack_latency_us", 3.0),
            cross_rack_us=net.get("cross_rack_latency_us", 15.0),
            remote_ssd_us=net.get("remote_ssd_latency_us", 200.0),
            p2p_rdma_bw_gbps=net.get("p2p_rdma_bw_gbps", 100.0),
            p2p_rdma_latency_us=net.get("p2p_rdma_latency_us", 5.0),
            nvlink_bw_gbps=net.get("nvlink_bw_gbps", 900.0),
            nvlink_latency_us=net.get("nvlink_latency_us", 1.0),
            gpus_per_node=net.get("gpus_per_node", 8),
            jitter_cv=net.get("jitter_cv", 0.0),
            tail_jitter_prob=net.get("tail_jitter_prob", 0.0),
            tail_jitter_multiplier=net.get("tail_jitter_multiplier", 1.0),
            contention_enabled=net.get("contention_enabled", False),
            seed=net.get(
                "seed",
                cfg.get("cluster_trace", cfg.get("pd_trace", cfg.get("trace", {}))).get(
                    "seed", 42
                ),
            ),
        )

    def __repr__(self) -> str:
        return (
            f"NetworkModel(intra_rack={self.intra_rack_us}μs, "
            f"cross_rack={self.cross_rack_us}μs, "
            f"remote_ssd={self.remote_ssd_us}μs, "
            f"jitter_cv={self.jitter_cv}, contention={self.contention_enabled})"
        )
