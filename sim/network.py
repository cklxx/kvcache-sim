"""
NetworkModel — latency and bandwidth model for cluster interconnects.

Models three network paths:
  1. Intra-rack (GPU ↔ EIC via CXL/RDMA): ~3 μs
  2. Cross-rack (spine fabric):             ~15 μs
  3. Remote SSD (disaggregated NVMe):       ~200 μs
"""
from __future__ import annotations


class NetworkModel:
    def __init__(
        self,
        intra_rack_us: float = 3.0,
        cross_rack_us: float = 15.0,
        remote_ssd_us: float = 200.0,
        p2p_rdma_bw_gbps: float = 100.0,
        p2p_rdma_latency_us: float = 5.0,
    ) -> None:
        self.intra_rack_us = intra_rack_us
        self.cross_rack_us = cross_rack_us
        self.remote_ssd_us = remote_ssd_us
        self.p2p_rdma_bw_gbps = p2p_rdma_bw_gbps
        self.p2p_rdma_latency_us = p2p_rdma_latency_us

    def intra_rack_ms(self) -> float:
        return self.intra_rack_us / 1000.0

    def cross_rack_ms(self) -> float:
        return self.cross_rack_us / 1000.0

    def remote_ssd_ms(self) -> float:
        return self.remote_ssd_us / 1000.0

    def p2p_transfer_ms(self, size_bytes: int, same_rack: bool) -> float:
        """GPU-to-GPU RDMA transfer latency (ms) for KV cache movement."""
        base_us = self.intra_rack_us if same_rack else self.cross_rack_us
        base_us += self.p2p_rdma_latency_us
        transfer_us = (size_bytes / (self.p2p_rdma_bw_gbps * 1e9)) * 1e6
        return (base_us + transfer_us) / 1000.0

    @staticmethod
    def from_config(cfg: dict) -> "NetworkModel":
        net = cfg.get("cluster", {}).get("network", {})
        return NetworkModel(
            intra_rack_us=net.get("intra_rack_latency_us", 3.0),
            cross_rack_us=net.get("cross_rack_latency_us", 15.0),
            remote_ssd_us=net.get("remote_ssd_latency_us", 200.0),
            p2p_rdma_bw_gbps=net.get("p2p_rdma_bw_gbps", 100.0),
            p2p_rdma_latency_us=net.get("p2p_rdma_latency_us", 5.0),
        )

    def __repr__(self) -> str:
        return (
            f"NetworkModel(intra_rack={self.intra_rack_us}μs, "
            f"cross_rack={self.cross_rack_us}μs, "
            f"remote_ssd={self.remote_ssd_us}μs)"
        )
