"""
PDCluster — cluster with separate Prefill and Decode node pools.

Partitions GPUs within each rack by P:D ratio.  Prefill and decode
nodes are co-located per rack so intra-rack RDMA transfer is fast.

Architecture
------------
  PDCluster
  ├── Rack 0
  │   ├── PrefillNode[0..P-1]
  │   ├── DecodeNode[0..D-1]
  │   └── EICPool (shared)
  ├── Rack 1 …
  └── NetworkModel
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .cluster import EICPool, Rack
from .kv_transfer import KVTransferModel, TransferConfig
from .network import NetworkModel
from .pd_nodes import ComputeConfig, DecodeNode, PrefillNode
from .policies import LRUPolicy, NoPrefetch, SessionAwarePrefetch
from .storage import StorageTier


@dataclass
class PDConfig:
    """Configuration for PD separation mode."""

    enabled: bool = False
    pd_ratio: Tuple[int, int] = (1, 3)  # P:D GPU ratio within each rack
    chunked_prefill: bool = False
    chunk_size_tokens: int = 512
    max_output_tokens: int = 128
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)

    @staticmethod
    def from_config(cfg: dict) -> "PDConfig":
        pd = cfg.get("pd_separation", {})
        ratio = pd.get("pd_ratio", [1, 3])
        return PDConfig(
            enabled=pd.get("enabled", False),
            pd_ratio=(ratio[0], ratio[1]),
            chunked_prefill=pd.get("chunked_prefill", False),
            chunk_size_tokens=pd.get("chunk_size_tokens", 512),
            max_output_tokens=pd.get("max_output_tokens", 128),
            compute=ComputeConfig.from_config(cfg),
            transfer=TransferConfig.from_config(cfg),
        )


class PDCluster:
    """Cluster with separate prefill and decode GPU pools."""

    def __init__(
        self,
        racks: List[Rack],
        network: NetworkModel,
        pd_config: PDConfig,
        prefill_nodes: List[PrefillNode],
        decode_nodes: List[DecodeNode],
    ) -> None:
        self.racks = racks
        self.network = network
        self.pd_config = pd_config
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes

    @property
    def total_prefill_gpus(self) -> int:
        return len(self.prefill_nodes)

    @property
    def total_decode_gpus(self) -> int:
        return len(self.decode_nodes)

    @property
    def total_gpus(self) -> int:
        return self.total_prefill_gpus + self.total_decode_gpus

    @property
    def total_racks(self) -> int:
        return len(self.racks)

    def reset_all(self) -> None:
        for n in self.prefill_nodes:
            n.reset_metrics()
            n.queue_depth = 0
            n.earliest_available_time = 0.0
        for n in self.decode_nodes:
            n.reset_metrics()

    def aggregate_prefill_metrics(self):
        from .metrics import Metrics
        combined = Metrics(tier_names=["HBM", "EIC", "Remote"])
        for n in self.prefill_nodes:
            m = n.metrics
            combined.total_requests += m.total_requests
            combined.total_hits += m.total_hits
            combined.total_misses += m.total_misses
            combined.evictions += m.evictions
            combined.total_latency_ms += m.total_latency_ms
            for t in m.tier_names:
                combined.tier_hits[t] = combined.tier_hits.get(t, 0) + m.tier_hits.get(t, 0)
        return combined

    def eic_utilizations(self) -> Dict[int, float]:
        return {r.rack_id: r.eic.utilization for r in self.racks}

    def total_cross_gpu_eic_hits(self) -> int:
        return sum(r.eic.cross_gpu_hits for r in self.racks)

    def summary(self) -> str:
        p, d = self.pd_config.pd_ratio
        return (
            f"PDCluster: {self.total_gpus} GPUs ({self.total_prefill_gpus}P + "
            f"{self.total_decode_gpus}D, ratio {p}:{d}) × {self.total_racks} racks\n"
            f"  Compute: prefill={self.pd_config.compute.prefill_ms_per_token:.3f} ms/tok, "
            f"decode={self.pd_config.compute.decode_ms_per_token:.3f} ms/tok\n"
            f"  Transfer: {self.pd_config.transfer.strategy}, "
            f"BW={self.pd_config.transfer.rdma_bw_gbps} Gbps, "
            f"pipeline={'on' if self.pd_config.transfer.pipelining else 'off'}"
        )


# ======================================================================
# Builder
# ======================================================================


def build_pd_cluster(
    cfg: dict,
    pd_config: PDConfig | None = None,
    eviction_factory=None,
    prefetch_factory=None,
) -> PDCluster:
    """
    Build a PDCluster from config.

    Partitions GPUs per rack: first P/(P+D) are prefill, rest are decode.
    Both share the same EIC pool within the rack.
    """
    if pd_config is None:
        pd_config = PDConfig.from_config(cfg)

    cc = cfg.get("cluster", {})
    net_cfg = cc.get("network", {})
    gpu_cfg = cc.get("gpu", {})
    eic_cfg = cc.get("eic", {})

    n_racks = cc.get("simulate_racks", 8)
    n_gpus_per_rack = cc.get("simulate_gpus_per_rack", 16)
    eic_nodes = eic_cfg.get("nodes_per_rack", 4)

    network = NetworkModel(
        intra_rack_us=net_cfg.get("intra_rack_latency_us", 3.0),
        cross_rack_us=net_cfg.get("cross_rack_latency_us", 15.0),
        remote_ssd_us=net_cfg.get("remote_ssd_latency_us", 200.0),
        p2p_rdma_bw_gbps=net_cfg.get("p2p_rdma_bw_gbps", 100.0),
        p2p_rdma_latency_us=net_cfg.get("p2p_rdma_latency_us", 5.0),
        nvlink_bw_gbps=net_cfg.get("nvlink_bw_gbps", 900.0),
        nvlink_latency_us=net_cfg.get("nvlink_latency_us", 1.0),
        gpus_per_node=net_cfg.get("gpus_per_node", 8),
    )

    # Compute P:D split per rack
    p_ratio, d_ratio = pd_config.pd_ratio
    total_ratio = p_ratio + d_ratio
    n_prefill_per_rack = max(1, round(n_gpus_per_rack * p_ratio / total_ratio))
    n_decode_per_rack = n_gpus_per_rack - n_prefill_per_rack

    all_prefill: List[PrefillNode] = []
    all_decode: List[DecodeNode] = []
    racks: List[Rack] = []
    gid = 0

    for rid in range(n_racks):
        eic = EICPool(
            rack_id=rid,
            num_nodes=eic_nodes,
            capacity_per_node_gb=eic_cfg.get("capacity_per_node_gb", 0.02),
            read_bw_gbps=eic_cfg.get("read_bw_gbps", 100.0),
            write_bw_gbps=eic_cfg.get("write_bw_gbps", 80.0),
            access_latency_ms=eic_cfg.get("access_latency_ms", 0.005),
        )

        gpu_nodes = []  # for Rack compatibility

        # Prefill nodes
        for _ in range(n_prefill_per_rack):
            hbm = StorageTier(
                name=f"HBM-P{gid}",
                capacity_bytes=int(gpu_cfg.get("hbm_capacity_gb", 0.003) * 1e9),
                read_bw_gbps=gpu_cfg.get("hbm_read_bw_gbps", 3200),
                write_bw_gbps=gpu_cfg.get("hbm_write_bw_gbps", 3200),
                read_latency_ms=gpu_cfg.get("hbm_latency_ms", 0.001),
            )
            eviction = eviction_factory() if eviction_factory else LRUPolicy()
            prefetch = prefetch_factory() if prefetch_factory else SessionAwarePrefetch()
            pn = PrefillNode(
                gpu_id=gid,
                rack_id=rid,
                hbm=hbm,
                eic=eic,
                eviction=eviction,
                prefetch=prefetch,
                network=network,
                compute_cfg=pd_config.compute,
            )
            all_prefill.append(pn)
            gid += 1

        # Decode nodes
        for _ in range(n_decode_per_rack):
            hbm = StorageTier(
                name=f"HBM-D{gid}",
                capacity_bytes=int(gpu_cfg.get("hbm_capacity_gb", 0.003) * 1e9),
                read_bw_gbps=gpu_cfg.get("hbm_read_bw_gbps", 3200),
                write_bw_gbps=gpu_cfg.get("hbm_write_bw_gbps", 3200),
                read_latency_ms=gpu_cfg.get("hbm_latency_ms", 0.001),
            )
            eviction = eviction_factory() if eviction_factory else LRUPolicy()
            dn = DecodeNode(
                gpu_id=gid,
                rack_id=rid,
                hbm=hbm,
                eic=eic,
                eviction=eviction,
                network=network,
                compute_cfg=pd_config.compute,
            )
            all_decode.append(dn)
            gid += 1

        racks.append(Rack(rid, gpu_nodes, eic))

    return PDCluster(racks, network, pd_config, all_prefill, all_decode)
