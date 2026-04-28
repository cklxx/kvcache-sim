"""
Cluster — 万卡级 KV Cache 集群模拟器

Architecture
------------
  Cluster (10 240 GPUs, 160 racks)
  ├── Rack 0
  │   ├── GPUNode[0..63]   (each: HBM private)
  │   ├── EICPool           (shared CXL/RDMA memory, 4 nodes × 2 TB)
  │   └── (RemoteSSD via fabric)
  ├── Rack 1
  │   └── …
  └── NetworkModel

Tier hierarchy per GPU read path:
  HBM (private, <1 μs) → EIC (shared in-rack, ~3 μs RDMA)
                        → Remote (cross-rack or SSD, 15-200 μs)
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .metrics import Metrics
from .network import NetworkModel
from .policies import EvictionPolicy, LRUPolicy, NoPrefetch, PrefetchPolicy
from .storage import KVBlock, StorageTier


# ======================================================================
# EIC Pool  (External Interconnect Cache — shared per rack)
# ======================================================================


class EICPool:
    """
    Shared disaggregated-memory pool within a rack, accessed via CXL/RDMA.

    All GPU nodes in the rack can read and write blocks here.  A rack-level
    LRU policy governs eviction.
    """

    def __init__(
        self,
        rack_id: int,
        num_nodes: int = 4,
        capacity_per_node_gb: float = 2048.0,
        read_bw_gbps: float = 100.0,
        write_bw_gbps: float = 80.0,
        access_latency_ms: float = 0.005,
    ) -> None:
        self.rack_id = rack_id
        self.num_nodes = num_nodes
        total = int(capacity_per_node_gb * num_nodes * 1e9)
        self.tier = StorageTier(
            name=f"EIC-R{rack_id}",
            capacity_bytes=total,
            read_bw_gbps=read_bw_gbps,
            write_bw_gbps=write_bw_gbps,
            read_latency_ms=access_latency_ms,
        )
        self.eviction = LRUPolicy()
        self.cross_gpu_hits: int = 0   # block was written by GPU A, read by GPU B
        self._block_owner: Dict[str, int] = {}

    # ── public API ────────────────────────────────────────────────────

    def read(self, block_hash: str, gpu_id: int, current_time: float) -> Optional[KVBlock]:
        blk = self.tier.get(block_hash)
        if blk is not None:
            blk.touch(current_time)
            self.eviction.record_access(block_hash, current_time)
            if self._block_owner.get(block_hash, gpu_id) != gpu_id:
                self.cross_gpu_hits += 1
        return blk

    def write(self, block: KVBlock, gpu_id: int, current_time: float) -> float:
        if self.tier.contains(block.block_hash):
            return 0.0
        while not self.tier.has_space(block.size_bytes):
            victim = self.eviction.evict_candidate(self.tier.blocks)
            if victim is None:
                break
            self.tier.remove(victim)
            self.eviction.remove(victim)
            self._block_owner.pop(victim, None)
        if self.tier.insert(block):
            self.eviction.record_access(block.block_hash, current_time)
            self._block_owner[block.block_hash] = gpu_id
            return self.tier.transfer_latency_ms(block.size_bytes, is_read=False)
        return 0.0

    def contains(self, block_hash: str) -> bool:
        return self.tier.contains(block_hash)

    @property
    def utilization(self) -> float:
        return self.tier.utilization

    @property
    def block_count(self) -> int:
        return len(self.tier.blocks)

    def __repr__(self) -> str:
        return (
            f"EICPool(rack={self.rack_id}, nodes={self.num_nodes}, "
            f"util={self.utilization:.1%}, blocks={self.block_count}, "
            f"cross_gpu_hits={self.cross_gpu_hits})"
        )


# ======================================================================
# GPU Node
# ======================================================================


class GPUNode:
    """
    Single GPU node with private HBM and access to the rack's shared EIC.

    read()  → HBM → EIC (same rack) → miss
    write() → HBM + async EIC backup
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
        selective_write: bool = False,
        selective_write_depth: int = 3,
    ) -> None:
        self.gpu_id = gpu_id
        self.rack_id = rack_id
        self.hbm = hbm
        self.eic = eic
        self.eviction = eviction
        self.prefetch = prefetch
        self.network = network
        self.selective_write = selective_write
        self.selective_write_depth = selective_write_depth
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])
        self._pending_eic: List[Tuple[KVBlock, float]] = []
        self._ever_computed: set = set()  # track all blocks ever computed on this GPU

    # ── read / write / tick ───────────────────────────────────────────

    def read(
        self,
        block_hash: str,
        size_bytes: int,
        prefix_depth: int,
        current_time: float,
        session_id: str = "",
    ) -> Tuple[Optional[str], float]:
        self.metrics.total_requests += 1

        # 1) HBM
        blk = self.hbm.get(block_hash)
        if blk is not None:
            blk.touch(current_time)
            self.eviction.record_access(block_hash, current_time)
            lat = self.hbm.transfer_latency_ms(size_bytes, is_read=True)
            self.metrics.record_hit("HBM", lat)
            return "HBM", lat

        # 2) EIC  (shared, same rack, via RDMA)
        blk = self.eic.read(block_hash, self.gpu_id, current_time)
        if blk is not None:
            lat = (
                self.eic.tier.transfer_latency_ms(size_bytes, is_read=True)
                + self.network.intra_rack_ms()
            )
            self.metrics.record_hit("EIC", lat)
            self.metrics.promotions += 1
            # promote into HBM
            promoted = KVBlock(
                block_hash=blk.block_hash,
                size_bytes=blk.size_bytes,
                prefix_depth=blk.prefix_depth,
                last_access_time=current_time,
                access_count=blk.access_count + 1,
            )
            self._insert_hbm(promoted, current_time)
            self.eviction.record_access(block_hash, current_time)
            return "EIC", lat

        # 3) miss → caller must recompute
        self.metrics.record_miss()
        return None, 0.0

    def write(
        self,
        block_hash: str,
        size_bytes: int,
        prefix_depth: int,
        current_time: float,
        session_id: str = "",
    ) -> float:
        if self.selective_write and prefix_depth > self.selective_write_depth:
            return 0.0
        if self.hbm.contains(block_hash):
            return 0.0

        blk = KVBlock(
            block_hash=block_hash,
            size_bytes=size_bytes,
            prefix_depth=prefix_depth,
            last_access_time=current_time,
            access_count=1,
        )
        self.eviction.record_access(block_hash, current_time)
        lat = self._insert_hbm(blk, current_time)

        # async EIC backup (flush in tick())
        self._pending_eic.append((blk, current_time + 1.0))
        return lat

    def tick(self, current_time: float) -> None:
        remaining = []
        for blk, eta in self._pending_eic:
            if current_time >= eta:
                if not self.eic.contains(blk.block_hash):
                    self.eic.write(blk, self.gpu_id, current_time)
            else:
                remaining.append((blk, eta))
        self._pending_eic = remaining

    # ── internal ──────────────────────────────────────────────────────

    def _insert_hbm(self, block: KVBlock, current_time: float) -> float:
        while not self.hbm.has_space(block.size_bytes):
            victim_hash = self.eviction.evict_candidate(self.hbm.blocks)
            if victim_hash is None:
                break
            evicted = self.hbm.remove(victim_hash)
            if evicted is not None:
                self.metrics.evictions += 1
                # demote to shared EIC
                self.eic.write(evicted, self.gpu_id, current_time)
        if self.hbm.insert(block):
            return self.hbm.transfer_latency_ms(block.size_bytes, is_read=False)
        return 0.0

    # ── helpers ───────────────────────────────────────────────────────

    def process_request(
        self,
        block_hashes: List[str],
        block_size: int,
        session_id: str,
        current_time: float,
    ) -> Tuple[int, float]:
        hits = 0
        total_lat = 0.0
        for depth, bh in enumerate(block_hashes):
            ever_seen = bh in self._ever_computed
            tier, lat = self.read(bh, block_size, depth, current_time, session_id)
            if tier:
                hits += 1
                total_lat += lat
                if ever_seen:
                    # Prefix block: computed before, found in cache
                    self.metrics.prefix_blocks += 1
                    self.metrics.prefix_hits += 1
                else:
                    # Shouldn't happen often (hit on never-computed block)
                    self.metrics.prefix_blocks += 1
                    self.metrics.prefix_hits += 1
            else:
                total_lat += self.write(bh, block_size, depth, current_time, session_id)
                self._ever_computed.add(bh)
                if ever_seen:
                    # EVICTED prefix block: was computed before but got kicked out
                    self.metrics.prefix_blocks += 1
                    # prefix_hits NOT incremented → this is a cache miss on a prefix
                else:
                    # Truly new block: first-time compute
                    self.metrics.new_blocks += 1
        self.tick(current_time)
        return hits, total_lat

    def cached_hashes(self) -> set:
        return set(self.hbm.blocks.keys())

    def reset_metrics(self) -> None:
        self.metrics = Metrics(tier_names=["HBM", "EIC", "Remote"])


# ======================================================================
# Rack
# ======================================================================


class Rack:
    def __init__(self, rack_id: int, gpu_nodes: List[GPUNode], eic: EICPool) -> None:
        self.rack_id = rack_id
        self.gpu_nodes = gpu_nodes
        self.eic = eic

    @property
    def num_gpus(self) -> int:
        return len(self.gpu_nodes)


# ======================================================================
# Cluster
# ======================================================================


class Cluster:
    """万卡级集群：N racks × M GPUs + shared EIC per rack."""

    def __init__(
        self,
        racks: List[Rack],
        network: NetworkModel,
        routing_seed: int = 0,
    ) -> None:
        self.racks = racks
        self.network = network
        self.routing_seed = routing_seed
        self.all_gpus: List[GPUNode] = []
        for rack in racks:
            self.all_gpus.extend(rack.gpu_nodes)

    @property
    def total_gpus(self) -> int:
        return len(self.all_gpus)

    @property
    def total_racks(self) -> int:
        return len(self.racks)

    @property
    def total_eic_nodes(self) -> int:
        return sum(r.eic.num_nodes for r in self.racks)

    @property
    def total_hbm_tb(self) -> float:
        return sum(g.hbm.capacity_bytes for g in self.all_gpus) / 1e12

    @property
    def total_eic_tb(self) -> float:
        return sum(r.eic.tier.capacity_bytes for r in self.racks) / 1e12

    def aggregate_metrics(self) -> Metrics:
        combined = Metrics(tier_names=["HBM", "EIC", "Remote"])
        for gpu in self.all_gpus:
            m = gpu.metrics
            combined.total_requests += m.total_requests
            combined.total_hits += m.total_hits
            combined.total_misses += m.total_misses
            combined.evictions += m.evictions
            combined.promotions += m.promotions
            combined.demotions += m.demotions
            combined.prefetches += m.prefetches
            combined.total_latency_ms += m.total_latency_ms
            combined.prefix_blocks += m.prefix_blocks
            combined.prefix_hits += m.prefix_hits
            combined.new_blocks += m.new_blocks
            for t in m.tier_names:
                combined.tier_hits[t] = (
                    combined.tier_hits.get(t, 0) + m.tier_hits.get(t, 0)
                )
                combined.tier_latency_ms[t] = (
                    combined.tier_latency_ms.get(t, 0.0)
                    + m.tier_latency_ms.get(t, 0.0)
                )
        return combined

    @property
    def total_cross_gpu_eic_hits(self) -> int:
        return sum(r.eic.cross_gpu_hits for r in self.racks)

    def eic_utilizations(self) -> Dict[int, float]:
        return {r.rack_id: r.eic.utilization for r in self.racks}

    def reset_all(self) -> None:
        for gpu in self.all_gpus:
            gpu.reset_metrics()

    def summary(self) -> str:
        lines = [
            f"Cluster: {self.total_gpus} GPUs × {self.total_racks} racks, "
            f"{self.total_eic_nodes} EIC nodes",
            f"  Total HBM : {self.total_hbm_tb * 1024:.1f} GB  "
            f"({self.all_gpus[0].hbm.capacity_bytes / 1e6:.0f} MB × {self.total_gpus})",
            f"  Total EIC : {self.total_eic_tb * 1024:.1f} GB  "
            f"({self.racks[0].eic.tier.capacity_bytes / 1e6:.0f} MB × {self.total_racks} racks)",
        ]
        return "\n".join(lines)


# ======================================================================
# Cluster Router (session-affinity + prefix scoring)
# ======================================================================


class ClusterRouter:
    """
    Routes requests across a GPU cluster with locality awareness.

    Strategy:
    1. Session affinity — subsequent turns of a session go to the same GPU
    2. For the first turn, score GPUs by HBM prefix hit + 0.5× EIC prefix hit
    3. Prefer GPUs within the same rack as existing session traffic
    """

    def __init__(self, cluster: Cluster, seed: Optional[int] = None) -> None:
        self.cluster = cluster
        self._rr: int = 0
        self._session_gpu: Dict[str, int] = {}
        self._rng = random.Random(cluster.routing_seed if seed is None else seed)

    def route(self, block_hashes: List[str], session_id: str = "") -> GPUNode:
        # Sticky routing for returning sessions
        if session_id in self._session_gpu:
            gid = self._session_gpu[session_id]
            return self.cluster.all_gpus[gid]

        if not block_hashes:
            return self._next_rr(session_id)

        # Score a random subset of GPUs (for scalability)
        candidates = self.cluster.all_gpus
        if len(candidates) > 32:
            candidates = self._rng.sample(candidates, 32)

        best: Optional[GPUNode] = None
        best_score = -1.0
        for gpu in candidates:
            cached = gpu.cached_hashes()
            score = sum(1.0 for h in block_hashes if h in cached)
            # Bonus for EIC availability (lower weight — higher latency)
            score += sum(0.3 for h in block_hashes if gpu.eic.contains(h))
            if score > best_score:
                best_score = score
                best = gpu

        if best is not None and best_score > 0:
            gid = self.cluster.all_gpus.index(best)
            self._session_gpu[session_id] = gid
            return best

        return self._next_rr(session_id)

    def _next_rr(self, session_id: str) -> GPUNode:
        gid = self._rr % len(self.cluster.all_gpus)
        self._session_gpu[session_id] = gid
        self._rr += 1
        return self.cluster.all_gpus[gid]


# ======================================================================
# Builder
# ======================================================================


def build_cluster(
    cfg: dict,
    eviction_factory=None,
    prefetch_factory=None,
    selective_write: bool = False,
    selective_write_depth: int = 3,
) -> Cluster:
    """
    Construct a Cluster from a config dict.

    The config describes the full 万卡 cluster; we simulate a
    representative subset controlled by ``simulate_racks`` and
    ``simulate_gpus_per_rack``.
    """
    cc = cfg.get("cluster", {})
    net_cfg = cc.get("network", {})
    gpu_cfg = cc.get("gpu", {})
    eic_cfg = cc.get("eic", {})

    n_racks = cc.get("simulate_racks", 8)
    n_gpus = cc.get("simulate_gpus_per_rack", 16)
    eic_nodes = eic_cfg.get("nodes_per_rack", 4)

    network = NetworkModel(
        intra_rack_us=net_cfg.get("intra_rack_latency_us", 3.0),
        cross_rack_us=net_cfg.get("cross_rack_latency_us", 15.0),
        remote_ssd_us=net_cfg.get("remote_ssd_latency_us", 200.0),
    )

    racks: List[Rack] = []
    gid = 0
    routing_seed = cc.get(
        "routing_seed",
        cfg.get("cluster_trace", cfg.get("trace", {})).get("seed", 42),
    )
    for rid in range(n_racks):
        # Shared EIC for this rack
        eic = EICPool(
            rack_id=rid,
            num_nodes=eic_nodes,
            capacity_per_node_gb=eic_cfg.get("capacity_per_node_gb", 0.02),
            read_bw_gbps=eic_cfg.get("read_bw_gbps", 100.0),
            write_bw_gbps=eic_cfg.get("write_bw_gbps", 80.0),
            access_latency_ms=eic_cfg.get("access_latency_ms", 0.005),
        )

        gpus: List[GPUNode] = []
        for local_idx in range(n_gpus):
            hbm = StorageTier(
                name=f"HBM-G{gid}",
                capacity_bytes=int(gpu_cfg.get("hbm_capacity_gb", 0.003) * 1e9),
                read_bw_gbps=gpu_cfg.get("hbm_read_bw_gbps", 3200),
                write_bw_gbps=gpu_cfg.get("hbm_write_bw_gbps", 3200),
                read_latency_ms=gpu_cfg.get("hbm_latency_ms", 0.001),
            )
            eviction = eviction_factory() if eviction_factory else LRUPolicy()
            prefetch = prefetch_factory() if prefetch_factory else NoPrefetch()

            gpu = GPUNode(
                gpu_id=gid,
                rack_id=rid,
                hbm=hbm,
                eic=eic,
                eviction=eviction,
                prefetch=prefetch,
                network=network,
                selective_write=selective_write,
                selective_write_depth=selective_write_depth,
            )
            gpus.append(gpu)
            gid += 1

        racks.append(Rack(rid, gpus, eic))

    return Cluster(racks, network, routing_seed=routing_seed)
