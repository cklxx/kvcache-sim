"""
ExperimentRunner — builds all 6 policy configurations and runs them.

Policy configurations
---------------------
  1. Baseline LRU           — LRU + NoPrefetch
  2. +ARC                   — ARC + NoPrefetch
  3. +SessionPrefetch       — LRU + SessionAwarePrefetch
  4. +SelectiveWrite        — LRU + NoPrefetch + selective_write=True
  5. +Learned               — Learned (LightGBM) + NoPrefetch
  6. Belady Oracle          — BeladyOracle + NoPrefetch (offline optimal)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sim.cache_manager import CacheManager
from sim.metrics import Metrics
from sim.policies import (
    ARCPolicy,
    BeladyOracle,
    LearnedPolicy,
    LRUPolicy,
    NoPrefetch,
    SessionAwarePrefetch,
)
from sim.router import Router, Worker
from sim.storage import StorageTier
from trace.generator import Request
from trace.replay import TraceReplayer


@dataclass
class PolicyConfig:
    name: str
    eviction_type: str          # "lru" | "arc" | "learned" | "belady"
    prefetch_type: str          # "none" | "session"
    selective_write: bool = False
    selective_write_depth: int = 3


def _build_tiers(hw_cfg: dict) -> List[StorageTier]:
    tiers = []
    for tier_key, tier_label in [("hbm", "HBM"), ("dram", "DRAM"), ("ssd", "SSD")]:
        cfg = hw_cfg.get(tier_key, {})
        tiers.append(
            StorageTier(
                name=tier_label,
                capacity_bytes=int(cfg.get("capacity_gb", 80) * 1e9),
                read_bw_gbps=cfg.get("read_bw_gbps", 3200),
                write_bw_gbps=cfg.get("write_bw_gbps", 3200),
                read_latency_ms=cfg.get("read_latency_ms", 0.001),
            )
        )
    return tiers


def _build_eviction(pc: PolicyConfig, future_access_map=None, learned_model=None):
    if pc.eviction_type == "lru":
        return LRUPolicy()
    elif pc.eviction_type == "arc":
        return ARCPolicy()
    elif pc.eviction_type == "learned":
        policy = LearnedPolicy()
        if learned_model is not None:
            policy.set_model(learned_model)
        return policy
    elif pc.eviction_type == "belady":
        return BeladyOracle(future_access_map or {})
    else:
        raise ValueError(f"Unknown eviction type: {pc.eviction_type}")


def _build_prefetch(pc: PolicyConfig):
    if pc.prefetch_type == "none":
        return NoPrefetch()
    elif pc.prefetch_type == "session":
        return SessionAwarePrefetch()
    else:
        raise ValueError(f"Unknown prefetch type: {pc.prefetch_type}")


def _build_workers(
    hw_cfg: dict,
    pc: PolicyConfig,
    num_workers: int,
    future_access_map=None,
    learned_model=None,
) -> List[Worker]:
    workers = []
    for wid in range(num_workers):
        tiers = _build_tiers(hw_cfg)
        eviction = _build_eviction(pc, future_access_map, learned_model)
        prefetch = _build_prefetch(pc)
        cm = CacheManager(
            tiers=tiers,
            eviction_policy=eviction,
            prefetch_policy=prefetch,
            selective_write=pc.selective_write,
            selective_write_depth=pc.selective_write_depth,
        )
        workers.append(Worker(wid, cm))
    return workers


class ExperimentRunner:
    CONFIGS: List[PolicyConfig] = [
        PolicyConfig("Baseline LRU",     "lru",     "none"),
        PolicyConfig("+ARC",             "arc",     "none"),
        PolicyConfig("+SessionPrefetch", "lru",     "session"),
        PolicyConfig("+SelectiveWrite",  "lru",     "none",    selective_write=True),
        PolicyConfig("+Learned",         "learned", "none"),
        PolicyConfig("Belady Oracle",    "belady",  "none"),
    ]

    def __init__(self, config: dict, warmup: int = 200) -> None:
        self.config = config
        self.warmup = warmup
        self._hw_cfg = config.get("hardware", {})
        self._num_workers = config.get("cache", {}).get("num_workers", 4)

    def run_all(
        self,
        requests: List[Request],
        learned_model=None,
    ) -> Dict[str, Metrics]:
        """
        Run every policy configuration against the same request list.
        Uses Router with multiple Workers for prefix-aware routing.
        """
        future_map = _build_future_map(requests)

        results: Dict[str, Metrics] = {}
        for pc in self.CONFIGS:
            print(f"  Running [{pc.name}] …", end=" ", flush=True)
            workers = _build_workers(
                self._hw_cfg, pc, self._num_workers,
                future_access_map=future_map,
                learned_model=learned_model,
            )
            router = Router(workers)
            replayer = TraceReplayer(router, warmup_count=self.warmup)
            metrics = replayer.run(requests)
            results[pc.name] = metrics
            print(f"hit_rate={metrics.hit_rate:.2%}  evictions={metrics.evictions}")

        return results


# ======================================================================
# Helpers
# ======================================================================


def _build_future_map(requests: List[Request]) -> dict:
    """Build {block_hash: [sorted timestamps]} for BeladyOracle."""
    from collections import defaultdict
    m: dict = defaultdict(list)
    for req in requests:
        for bh in req.block_hashes:
            m[bh].append(req.timestamp)
    return {k: sorted(v) for k, v in m.items()}


# ======================================================================
# Cluster-scale experiments
# ======================================================================


class ClusterExperimentRunner:
    """
    Run experiments on the 万卡-scale cluster simulator.

    Compares EIC sizing (no EIC / small / medium / large) and
    eviction policies at cluster scale.
    """

    def __init__(self, config: dict, warmup: int = 500) -> None:
        self.config = config
        self.warmup = warmup

    def run_eic_sizing(self, requests: List[Request]) -> Dict[str, Metrics]:
        """Compare different EIC capacities per rack."""
        from sim.cluster import build_cluster
        from trace.cluster_replay import ClusterReplayer

        configs = [
            ("No EIC (HBM only)", 0.0, 0),
            ("EIC 2×20 MB",       0.02, 2),
            ("EIC 4×20 MB",       0.02, 4),
            ("EIC 4×50 MB",       0.05, 4),
            ("EIC 8×50 MB",       0.05, 8),
        ]

        results: Dict[str, Metrics] = {}
        for name, cap_per_node, n_nodes in configs:
            print(f"  [{name}] …", end=" ", flush=True)
            cfg = _override_eic(self.config, cap_per_node, n_nodes)
            cluster = build_cluster(cfg)
            replayer = ClusterReplayer(cluster, warmup_count=self.warmup)
            metrics = replayer.run(requests)
            cross = cluster.total_cross_gpu_eic_hits
            results[name] = metrics
            print(
                f"hit={metrics.hit_rate:.2%}  "
                f"HBM={metrics.tier_hit_rate('HBM'):.2%}  "
                f"EIC={metrics.tier_hit_rate('EIC'):.2%}  "
                f"xGPU_EIC={cross}"
            )

        return results

    def run_eviction_at_scale(self, requests: List[Request], learned_model=None) -> Dict[str, Metrics]:
        """Compare eviction policies on the cluster."""
        from sim.cluster import build_cluster
        from sim.policies import ARCPolicy, LearnedPolicy, BeladyOracle
        from trace.cluster_replay import ClusterReplayer

        future_map = _build_future_map(requests)
        configs = [
            ("LRU (cluster)",   lambda: LRUPolicy()),
            ("ARC (cluster)",   lambda: ARCPolicy()),
            ("Belady (cluster)", lambda: BeladyOracle(future_map)),
        ]

        if learned_model is not None:
            def _mk_learned():
                p = LearnedPolicy()
                p.set_model(learned_model)
                return p
            configs.insert(2, ("Learned (cluster)", _mk_learned))

        results: Dict[str, Metrics] = {}
        for name, factory in configs:
            print(f"  [{name}] …", end=" ", flush=True)
            cluster = build_cluster(self.config, eviction_factory=factory)
            replayer = ClusterReplayer(cluster, warmup_count=self.warmup)
            metrics = replayer.run(requests)
            results[name] = metrics
            print(
                f"hit={metrics.hit_rate:.2%}  "
                f"EIC={metrics.tier_hit_rate('EIC'):.2%}  "
                f"evict={metrics.evictions}"
            )

        return results


def _override_eic(config: dict, capacity_per_node_gb: float, num_nodes: int) -> dict:
    """Return a copy of config with EIC parameters overridden."""
    import copy
    cfg = copy.deepcopy(config)
    eic = cfg.setdefault("cluster", {}).setdefault("eic", {})
    eic["capacity_per_node_gb"] = capacity_per_node_gb
    eic["nodes_per_rack"] = num_nodes
    return cfg
