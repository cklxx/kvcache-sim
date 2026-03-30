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

import copy
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


def _build_manager(
    hw_cfg: dict,
    pc: PolicyConfig,
    future_access_map: Optional[dict] = None,
    learned_model=None,
) -> CacheManager:
    tiers = _build_tiers(hw_cfg)

    # Eviction
    if pc.eviction_type == "lru":
        eviction = LRUPolicy()
    elif pc.eviction_type == "arc":
        eviction = ARCPolicy()
    elif pc.eviction_type == "learned":
        eviction = LearnedPolicy()
        if learned_model is not None:
            eviction.set_model(learned_model)
    elif pc.eviction_type == "belady":
        eviction = BeladyOracle(future_access_map or {})
    else:
        raise ValueError(f"Unknown eviction type: {pc.eviction_type}")

    # Prefetch
    if pc.prefetch_type == "none":
        prefetch = NoPrefetch()
    elif pc.prefetch_type == "session":
        prefetch = SessionAwarePrefetch()
    else:
        raise ValueError(f"Unknown prefetch type: {pc.prefetch_type}")

    return CacheManager(
        tiers=tiers,
        eviction_policy=eviction,
        prefetch_policy=prefetch,
        selective_write=pc.selective_write,
        selective_write_depth=pc.selective_write_depth,
    )


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

    def run_all(
        self,
        requests: List[Request],
        learned_model=None,
    ) -> Dict[str, Metrics]:
        """
        Run every policy configuration against the same request list.

        Returns
        -------
        dict mapping policy name → Metrics
        """
        # Pre-build Belady future map once (shared across oracle runs)
        future_map = _build_future_map(requests)

        results: Dict[str, Metrics] = {}
        for pc in self.CONFIGS:
            print(f"  Running [{pc.name}] …", end=" ", flush=True)
            manager = _build_manager(
                self._hw_cfg,
                pc,
                future_access_map=future_map,
                learned_model=learned_model,
            )
            replayer = TraceReplayer(manager, warmup_count=self.warmup)
            metrics = replayer.run(requests)
            results[pc.name] = metrics
            print(f"hit_rate={metrics.hit_rate:.2%}  evictions={metrics.evictions}")

        return results


# ======================================================================
# Helpers
# ======================================================================


def _build_future_map(requests: List[Request]) -> dict:
    """
    Build {block_hash: [sorted timestamps]} for BeladyOracle.
    """
    from collections import defaultdict
    m: dict = defaultdict(list)
    for req in requests:
        for bh in req.block_hashes:
            m[bh].append(req.timestamp)
    return {k: sorted(v) for k, v in m.items()}
