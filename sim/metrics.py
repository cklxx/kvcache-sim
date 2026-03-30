"""
Metrics — lightweight counters and a matplotlib visualiser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Metrics:
    tier_names: List[str] = field(default_factory=list)

    # Request counts
    total_requests: int = 0
    total_hits: int = 0
    total_misses: int = 0

    # Per-tier hits and latency
    tier_hits: Dict[str, int] = field(default_factory=dict)
    tier_latency_ms: Dict[str, float] = field(default_factory=dict)

    # System events
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    prefetches: int = 0

    # Total latency (hits only)
    total_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in self.tier_names:
            self.tier_hits.setdefault(name, 0)
            self.tier_latency_ms.setdefault(name, 0.0)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_hit(self, tier_name: str, latency_ms: float) -> None:
        self.total_hits += 1
        self.tier_hits[tier_name] = self.tier_hits.get(tier_name, 0) + 1
        self.tier_latency_ms[tier_name] = self.tier_latency_ms.get(tier_name, 0.0) + latency_ms
        self.total_latency_ms += latency_ms

    def record_miss(self) -> None:
        self.total_misses += 1

    # ------------------------------------------------------------------
    # Derived KPIs
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

    def tier_hit_rate(self, tier_name: str) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.tier_hits.get(tier_name, 0) / self.total_requests

    @property
    def avg_hit_latency_ms(self) -> float:
        if self.total_hits == 0:
            return 0.0
        return self.total_latency_ms / self.total_hits

    def summary(self) -> Dict:
        d: Dict = {
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 4),
            "evictions": self.evictions,
            "promotions": self.promotions,
            "prefetches": self.prefetches,
        }
        for t in self.tier_names:
            d[f"hit_rate_{t}"] = round(self.tier_hit_rate(t), 4)
        return d

    def __repr__(self) -> str:
        return (
            f"Metrics(requests={self.total_requests}, "
            f"hit_rate={self.hit_rate:.2%}, "
            f"avg_latency={self.avg_hit_latency_ms:.3f}ms)"
        )


# ======================================================================
# Visualiser (matplotlib)
# ======================================================================


def plot_comparison(
    results: Dict[str, Metrics],
    tier_names: List[str],
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Draw a multi-panel comparison figure for a dict of
    {policy_name: Metrics}.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[metrics] matplotlib not installed — skipping plot.")
        return

    policy_names = list(results.keys())
    n = len(policy_names)
    x = np.arange(n)
    colors = plt.cm.tab10.colors  # type: ignore

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("KV-Cache Multi-Tier Simulator — Policy Comparison", fontsize=14, fontweight="bold")

    # ── Panel 1: overall hit rate ────────────────────────────────────
    ax = axes[0, 0]
    hit_rates = [results[p].hit_rate for p in policy_names]
    bars = ax.bar(x, [h * 100 for h in hit_rates], color=colors[:n])
    ax.set_title("Overall Cache Hit Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    for bar, v in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1%}", ha="center", va="bottom", fontsize=7)

    # ── Panel 2: per-tier hit rate stacked bar ───────────────────────
    ax = axes[0, 1]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bottoms = np.zeros(n)
    for i, tier in enumerate(tier_names):
        vals = np.array([results[p].tier_hit_rate(tier) * 100 for p in policy_names])
        ax.bar(x, vals, bottom=bottoms, label=tier, color=tier_colors[i % len(tier_colors)])
        bottoms += vals
    ax.set_title("Hit Rate by Tier (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

    # ── Panel 3: average hit latency ────────────────────────────────
    ax = axes[1, 0]
    latencies = [results[p].avg_hit_latency_ms for p in policy_names]
    bars = ax.bar(x, latencies, color=colors[:n])
    ax.set_title("Average Hit Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ms")
    for bar, v in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(latencies) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Panel 4: evictions and prefetches ───────────────────────────
    ax = axes[1, 1]
    evictions = [results[p].evictions for p in policy_names]
    prefetches = [results[p].prefetches for p in policy_names]
    w = 0.35
    ax.bar(x - w / 2, evictions, w, label="Evictions", color="#E53935")
    ax.bar(x + w / 2, prefetches, w, label="Prefetches", color="#43A047")
    ax.set_title("Evictions vs Prefetches")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] Figure saved → {output_path}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
