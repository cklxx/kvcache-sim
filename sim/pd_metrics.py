"""
PDMetrics — phase-specific metrics for PD-separated serving.

Tracks TTFT, TPOT, KV transfer overhead, prefix cache hit rate,
queue depths, and per-pool utilisation with percentile support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import Metrics


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


@dataclass
class PDMetrics:
    """Aggregated metrics for PD-separated simulation."""

    # Cache metrics per pool
    prefill_cache: Metrics = field(default_factory=lambda: Metrics(tier_names=["HBM", "EIC", "Remote"]))
    decode_cache: Metrics = field(default_factory=lambda: Metrics(tier_names=["HBM", "EIC", "Remote"]))

    # Timing distributions (ms)
    ttft_ms: List[float] = field(default_factory=list)
    tpot_ms: List[float] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)
    prefill_compute_ms: List[float] = field(default_factory=list)
    transfer_ms: List[float] = field(default_factory=list)
    queue_wait_ms: List[float] = field(default_factory=list)

    # Transfer stats
    total_kv_bytes_transferred: int = 0
    total_transfers: int = 0
    same_rack_transfers: int = 0
    cross_rack_transfers: int = 0

    # Prefix sharing stats
    total_prefix_cache_hits: int = 0
    total_blocks_processed: int = 0

    # Request count
    total_requests: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(self, result) -> None:
        """Record a PDStepResult."""
        self.total_requests += 1
        self.ttft_ms.append(result.ttft_ms)
        self.e2e_ms.append(result.e2e_ms)
        self.prefill_compute_ms.append(result.prefill_compute_ms)
        self.transfer_ms.append(result.kv_transfer_ms)
        self.queue_wait_ms.append(result.prefill_queue_wait_ms)

        if result.output_tokens > 0:
            self.tpot_ms.append(result.decode_total_ms / result.output_tokens)

        self.total_kv_bytes_transferred += result.kv_bytes_transferred
        self.total_transfers += 1
        if result.same_rack:
            self.same_rack_transfers += 1
        else:
            self.cross_rack_transfers += 1

        self.total_prefix_cache_hits += result.prefix_hit_blocks
        self.total_blocks_processed += result.prefix_hit_blocks + result.new_computed_blocks

    # ------------------------------------------------------------------
    # Derived KPIs
    # ------------------------------------------------------------------

    @property
    def ttft_p50(self) -> float:
        return _percentile(self.ttft_ms, 50)

    @property
    def ttft_p99(self) -> float:
        return _percentile(self.ttft_ms, 99)

    @property
    def ttft_avg(self) -> float:
        return sum(self.ttft_ms) / len(self.ttft_ms) if self.ttft_ms else 0.0

    @property
    def tpot_avg(self) -> float:
        return sum(self.tpot_ms) / len(self.tpot_ms) if self.tpot_ms else 0.0

    @property
    def tpot_p50(self) -> float:
        return _percentile(self.tpot_ms, 50)

    @property
    def tpot_p99(self) -> float:
        return _percentile(self.tpot_ms, 99)

    @property
    def e2e_p50(self) -> float:
        return _percentile(self.e2e_ms, 50)

    @property
    def e2e_p99(self) -> float:
        return _percentile(self.e2e_ms, 99)

    @property
    def avg_prefill_compute_ms(self) -> float:
        return sum(self.prefill_compute_ms) / len(self.prefill_compute_ms) if self.prefill_compute_ms else 0.0

    @property
    def avg_transfer_ms(self) -> float:
        return sum(self.transfer_ms) / len(self.transfer_ms) if self.transfer_ms else 0.0

    @property
    def avg_queue_wait_ms(self) -> float:
        return sum(self.queue_wait_ms) / len(self.queue_wait_ms) if self.queue_wait_ms else 0.0

    @property
    def prefix_cache_hit_rate(self) -> float:
        if self.total_blocks_processed == 0:
            return 0.0
        return self.total_prefix_cache_hits / self.total_blocks_processed

    @property
    def same_rack_ratio(self) -> float:
        if self.total_transfers == 0:
            return 0.0
        return self.same_rack_transfers / self.total_transfers

    def summary(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "ttft_p50_ms": round(self.ttft_p50, 3),
            "ttft_p99_ms": round(self.ttft_p99, 3),
            "tpot_avg_ms": round(self.tpot_avg, 3),
            "e2e_p50_ms": round(self.e2e_p50, 3),
            "avg_prefill_ms": round(self.avg_prefill_compute_ms, 3),
            "avg_transfer_ms": round(self.avg_transfer_ms, 3),
            "avg_queue_ms": round(self.avg_queue_wait_ms, 3),
            "prefix_cache_hit_rate": round(self.prefix_cache_hit_rate, 4),
            "same_rack_ratio": round(self.same_rack_ratio, 4),
            "total_kv_gb": round(self.total_kv_bytes_transferred / 1e9, 3),
        }

    def __repr__(self) -> str:
        return (
            f"PDMetrics(reqs={self.total_requests}, "
            f"TTFT_p50={self.ttft_p50:.1f}ms, "
            f"TPOT={self.tpot_avg:.1f}ms, "
            f"prefix_hit={self.prefix_cache_hit_rate:.1%})"
        )
