"""
Cluster diagnostics for post-replay sanity checks.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

from .cluster import Cluster


def summarize_cluster_health(
    cluster: Cluster,
    skew_ratio_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Return a compact health summary for a replayed cluster.

    Request counts are request-level GPUNode.process_request() counts, not the
    block-level Metrics.total_requests counters.
    """
    if skew_ratio_threshold <= 0.0:
        raise ValueError("skew_ratio_threshold must be positive")

    gpu_counts = _request_counts_by_gpu(cluster)
    rack_counts = _request_counts_by_rack(cluster, gpu_counts)

    active_gpus = (
        sum(1 for count in gpu_counts.values() if count > 0)
        if gpu_counts is not None
        else 0
    )
    active_racks = (
        sum(1 for count in rack_counts.values() if count > 0)
        if rack_counts is not None
        else 0
    )

    gpu_skew = _skew_details(gpu_counts, skew_ratio_threshold, "gpu_id")
    rack_skew = _skew_details(rack_counts, skew_ratio_threshold, "rack_id")

    messages = []
    if gpu_skew["warning"]:
        messages.append(
            "GPU request skew: max GPU "
            f"{gpu_skew['hot_id']} handled {gpu_skew['max_count']} requests "
            f"({gpu_skew['max_to_mean']:.2f}x mean)."
        )
    if rack_skew["warning"]:
        messages.append(
            "Rack request skew: max rack "
            f"{rack_skew['hot_id']} handled {rack_skew['max_count']} requests "
            f"({rack_skew['max_to_mean']:.2f}x mean)."
        )

    return {
        "total_racks": cluster.total_racks,
        "total_gpus": cluster.total_gpus,
        "active_racks": active_racks,
        "active_gpus": active_gpus,
        "request_distribution": _request_distribution(gpu_counts, rack_counts),
        "utilization": {
            "hbm": _summary(gpu.hbm.utilization for gpu in cluster.all_gpus),
            "eic": _summary(rack.eic.utilization for rack in cluster.racks),
        },
        "eic_cross_gpu_hits": cluster.total_cross_gpu_eic_hits,
        "warnings": {
            "request_skew": bool(gpu_skew["warning"] or rack_skew["warning"]),
            "gpu_request_skew": gpu_skew["warning"],
            "rack_request_skew": rack_skew["warning"],
            "inactive_gpus": cluster.total_gpus - active_gpus,
            "inactive_racks": cluster.total_racks - active_racks,
            "gpu_hot_id": gpu_skew["hot_id"],
            "gpu_hot_requests": gpu_skew["max_count"],
            "rack_hot_id": rack_skew["hot_id"],
            "rack_hot_requests": rack_skew["max_count"],
            "gpu_max_to_mean": gpu_skew["max_to_mean"],
            "rack_max_to_mean": rack_skew["max_to_mean"],
            "skew_ratio_threshold": skew_ratio_threshold,
            "messages": messages,
        },
    }


def _request_distribution(
    gpu_counts: Optional[Dict[int, int]],
    rack_counts: Optional[Dict[int, int]],
) -> Dict[str, Any]:
    if gpu_counts is None or rack_counts is None:
        return {"available": False}

    return {
        "available": True,
        "total_processed_requests": sum(gpu_counts.values()),
        "per_gpu": gpu_counts,
        "per_rack": rack_counts,
        "gpu_summary": _summary(gpu_counts.values()),
        "rack_summary": _summary(rack_counts.values()),
    }


def _request_counts_by_gpu(cluster: Cluster) -> Optional[Dict[int, int]]:
    getter = getattr(cluster, "request_counts_by_gpu", None)
    if callable(getter):
        return {int(gpu_id): int(count) for gpu_id, count in getter().items()}

    counts: Dict[int, int] = {}
    for idx, gpu in enumerate(getattr(cluster, "all_gpus", [])):
        count = getattr(gpu, "processed_requests", None)
        if count is None:
            return None
        gpu_id = int(getattr(gpu, "gpu_id", idx))
        counts[gpu_id] = int(count)
    return counts


def _request_counts_by_rack(
    cluster: Cluster,
    gpu_counts: Optional[Mapping[int, int]],
) -> Optional[Dict[int, int]]:
    getter = getattr(cluster, "request_counts_by_rack", None)
    if callable(getter):
        return {int(rack_id): int(count) for rack_id, count in getter().items()}

    if gpu_counts is None:
        return None

    counts: Dict[int, int] = {}
    for rack in getattr(cluster, "racks", []):
        counts[int(rack.rack_id)] = sum(
            int(gpu_counts.get(int(gpu.gpu_id), 0)) for gpu in rack.gpu_nodes
        )
    return counts


def _summary(values: Iterable[float]) -> Dict[str, float]:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    return {
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 50.0),
        "p95": _percentile(sorted_values, 95.0),
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]

    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _skew_details(
    counts: Optional[Mapping[int, int]],
    threshold: float,
    hot_key: str,
) -> Dict[str, Any]:
    if not counts:
        return {
            "warning": False,
            "hot_id": None,
            "hot_key": hot_key,
            "max_count": 0,
            "mean": 0.0,
            "max_to_mean": 0.0,
        }

    total = sum(counts.values())
    mean = total / len(counts) if counts else 0.0
    hot_id, max_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    max_to_mean = (max_count / mean) if mean > 0.0 else 0.0
    warning = total >= len(counts) and max_count > 0 and max_to_mean >= threshold

    return {
        "warning": warning,
        "hot_id": hot_id,
        "hot_key": hot_key,
        "max_count": max_count,
        "mean": mean,
        "max_to_mean": max_to_mean,
    }
