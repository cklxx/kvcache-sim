"""
Reporting helpers for simulator runs.

This module intentionally uses only the standard library so report export does
not depend on optional plotting/table packages.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import Metrics
from .pd_metrics import PDMetrics


def metrics_to_dict(metrics: Metrics) -> dict[str, Any]:
    """Serialize cache metrics into a JSON-compatible dictionary."""
    tiers = _tier_names(metrics)
    return {
        "type": "Metrics",
        "tier_names": tiers,
        "total_requests": metrics.total_requests,
        "total_hits": metrics.total_hits,
        "total_misses": metrics.total_misses,
        "hit_rate": _finite(metrics.hit_rate),
        "miss_rate": _finite(metrics.miss_rate),
        "avg_hit_latency_ms": _finite(metrics.avg_hit_latency_ms),
        "total_latency_ms": _finite(metrics.total_latency_ms),
        "events": {
            "evictions": metrics.evictions,
            "promotions": metrics.promotions,
            "demotions": metrics.demotions,
            "prefetches": metrics.prefetches,
        },
        "prefix": {
            "prefix_blocks": metrics.prefix_blocks,
            "prefix_hits": metrics.prefix_hits,
            "prefix_hit_rate": _finite(metrics.prefix_hit_rate),
            "new_blocks": metrics.new_blocks,
        },
        "tiers": {
            tier: _tier_to_dict(metrics, tier)
            for tier in tiers
        },
        "warnings": _metrics_warnings(metrics, tiers),
        "summary": make_json_safe(metrics.summary()),
    }


def pd_metrics_to_dict(metrics: PDMetrics) -> dict[str, Any]:
    """Serialize PD-separated metrics into a JSON-compatible dictionary."""
    return {
        "type": "PDMetrics",
        "total_requests": metrics.total_requests,
        "latency_ms": {
            "ttft": _series_stats(metrics.ttft_ms),
            "tpot": _series_stats(metrics.tpot_ms),
            "e2e": _series_stats(metrics.e2e_ms),
            "prefill_compute": _series_stats(metrics.prefill_compute_ms),
            "transfer": _series_stats(metrics.transfer_ms),
            "queue_wait": _series_stats(metrics.queue_wait_ms),
            "derived": {
                "ttft_avg": _finite(metrics.ttft_avg),
                "ttft_p50": _finite(metrics.ttft_p50),
                "ttft_p99": _finite(metrics.ttft_p99),
                "tpot_avg": _finite(metrics.tpot_avg),
                "tpot_p50": _finite(metrics.tpot_p50),
                "tpot_p99": _finite(metrics.tpot_p99),
                "e2e_p50": _finite(metrics.e2e_p50),
                "e2e_p99": _finite(metrics.e2e_p99),
                "avg_prefill_compute_ms": _finite(metrics.avg_prefill_compute_ms),
                "avg_transfer_ms": _finite(metrics.avg_transfer_ms),
                "avg_queue_wait_ms": _finite(metrics.avg_queue_wait_ms),
            },
        },
        "transfers": {
            "total_kv_bytes_transferred": metrics.total_kv_bytes_transferred,
            "total_kv_gb": _finite(metrics.total_kv_bytes_transferred / 1e9),
            "total_transfers": metrics.total_transfers,
            "same_rack_transfers": metrics.same_rack_transfers,
            "cross_rack_transfers": metrics.cross_rack_transfers,
            "same_rack_ratio": _finite(metrics.same_rack_ratio),
        },
        "prefix_cache": {
            "total_prefix_cache_hits": metrics.total_prefix_cache_hits,
            "total_blocks_processed": metrics.total_blocks_processed,
            "prefix_cache_hit_rate": _finite(metrics.prefix_cache_hit_rate),
        },
        "caches": {
            "prefill": metrics_to_dict(metrics.prefill_cache),
            "decode": metrics_to_dict(metrics.decode_cache),
        },
        "summary": make_json_safe(metrics.summary()),
    }


def results_to_dict(results: Any) -> Any:
    """Serialize nested result objects into JSON-compatible structures."""
    return make_json_safe(results)


def build_report(
    *,
    mode: str,
    args: argparse.Namespace | Mapping[str, Any] | None,
    config_path: str,
    config: Mapping[str, Any],
    results: Any,
    elapsed_seconds: float,
    run_id: str | None = None,
    timestamp: str | None = None,
    argv: list[str] | None = None,
    calibration_summary: Mapping[str, Any] | None = None,
    workload_summary: Mapping[str, Any] | None = None,
    run_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete simulator run report."""
    metadata: dict[str, Any] = {
        "run_id": run_id or uuid.uuid4().hex,
        "mode": mode,
        "timestamp": timestamp or _utc_timestamp(),
        "elapsed_seconds": _finite(elapsed_seconds),
        "config_path": config_path,
        "argv": make_json_safe(sys.argv[1:] if argv is None else argv),
        "args": namespace_to_dict(args),
    }

    seed_info = extract_seed_info(config, mode)
    if seed_info:
        metadata["seed_info"] = seed_info
    if calibration_summary:
        metadata["calibration"] = make_json_safe(calibration_summary)
    if workload_summary:
        metadata["workload"] = make_json_safe(workload_summary)
    if run_summary:
        metadata["run_summary"] = make_json_safe(run_summary)

    return {
        "metadata": metadata,
        "config": make_json_safe(config),
        "results": results_to_dict(results),
    }


def write_json_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a strict JSON report and return the output path."""
    out = _prepare_output_path(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return out


def write_csv_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write one flattened metrics row per section/config result leaf."""
    out = _prepare_output_path(path)
    rows = flatten_result_rows(
        report.get("results", {}),
        metadata=report.get("metadata", {}),
    )
    preferred = [
        "run_id",
        "mode",
        "timestamp",
        "elapsed_seconds",
        "config_path",
        "section",
        "config",
        "variant",
        "path",
        "subpath",
        "type",
    ]
    all_fields = {field for row in rows for field in row}
    fieldnames = preferred + sorted(all_fields - set(preferred))

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
    return out


def flatten_result_rows(
    results: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten serialized or raw result metrics into CSV-friendly rows."""
    serialized = results_to_dict(results)
    meta = metadata or {}
    base = {
        "run_id": meta.get("run_id", ""),
        "mode": meta.get("mode", ""),
        "timestamp": meta.get("timestamp", ""),
        "elapsed_seconds": meta.get("elapsed_seconds", ""),
        "config_path": meta.get("config_path", ""),
    }
    rows: list[dict[str, Any]] = []
    _walk_result_rows(serialized, [], {}, base, rows)
    return rows


def namespace_to_dict(
    args: argparse.Namespace | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return public argparse values as JSON-compatible data."""
    if args is None:
        return {}
    if isinstance(args, argparse.Namespace):
        raw = vars(args)
    elif isinstance(args, Mapping):
        raw = dict(args)
    else:
        return {"value": str(args)}
    return {
        str(key): make_json_safe(value)
        for key, value in raw.items()
        if not str(key).startswith("_")
    }


def extract_seed_info(config: Mapping[str, Any], mode: str | None = None) -> dict[str, Any]:
    """Collect configured seeds and mark the ones used by the active mode."""
    seed_paths = {
        "trace.seed": ("trace", "seed"),
        "cluster_trace.seed": ("cluster_trace", "seed"),
        "pd_trace.seed": ("pd_trace", "seed"),
        "agent_trace.seed": ("agent_trace", "seed"),
        "cluster.network.seed": ("cluster", "network", "seed"),
        "cluster.routing_seed": ("cluster", "routing_seed"),
        "pd_separation.routing_seed": ("pd_separation", "routing_seed"),
    }
    available: dict[str, Any] = {}
    for label, path in seed_paths.items():
        value = _deep_get(config, path)
        if value is not None:
            available[label] = make_json_safe(value)

    if not available:
        return {}

    active_labels = {
        "single_node": ("trace.seed",),
        "cluster": (
            "cluster_trace.seed",
            "cluster.network.seed",
            "cluster.routing_seed",
        ),
        "pd": (
            "pd_trace.seed",
            "cluster.network.seed",
            "pd_separation.routing_seed",
        ),
    }.get(mode or "", ())
    active = {label: available[label] for label in active_labels if label in available}

    result: dict[str, Any] = {"available": available}
    if active:
        result["active"] = active
    return result


def make_json_safe(value: Any) -> Any:
    """Convert supported Python objects into strict JSON-compatible values."""
    if isinstance(value, Metrics):
        return metrics_to_dict(value)
    if isinstance(value, PDMetrics):
        return pd_metrics_to_dict(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    return str(value)


def _tier_names(metrics: Metrics) -> list[str]:
    seen: set[str] = set()
    tiers: list[str] = []
    for source in (
        metrics.tier_names,
        metrics.tier_hits.keys(),
        metrics.tier_latency_ms.keys(),
        metrics.tier_used_bytes.keys(),
        metrics.tier_capacity_bytes.keys(),
        metrics.tier_blocks.keys(),
    ):
        for tier in source:
            if tier not in seen:
                seen.add(tier)
                tiers.append(tier)
    return tiers


def _tier_to_dict(metrics: Metrics, tier: str) -> dict[str, Any]:
    hits = metrics.tier_hits.get(tier, 0)
    latency_ms = metrics.tier_latency_ms.get(tier, 0.0)
    avg_latency = latency_ms / hits if hits else 0.0
    capacity = metrics.tier_capacity_bytes.get(tier, 0)
    used = metrics.tier_used_bytes.get(tier, 0)
    utilization = used / capacity if capacity > 0 else 0.0
    return {
        "hits": hits,
        "hit_rate": _finite(metrics.tier_hit_rate(tier)),
        "total_latency_ms": _finite(latency_ms),
        "avg_latency_ms": _finite(avg_latency),
        "used_bytes": used,
        "capacity_bytes": capacity,
        "utilization": _finite(utilization),
        "blocks": metrics.tier_blocks.get(tier, 0),
    }


def _metrics_warnings(metrics: Metrics, tiers: list[str]) -> list[str]:
    warnings: list[str] = []
    for tier in tiers:
        capacity = metrics.tier_capacity_bytes.get(tier, 0)
        used = metrics.tier_used_bytes.get(tier, 0)
        if capacity > 0 and used > capacity:
            warnings.append(
                f"{tier} storage usage exceeds capacity "
                f"({used / capacity:.2f}x); inspect cache capacity model."
            )
    return warnings


def _series_stats(values: list[float]) -> dict[str, Any]:
    finite_values: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite_values.append(number)
    if not finite_values:
        return {
            "count": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(finite_values),
        "avg": _finite(sum(finite_values) / len(finite_values)),
        "min": _finite(min(finite_values)),
        "max": _finite(max(finite_values)),
        "p50": _finite(_percentile(finite_values, 50)),
        "p95": _finite(_percentile(finite_values, 95)),
        "p99": _finite(_percentile(finite_values, 99)),
    }


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def _walk_result_rows(
    node: Any,
    path: list[str],
    context: dict[str, Any],
    base: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if _is_metric_node(node):
        row = dict(base)
        row.update(_path_columns(path))
        row.update(context)
        row.update(_flatten_scalars(node))
        rows.append(row)
        return

    if isinstance(node, Mapping):
        scalars = {
            f"context.{key}": value
            for key, value in node.items()
            if _is_csv_scalar(value)
        }
        child_context = {**context, **scalars}
        for key, value in node.items():
            if _is_csv_scalar(value):
                continue
            _walk_result_rows(value, path + [str(key)], child_context, base, rows)


def _path_columns(path: list[str]) -> dict[str, str]:
    return {
        "section": path[0] if path else "",
        "config": path[1] if len(path) > 1 else (path[0] if path else ""),
        "variant": path[2] if len(path) > 2 else "",
        "path": ".".join(path),
        "subpath": ".".join(path[3:]) if len(path) > 3 else "",
    }


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_scalars(child, child_prefix))
        return flattened
    if _is_csv_scalar(value):
        return {prefix: value}
    return {prefix: json.dumps(make_json_safe(value), sort_keys=True)}


def _is_metric_node(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("type") in {"Metrics", "PDMetrics"}


def _is_csv_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_csv_scalar(item) for item in value)
    return False


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(make_json_safe(value), sort_keys=True)
    return value


def _prepare_output_path(path: str | Path) -> Path:
    out = Path(path)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _deep_get(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
