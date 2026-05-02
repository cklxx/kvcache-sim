"""
Trace validation helpers.

The validator is intentionally independent from the replay loop. It reports
trace-shape metrics that are useful before running expensive simulations and
flags inputs that are likely to produce misleading cache or calibration results.
"""
from __future__ import annotations

import math
import warnings as warnings_module
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from trace.generator import Request
from trace.workload import WorkloadTrace


@dataclass(frozen=True)
class TraceValidationThresholds:
    tiny_request_count: int = 100
    skipped_row_ratio: float = 0.05
    low_session_reuse_ratio: float = 0.05
    high_zero_output_ratio: float = 0.50


def validate_workload_trace(
    trace: WorkloadTrace | Sequence[Request],
    *,
    thresholds: TraceValidationThresholds | None = None,
    emit_warnings: bool = False,
) -> dict[str, Any]:
    """
    Return validation metrics and warnings for a workload trace.

    ``trace`` may be a loaded :class:`WorkloadTrace` or a plain request list.
    For plain request lists, source-row and external-hash metadata are unknown,
    so ``skipped_rows`` is ``None`` and hash-ID coverage is reported as zero.
    """
    thresholds = thresholds or TraceValidationThresholds()
    requests, meta = _coerce_trace(trace)

    request_count = len(requests)
    session_counts: Counter[str] = Counter(r.session_id for r in requests)
    session_count = len(session_counts)
    max_turns_per_session = max(session_counts.values(), default=0)

    timestamps = [float(r.timestamp) for r in requests]
    duration_ms = max(timestamps) - min(timestamps) if timestamps else 0.0
    rps = request_count / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

    prompts = [int(r.prompt_tokens) for r in requests]
    outputs = [int(r.output_tokens) for r in requests]
    zero_output_count = sum(1 for value in outputs if value <= 0)
    zero_output_ratio = zero_output_count / request_count if request_count else 0.0

    block_counts: Counter[str] = Counter()
    for request in requests:
        block_counts.update(str(block) for block in request.block_hashes)
    total_block_refs = sum(block_counts.values())
    unique_block_count = len(block_counts)
    repeated_block_ratio = (
        (total_block_refs - unique_block_count) / total_block_refs
        if total_block_refs
        else 0.0
    )

    hash_backed_requests = min(meta["hash_backed_requests"], request_count)
    hash_id_coverage = hash_backed_requests / request_count if request_count else 0.0
    skipped_rows = meta["skipped_rows"]
    source_rows = meta["source_rows"]
    skipped_row_ratio = (
        skipped_rows / source_rows
        if isinstance(skipped_rows, int) and source_rows
        else None
    )
    session_reuse_ratio = (
        (request_count - session_count) / request_count if request_count else 0.0
    )

    report: dict[str, Any] = {
        "request_count": request_count,
        "session_count": session_count,
        "duration_ms": duration_ms,
        "rps": rps,
        "hash_id_coverage": hash_id_coverage,
        "hash_backed": hash_id_coverage > 0.0,
        "prompt_p95": _percentile(prompts, 95),
        "output_p95": _percentile(outputs, 95),
        "skipped_rows": skipped_rows,
        "unique_block_count": unique_block_count,
        "repeated_block_ratio": repeated_block_ratio,
        "max_turns_per_session": max_turns_per_session,
        "session_reuse_ratio": session_reuse_ratio,
        "zero_output_ratio": zero_output_ratio,
        "warnings": [],
    }
    if source_rows is not None:
        report["source_rows"] = source_rows
    if skipped_row_ratio is not None:
        report["skipped_row_ratio"] = skipped_row_ratio

    validation_warnings = _build_warnings(
        report=report,
        thresholds=thresholds,
        skipped_row_ratio=skipped_row_ratio,
    )
    report["warnings"] = validation_warnings

    if emit_warnings:
        for message in validation_warnings:
            warnings_module.warn(message, RuntimeWarning, stacklevel=2)

    return report


def _coerce_trace(
    trace: WorkloadTrace | Sequence[Request],
) -> tuple[list[Request], dict[str, Any]]:
    if isinstance(trace, WorkloadTrace):
        requests = list(trace.requests)
        hash_backed_requests = trace.hash_backed_requests
        if hash_backed_requests == 0 and trace.used_hash_ids:
            hash_backed_requests = len(requests)
        return requests, {
            "source_rows": trace.source_rows,
            "skipped_rows": trace.skipped_rows,
            "hash_backed_requests": hash_backed_requests,
        }
    return list(trace), {
        "source_rows": None,
        "skipped_rows": None,
        "hash_backed_requests": 0,
    }


def _build_warnings(
    *,
    report: dict[str, Any],
    thresholds: TraceValidationThresholds,
    skipped_row_ratio: float | None,
) -> list[str]:
    warnings: list[str] = []
    request_count = report["request_count"]

    if request_count < thresholds.tiny_request_count:
        warnings.append(
            f"Trace has a tiny request count ({request_count}); "
            "production conclusions may be noisy."
        )
    if report["duration_ms"] <= 0:
        warnings.append(
            "Trace duration is non-positive; request rate cannot be calibrated."
        )
    if report["hash_id_coverage"] <= 0:
        warnings.append(
            "Trace has no external hash IDs; cache reuse depends on synthetic block hashes."
        )
    elif report["hash_id_coverage"] < 1.0:
        pct = report["hash_id_coverage"] * 100.0
        warnings.append(f"Trace has partial hash ID coverage ({pct:.1f}%).")
    if (
        skipped_row_ratio is not None
        and skipped_row_ratio > thresholds.skipped_row_ratio
    ):
        pct = skipped_row_ratio * 100.0
        warnings.append(
            f"Trace skipped {report['skipped_rows']} source rows ({pct:.1f}%)."
        )
    if (
        request_count > 1
        and report["session_reuse_ratio"] < thresholds.low_session_reuse_ratio
    ):
        pct = report["session_reuse_ratio"] * 100.0
        warnings.append(
            f"Trace has very low session reuse ({pct:.1f}% repeated-session requests)."
        )
    if request_count and report["zero_output_ratio"] >= 1.0:
        warnings.append(
            "Trace has zero or unknown output tokens for every request; "
            "decode calibration may be invalid."
        )
    elif report["zero_output_ratio"] > thresholds.high_zero_output_ratio:
        pct = report["zero_output_ratio"] * 100.0
        warnings.append(
            f"Trace has many zero or unknown output token counts ({pct:.1f}%)."
        )

    return warnings


def _percentile(values: list[int], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight
