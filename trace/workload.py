"""
External workload trace loader.

The public production traces used by LLM serving papers mostly expose arrival
time plus input/output token counts, not prompt text. This module converts
those records into the Request shape used by the simulator. If a trace includes
prefix/block IDs, such as Mooncake hash_ids, those IDs are preserved. Otherwise
the loader synthesizes deterministic block IDs from session IDs when available
and falls back to per-request unique blocks.
"""
from __future__ import annotations

import ast
import csv
import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from trace.generator import Request


TIMESTAMP_ALIASES = (
    "timestamp",
    "time",
    "arrival_timestamp",
    "arrival_time",
    "request_time",
    "timems",
)
PROMPT_ALIASES = (
    "request_tokens",
    "request tokens",
    "contexttokens",
    "context_tokens",
    "prompt_tokens",
    "prompt_size",
    "input_tokens",
    "input_length",
)
OUTPUT_ALIASES = (
    "response_tokens",
    "response tokens",
    "generatedtokens",
    "generated_tokens",
    "completion_tokens",
    "output_tokens",
    "output_length",
    "token_size",
)
SESSION_ALIASES = (
    "session_id",
    "session id",
    "conversation_id",
    "conversationid",
    "session",
)
MODEL_ALIASES = ("model", "model_name", "modelname")
HASH_ALIASES = ("hash_ids", "hash ids", "block_hashes", "block hashes")

DEFAULT_HASH_TOKENS_PER_BLOCK = {
    "mooncake": 512,
}


@dataclass
class WorkloadTrace:
    requests: list[Request]
    format_name: str
    source_rows: int
    skipped_rows: int = 0
    used_hash_ids: bool = False
    hash_tokens_per_block: int | None = None


def load_workload_trace(
    path: str | Path,
    *,
    block_size_bytes: int,
    kv_bytes_per_token: int | None = None,
    tokens_per_block: int = 16,
    format_name: str = "auto",
    limit: int | None = None,
    timestamp_unit: str = "auto",
    arrival_scale: float = 1.0,
    include_failed: bool = False,
    hash_tokens_per_block: int | None = None,
) -> WorkloadTrace:
    """
    Load a production-style workload file as simulator Requests.

    Supported inputs:
    - BurstGPT CSV: Timestamp, Session ID, Model, Request tokens, Response tokens
    - Azure LLM CSV: TIMESTAMP, ContextTokens, GeneratedTokens
    - SplitwiseSim CSV: arrival_timestamp, prompt_size, token_size
    - Mooncake-style CSV/JSONL: timestamp, input_length, output_length, hash_ids
    - Generic CSV/JSONL with recognizable aliases listed above

    ``arrival_scale`` scales request rate. A value of 2.0 halves inter-arrival
    time, while 0.5 doubles it.
    """
    if arrival_scale <= 0:
        raise ValueError("arrival_scale must be > 0")
    if tokens_per_block <= 0:
        raise ValueError("tokens_per_block must be > 0")

    path = Path(path)
    inferred_format: str | None = None
    hash_block_tokens: int | None = None
    raw_records: list[dict[str, Any]] = []
    skipped = 0
    source_rows = 0
    used_hash_ids = False
    session_turns: dict[str, int] = {}

    for source_idx, row in enumerate(_iter_rows(path)):
        source_rows += 1
        if inferred_format is None:
            inferred_format = (
                _infer_format(row) if format_name == "auto" else format_name
            ).lower()
            hash_block_tokens = hash_tokens_per_block
            if hash_block_tokens is None:
                hash_block_tokens = DEFAULT_HASH_TOKENS_PER_BLOCK.get(
                    inferred_format, tokens_per_block
                )

        timestamp_raw = _get(row, TIMESTAMP_ALIASES)
        prompt_raw = _get(row, PROMPT_ALIASES)
        output_raw = _get(row, OUTPUT_ALIASES)
        hash_raw = _get(row, HASH_ALIASES)

        if timestamp_raw in (None, "") or prompt_raw in (None, ""):
            skipped += 1
            continue

        try:
            prompt_tokens = _parse_int(prompt_raw)
            output_tokens = _parse_int(output_raw) if output_raw not in (None, "") else 0
            timestamp = _parse_timestamp(
                timestamp_raw,
                _effective_timestamp_unit(timestamp_unit, inferred_format),
            )
        except (TypeError, ValueError):
            skipped += 1
            continue

        if prompt_tokens <= 0:
            skipped += 1
            continue
        if output_raw not in (None, "") and output_tokens <= 0 and not include_failed:
            skipped += 1
            continue

        session_id = str(_get(row, SESSION_ALIASES) or "").strip()
        if not session_id:
            session_id = f"request_{source_idx:08d}"
        model = str(_get(row, MODEL_ALIASES) or "").strip()

        if hash_raw not in (None, ""):
            hash_ids = _parse_hash_ids(hash_raw)
            block_hashes = [f"{inferred_format}:h:{h}" for h in hash_ids]
            if not block_hashes:
                skipped += 1
                continue
            used_hash_ids = True
            request_block_size = _block_size_for_hash_ids(
                block_size_bytes,
                kv_bytes_per_token,
                hash_block_tokens or tokens_per_block,
            )
        else:
            n_blocks = max(1, math.ceil(prompt_tokens / tokens_per_block))
            block_hashes = _synthetic_block_hashes(
                source_idx=source_idx,
                session_id=session_id,
                model=model,
                num_blocks=n_blocks,
            )
            request_block_size = block_size_bytes

        turn_id = session_turns.get(session_id, 0)
        session_turns[session_id] = turn_id + 1

        raw_records.append(
            {
                "timestamp": timestamp,
                "request": Request(
                    session_id=session_id,
                    turn_id=turn_id,
                    timestamp=timestamp,
                    block_hashes=block_hashes,
                    block_size=request_block_size,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                ),
            }
        )
        if limit is not None and len(raw_records) >= limit:
            break

    inferred_format = inferred_format or "unknown"
    if not raw_records:
        return WorkloadTrace([], inferred_format, source_rows, skipped)

    raw_records.sort(key=lambda r: r["timestamp"])
    t0 = raw_records[0]["timestamp"]
    requests: list[Request] = []
    for record in raw_records:
        req = record["request"]
        req.timestamp = round(((record["timestamp"] - t0) * 1000.0) / arrival_scale, 6)
        requests.append(req)

    return WorkloadTrace(
        requests=requests,
        format_name=inferred_format,
        source_rows=source_rows,
        skipped_rows=skipped,
        used_hash_ids=used_hash_ids,
        hash_tokens_per_block=hash_block_tokens if used_hash_ids else None,
    )


def summarize_workload(trace: WorkloadTrace) -> dict[str, Any]:
    requests = trace.requests
    if not requests:
        return {
            "format": trace.format_name,
            "requests": 0,
            "skipped_rows": trace.skipped_rows,
        }

    prompts = [r.prompt_tokens for r in requests]
    outputs = [r.output_tokens for r in requests if r.output_tokens > 0]
    duration_ms = requests[-1].timestamp - requests[0].timestamp
    sessions = len({r.session_id for r in requests})

    summary: dict[str, Any] = {
        "format": trace.format_name,
        "requests": len(requests),
        "sessions": sessions,
        "duration_ms": duration_ms,
        "rps": len(requests) / (duration_ms / 1000.0) if duration_ms > 0 else float("inf"),
        "prompt_avg": mean(prompts),
        "prompt_p50": median(prompts),
        "prompt_p95": _percentile(prompts, 95),
        "skipped_rows": trace.skipped_rows,
        "hash_backed": trace.used_hash_ids,
    }
    if outputs:
        summary.update(
            {
                "output_avg": mean(outputs),
                "output_p50": median(outputs),
                "output_p95": _percentile(outputs, 95),
            }
        )
    if trace.hash_tokens_per_block:
        summary["hash_tokens_per_block"] = trace.hash_tokens_per_block
    return summary


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    opener = gzip.open if suffixes and suffixes[-1] == ".gz" else open
    if ".jsonl" in suffixes:
        with opener(path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if ".json" in suffixes:
        with opener(path, "rt") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for row in payload:
                yield row
        elif isinstance(payload, dict):
            rows = payload.get("data") or payload.get("rows") or payload.get("requests")
            if not isinstance(rows, list):
                raise ValueError("JSON workload must be a list or contain data/rows/requests")
            for row in rows:
                yield row
        else:
            raise ValueError("JSON workload must be a list or mapping")
        return

    with opener(path, "rt", newline="") as f:
        yield from csv.DictReader(f)


def _infer_format(row: Mapping[str, Any]) -> str:
    keys = {_normalise_key(k) for k in row}
    if {"requesttokens", "responsetokens"}.issubset(keys):
        return "burstgpt"
    if {"contexttokens", "generatedtokens"}.issubset(keys):
        return "azure"
    if {"inputlength", "outputlength", "hashids"}.issubset(keys):
        return "mooncake"
    if {"promptsize", "tokensize", "arrivaltimestamp"}.issubset(keys):
        return "splitwise"
    return "generic"


def _get(row: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    lookup = {_normalise_key(k): v for k, v in row.items()}
    for alias in aliases:
        key = _normalise_key(alias)
        if key in lookup:
            return lookup[key]
    return None


def _normalise_key(key: str) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _parse_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer token count")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(float(str(value).strip().replace(",", "")))


def _effective_timestamp_unit(requested: str, format_name: str) -> str:
    unit = requested.lower()
    if unit != "auto":
        return unit
    if format_name == "mooncake":
        return "ms"
    return "auto"


def _parse_timestamp(value: Any, unit: str) -> float:
    if isinstance(value, (int, float)):
        return _convert_numeric_timestamp(float(value), unit)

    text = str(value).strip()
    try:
        return _convert_numeric_timestamp(float(text), unit)
    except ValueError:
        pass

    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt.timestamp()


def _convert_numeric_timestamp(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in {"auto", "s", "sec", "second", "seconds"}:
        return value
    if unit in {"ms", "millisecond", "milliseconds"}:
        return value / 1000.0
    if unit in {"us", "microsecond", "microseconds"}:
        return value / 1_000_000.0
    if unit in {"ns", "nanosecond", "nanoseconds"}:
        return value / 1_000_000_000.0
    raise ValueError(f"unsupported timestamp unit: {unit}")


def _parse_hash_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parts = [p.strip() for p in text.strip("[]").split(",")]
        return [p for p in parts if p]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("hash_ids must parse to a list")
    return [str(v) for v in parsed]


def _synthetic_block_hashes(
    *,
    source_idx: int,
    session_id: str,
    model: str,
    num_blocks: int,
) -> list[str]:
    if session_id.startswith("request_"):
        prefix = f"req:{source_idx:08d}"
    else:
        model_part = f":{model}" if model else ""
        prefix = f"sess:{session_id}{model_part}"
    return [f"{prefix}:b:{i}" for i in range(num_blocks)]


def _block_size_for_hash_ids(
    default_block_size_bytes: int,
    kv_bytes_per_token: int | None,
    hash_tokens_per_block: int,
) -> int:
    if kv_bytes_per_token is None:
        return default_block_size_bytes
    return max(1, int(kv_bytes_per_token) * int(hash_tokens_per_block))


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
