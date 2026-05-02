import json

import pytest

from trace.generator import Request
from trace.validation import TraceValidationThresholds, validate_workload_trace
from trace.workload import load_workload_trace


def test_validate_request_list_reports_core_metrics() -> None:
    requests = [
        Request(
            session_id="s1",
            turn_id=0,
            timestamp=0.0,
            block_hashes=["a", "b"],
            prompt_tokens=10,
            output_tokens=5,
        ),
        Request(
            session_id="s1",
            turn_id=1,
            timestamp=1000.0,
            block_hashes=["a", "b", "c"],
            prompt_tokens=20,
            output_tokens=7,
        ),
        Request(
            session_id="s2",
            turn_id=0,
            timestamp=2000.0,
            block_hashes=["x"],
            prompt_tokens=30,
            output_tokens=9,
        ),
    ]

    report = validate_workload_trace(
        requests,
        thresholds=TraceValidationThresholds(tiny_request_count=1),
    )

    assert report["request_count"] == 3
    assert report["session_count"] == 2
    assert report["duration_ms"] == 2000.0
    assert report["rps"] == pytest.approx(1.5)
    assert report["hash_id_coverage"] == 0.0
    assert report["hash_backed"] is False
    assert report["prompt_p95"] == pytest.approx(29.0)
    assert report["output_p95"] == pytest.approx(8.8)
    assert report["skipped_rows"] is None
    assert report["unique_block_count"] == 4
    assert report["repeated_block_ratio"] == pytest.approx(2 / 6)
    assert report["max_turns_per_session"] == 2
    assert any("no external hash IDs" in msg for msg in report["warnings"])


def test_validate_workload_trace_reports_hash_coverage_and_skipped_rows(tmp_path) -> None:
    path = tmp_path / "partial_hash.jsonl"
    rows = [
        {
            "timestamp": 0,
            "input_length": 512,
            "output_length": 10,
            "hash_ids": [1],
        },
        {"timestamp": 1000, "input_length": 256, "output_length": 12},
        {"timestamp": 2000, "input_length": 0, "output_length": 5},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    trace = load_workload_trace(path, block_size_bytes=1024, tokens_per_block=16)

    report = validate_workload_trace(
        trace,
        thresholds=TraceValidationThresholds(tiny_request_count=1),
    )

    assert trace.hash_backed_requests == 1
    assert report["request_count"] == 2
    assert report["source_rows"] == 3
    assert report["skipped_rows"] == 1
    assert report["skipped_row_ratio"] == pytest.approx(1 / 3)
    assert report["hash_id_coverage"] == pytest.approx(0.5)
    assert report["hash_backed"] is True
    assert any("partial hash ID coverage" in msg for msg in report["warnings"])
    assert any("skipped 1 source rows" in msg for msg in report["warnings"])


def test_validate_workload_trace_warns_for_risky_trace() -> None:
    requests = [
        Request(
            session_id="s1",
            turn_id=0,
            timestamp=0.0,
            block_hashes=["a"],
            prompt_tokens=10,
            output_tokens=0,
        ),
        Request(
            session_id="s2",
            turn_id=0,
            timestamp=0.0,
            block_hashes=["b"],
            prompt_tokens=20,
            output_tokens=0,
        ),
    ]

    with pytest.warns(RuntimeWarning):
        report = validate_workload_trace(requests, emit_warnings=True)

    warnings = report["warnings"]
    assert any("tiny request count" in msg for msg in warnings)
    assert any("non-positive" in msg for msg in warnings)
    assert any("no external hash IDs" in msg for msg in warnings)
    assert any("very low session reuse" in msg for msg in warnings)
    assert any("zero or unknown output tokens" in msg for msg in warnings)
