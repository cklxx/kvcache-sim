import csv
import json

from sim.metrics import Metrics
from sim.reporting import build_report, write_csv_report, write_json_report


def test_json_report_contains_metadata_config_and_metrics(tmp_path) -> None:
    metrics = Metrics(tier_names=["HBM"])
    metrics.total_requests = 2
    metrics.record_hit("HBM", 0.1)
    metrics.record_miss()

    report = build_report(
        mode="cluster",
        args={"preset": "smoke"},
        config_path="config.yaml",
        config={"trace": {"seed": 1}},
        results={"policy_results": {"LRU": metrics}},
        elapsed_seconds=1.25,
        run_id="test-run",
        timestamp="2026-01-01T00:00:00Z",
        argv=["--cluster"],
        run_summary={"request_count": 2},
    )
    out = write_json_report(report, tmp_path / "report.json")

    payload = json.loads(out.read_text())

    assert payload["metadata"]["run_id"] == "test-run"
    assert payload["metadata"]["mode"] == "cluster"
    assert payload["metadata"]["args"]["preset"] == "smoke"
    assert payload["metadata"]["run_summary"]["request_count"] == 2
    assert payload["config"]["trace"]["seed"] == 1
    assert payload["results"]["policy_results"]["LRU"]["hit_rate"] == 0.5


def test_csv_report_flattens_metric_rows(tmp_path) -> None:
    metrics = Metrics(tier_names=["HBM"])
    metrics.total_requests = 1
    metrics.record_hit("HBM", 0.2)
    report = build_report(
        mode="single_node",
        args={},
        config_path="config.yaml",
        config={},
        results={"policy_results": {"LRU": metrics}},
        elapsed_seconds=0.1,
        run_id="csv-run",
        timestamp="2026-01-01T00:00:00Z",
    )

    out = write_csv_report(report, tmp_path / "report.csv")
    rows = list(csv.DictReader(out.open()))

    assert len(rows) == 1
    assert rows[0]["run_id"] == "csv-run"
    assert rows[0]["section"] == "policy_results"
    assert rows[0]["config"] == "LRU"
    assert rows[0]["hit_rate"] == "1.0"


def test_report_warns_on_storage_over_capacity() -> None:
    metrics = Metrics(tier_names=["HBM"])
    metrics.record_storage("HBM", used_bytes=20, capacity_bytes=10, block_count=2)

    report = build_report(
        mode="pd",
        args={},
        config_path="config.yaml",
        config={},
        results={"over": metrics},
        elapsed_seconds=0.1,
    )

    warnings = report["results"]["over"]["warnings"]
    assert warnings == [
        "HBM storage usage exceeds capacity (2.00x); inspect cache capacity model."
    ]
