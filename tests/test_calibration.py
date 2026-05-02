import pytest

from sim.calibration import (
    apply_calibration_profile,
    assess_calibration_readiness,
    profile_name,
)


def test_calibration_profile_deep_merges_allowed_sections() -> None:
    cfg = {
        "hardware": {"hbm": {"read_bw_gbps": 1, "capacity_gb": 2}},
        "cluster": {"network": {"jitter_cv": 0.0, "seed": 1}},
        "trace": {"seed": 99},
    }
    profile = {
        "name": "test-profile",
        "overrides": {
            "hardware": {"hbm": {"read_bw_gbps": 3}},
            "cluster": {"network": {"jitter_cv": 0.2}},
        },
    }

    merged = apply_calibration_profile(cfg, profile)

    assert profile_name(profile, "fallback.yaml") == "test-profile"
    assert merged["hardware"]["hbm"]["read_bw_gbps"] == 3
    assert merged["hardware"]["hbm"]["capacity_gb"] == 2
    assert merged["cluster"]["network"]["jitter_cv"] == 0.2
    assert merged["cluster"]["network"]["seed"] == 1
    assert cfg["hardware"]["hbm"]["read_bw_gbps"] == 1


def test_calibration_profile_rejects_unknown_sections() -> None:
    with pytest.raises(ValueError):
        apply_calibration_profile({}, {"overrides": {"unknown": {"x": 1}}})


def test_assess_calibration_readiness_reports_missing_coverage() -> None:
    profile = {
        "name": "partial",
        "sources": {"gpu_compute": "bench"},
        "overrides": {"hardware": {"hbm": {"read_bw_gbps": 1}}},
    }

    readiness = assess_calibration_readiness(profile)

    assert readiness["ready"] is False
    assert readiness["override_sections"] == ["hardware"]
    assert readiness["missing_override_sections"] == ["cluster", "pd_separation"]
    assert readiness["source_keys"] == ["gpu_compute"]
    assert readiness["missing_source_keys"] == ["hbm_dram", "ssd", "network"]
    assert any("missing recommended override" in msg for msg in readiness["warnings"])


def test_assess_calibration_readiness_accepts_full_profile_metadata() -> None:
    profile = {
        "sources": {
            "gpu_compute": "operator profile",
            "hbm_dram": "bandwidth benchmark",
            "ssd": "fio",
            "network": "rdma benchmark",
        },
        "overrides": {
            "hardware": {},
            "cluster": {},
            "pd_separation": {},
        },
    }

    readiness = assess_calibration_readiness(profile)

    assert readiness["ready"] is True
    assert readiness["warnings"] == []
