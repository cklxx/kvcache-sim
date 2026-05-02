"""
Calibration profile support.

Profiles are lightweight YAML overlays for parameters that should come from
microbenchmarks or external simulators such as Vidur, Accel-Sim, Ramulator,
MQSim, or SimpleSSD. They intentionally tune this system-level simulator
without embedding those heavyweight simulators in the replay loop.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_TOP_LEVEL = {
    "hardware",
    "cache",
    "cluster",
    "pd_separation",
}
READINESS_OVERRIDE_SECTIONS = ("hardware", "cluster", "pd_separation")
READINESS_SOURCE_KEYS = ("gpu_compute", "hbm_dram", "ssd", "network")


def load_calibration_profile(path: str | Path) -> dict:
    """Load a calibration YAML profile."""
    with open(path) as f:
        profile = yaml.safe_load(f) or {}
    if not isinstance(profile, dict):
        raise ValueError("calibration profile must be a YAML mapping")
    return profile


def apply_calibration_profile(config: dict, profile: Mapping[str, Any]) -> dict:
    """
    Return a config copy with the profile overrides applied.

    Supported profile shape:

      name: h100_70b_reference
      overrides:
        hardware: ...
        cluster:
          network: ...
        pd_separation:
          compute: ...

    For convenience, the allowed top-level sections may also be placed directly
    in the profile. Unknown top-level override sections are rejected so typos do
    not silently create unused config keys.
    """
    overrides = profile.get("overrides", profile)
    if not isinstance(overrides, Mapping):
        raise ValueError("calibration profile overrides must be a mapping")

    unknown = set(overrides) - ALLOWED_TOP_LEVEL
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported calibration override section(s): {names}")

    merged = copy.deepcopy(config)
    _deep_merge(merged, overrides)
    return merged


def profile_name(profile: Mapping[str, Any], path: str | Path) -> str:
    return str(profile.get("name") or Path(path).stem)


def assess_calibration_readiness(profile: Mapping[str, Any]) -> dict[str, Any]:
    """
    Report whether a calibration profile has production-ready coverage metadata.

    This is intentionally advisory. It does not apply overrides and does not
    change the validation behavior in :func:`apply_calibration_profile`.
    """
    overrides = profile.get("overrides", profile)
    if not isinstance(overrides, Mapping):
        raise ValueError("calibration profile overrides must be a mapping")

    override_keys = set(overrides)
    unknown_sections = sorted(str(k) for k in override_keys - ALLOWED_TOP_LEVEL)
    present_sections = [
        section for section in READINESS_OVERRIDE_SECTIONS if section in override_keys
    ]
    missing_sections = [
        section
        for section in READINESS_OVERRIDE_SECTIONS
        if section not in override_keys
    ]

    sources_raw = profile.get("sources", {})
    sources = sources_raw if isinstance(sources_raw, Mapping) else {}
    source_keys = sorted(str(key) for key, value in sources.items() if value)
    missing_source_keys = [
        key for key in READINESS_SOURCE_KEYS if not sources.get(key)
    ]

    warnings: list[str] = []
    if unknown_sections:
        warnings.append(
            "Calibration profile contains unsupported override section(s): "
            + ", ".join(unknown_sections)
        )
    if missing_sections:
        warnings.append(
            "Calibration profile is missing recommended override section(s): "
            + ", ".join(missing_sections)
        )
    if missing_source_keys:
        warnings.append(
            "Calibration profile is missing recommended source metadata: "
            + ", ".join(missing_source_keys)
        )

    return {
        "ready": not warnings,
        "override_sections": present_sections,
        "missing_override_sections": missing_sections,
        "source_keys": source_keys,
        "missing_source_keys": missing_source_keys,
        "warnings": warnings,
    }


def _deep_merge(dst: dict, src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
