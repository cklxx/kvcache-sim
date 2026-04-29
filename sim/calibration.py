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


def _deep_merge(dst: dict, src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
