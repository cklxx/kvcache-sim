"""
Run presets for common simulator workflows.

Presets intentionally change only scale/runtime defaults. Hardware and model
calibration still belongs in calibration profiles.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping


PRESET_NAMES = ("smoke", "dev", "prod-eval")

_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "smoke": {
        "experiments": {"num_requests": 80, "warmup_requests": 10},
        "cluster": {"simulate_racks": 2, "simulate_gpus_per_rack": 8},
        "cluster_experiments": {"num_requests": 120, "warmup_requests": 20},
        "pd_experiments": {"num_requests": 80, "warmup_requests": 10},
        "trace": {"num_sessions": 40, "turns_per_session": 3},
        "cluster_trace": {
            "num_sessions": 80,
            "turns_per_session": 3,
            "num_shared_docs": 20,
        },
        "pd_trace": {"num_sessions": 40, "turns_per_session": 3},
    },
    "dev": {
        "experiments": {"num_requests": 1000, "warmup_requests": 100},
        "cluster": {"simulate_racks": 8, "simulate_gpus_per_rack": 16},
        "cluster_experiments": {"num_requests": 1500, "warmup_requests": 150},
        "pd_experiments": {"num_requests": 1000, "warmup_requests": 100},
        "cluster_trace": {
            "num_sessions": 1500,
            "turns_per_session": 4,
            "num_shared_docs": 80,
        },
        "pd_trace": {"num_sessions": 1000, "turns_per_session": 4},
    },
    "prod-eval": {},
}

_RUNTIME_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {"no_plot": True, "skip_context_sweep": True, "no_train": True},
    "dev": {"no_plot": True, "skip_context_sweep": True},
    "prod-eval": {},
}


def apply_preset(config: Mapping[str, Any], preset: str | None) -> dict:
    """Return a config copy with the selected preset overrides applied."""
    if not preset:
        return copy.deepcopy(config)
    _validate_preset(preset)

    merged = copy.deepcopy(config)
    _deep_merge(merged, _CONFIG_OVERRIDES[preset])
    return merged


def runtime_defaults(preset: str | None) -> dict[str, Any]:
    """Return argparse attribute defaults implied by a preset."""
    if not preset:
        return {}
    _validate_preset(preset)
    return copy.deepcopy(_RUNTIME_DEFAULTS[preset])


def apply_runtime_defaults(args: Any, preset: str | None) -> None:
    """
    Mutate an argparse namespace with preset runtime defaults.

    Defaults only fill falsey boolean flags so explicit user flags keep
    priority. This function is deliberately narrow to avoid surprising callers.
    """
    for key, value in runtime_defaults(preset).items():
        if isinstance(value, bool):
            if not getattr(args, key, False):
                setattr(args, key, value)
        elif getattr(args, key, None) is None:
            setattr(args, key, value)


def _validate_preset(preset: str) -> None:
    if preset not in PRESET_NAMES:
        allowed = ", ".join(PRESET_NAMES)
        raise ValueError(f"unknown preset {preset!r}; expected one of: {allowed}")


def _deep_merge(dst: dict, src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
