from types import SimpleNamespace

from sim.presets import apply_preset, apply_runtime_defaults


def test_smoke_preset_scales_cluster_without_mutating_config() -> None:
    cfg = {
        "cluster": {"simulate_racks": 160, "simulate_gpus_per_rack": 64},
        "cluster_experiments": {"num_requests": 8000, "warmup_requests": 1000},
    }

    merged = apply_preset(cfg, "smoke")

    assert merged["cluster"]["simulate_racks"] == 2
    assert merged["cluster"]["simulate_gpus_per_rack"] == 8
    assert merged["cluster_experiments"]["num_requests"] == 120
    assert cfg["cluster"]["simulate_racks"] == 160


def test_runtime_defaults_do_not_override_explicit_true_flags() -> None:
    args = SimpleNamespace(no_plot=True, skip_context_sweep=False, no_train=False)

    apply_runtime_defaults(args, "smoke")

    assert args.no_plot is True
    assert args.skip_context_sweep is True
    assert args.no_train is True
