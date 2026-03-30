#!/usr/bin/env python3
"""
kvcache-sim — KV Cache Multi-Tier Storage Simulator

Entry point:  python main.py [--config config.yaml] [--no-train] [--no-plot]

Steps
-----
  1. Load config
  2. Generate synthetic multi-turn conversation trace
  3. (Optional) train the Learned policy model on the trace
  4. Run all 6 policy configurations
  5. Print summary table
  6. Save + display comparison plots
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _print_table(results: dict) -> None:
    try:
        from tabulate import tabulate
        rows = []
        for name, m in results.items():
            rows.append([
                name,
                f"{m.hit_rate:.2%}",
                f"{m.tier_hit_rate('HBM'):.2%}",
                f"{m.tier_hit_rate('DRAM'):.2%}",
                f"{m.tier_hit_rate('SSD'):.2%}",
                f"{m.avg_hit_latency_ms:.4f}",
                m.evictions,
                m.prefetches,
            ])
        headers = ["Policy", "HitRate", "HBM", "DRAM", "SSD", "AvgLatency(ms)", "Evictions", "Prefetches"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        # Fallback plain table
        header = f"{'Policy':<22} {'HitRate':>8} {'HBM':>7} {'DRAM':>7} {'SSD':>7} {'Latency(ms)':>12} {'Evictions':>10} {'Prefetches':>11}"
        print("\n" + header)
        print("-" * len(header))
        for name, m in results.items():
            print(
                f"{name:<22} {m.hit_rate:>7.2%} {m.tier_hit_rate('HBM'):>7.2%} "
                f"{m.tier_hit_rate('DRAM'):>7.2%} {m.tier_hit_rate('SSD'):>7.2%} "
                f"{m.avg_hit_latency_ms:>12.4f} {m.evictions:>10} {m.prefetches:>11}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache Multi-Tier Simulator")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-train", action="store_true", help="Skip training the Learned policy")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving/displaying the plot")
    parser.add_argument("--show-plot", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    print("=" * 60)
    print("  kvcache-sim  —  KV Cache Multi-Tier Storage Simulator")
    print("=" * 60)

    # ── 1. Config ────────────────────────────────────────────────────
    cfg = _load_config(args.config)
    print(f"\n[1/5] Loaded config: {args.config}")

    # ── 2. Generate trace ────────────────────────────────────────────
    from trace.generator import TraceGenerator
    gen = TraceGenerator.from_config(cfg)
    t0 = time.perf_counter()
    requests = gen.generate()
    elapsed = time.perf_counter() - t0

    num_req = cfg.get("experiments", {}).get("num_requests", len(requests))
    requests = requests[:num_req]
    warmup   = cfg.get("experiments", {}).get("warmup_requests", 200)

    print(f"[2/5] Generated {len(requests)} requests ({elapsed:.2f}s)  "
          f"[warmup={warmup}, sessions={gen.num_sessions}, turns={gen.turns_per_session}]")

    # ── 3. Train Learned model ───────────────────────────────────────
    learned_model = None
    if not args.no_train:
        print("[3/5] Training Learned policy model …")
        from learned.train import ModelTrainer
        from learned.model import LearnedModel
        trainer = ModelTrainer()
        trainer.collect(requests)
        raw = trainer.train()
        if raw is not None:
            learned_model = LearnedModel(raw)
            print("      Model ready.")
        else:
            print("      Training skipped (insufficient data or missing deps).")
    else:
        print("[3/5] Training skipped (--no-train).")

    # ── 4. Run all experiments ───────────────────────────────────────
    print("[4/5] Running 6 policy configurations …")
    from experiments.run_all import ExperimentRunner
    runner = ExperimentRunner(cfg, warmup=warmup)
    results = runner.run_all(requests, learned_model=learned_model)

    # ── 5. Print results ─────────────────────────────────────────────
    print("\n[5/5] Results summary:")
    _print_table(results)

    # ── 6. Plot ──────────────────────────────────────────────────────
    if not args.no_plot:
        from experiments.plot import plot_results
        tier_names = ["HBM", "DRAM", "SSD"]
        out = plot_results(
            results,
            tier_names=tier_names,
            output_dir=cfg.get("experiments", {}).get("output_dir", "results"),
            show=args.show_plot,
        )
        print(f"\nPlot saved → {out}")
    else:
        print("\n[plot] Skipped (--no-plot).")

    print("\nDone.")


if __name__ == "__main__":
    main()
