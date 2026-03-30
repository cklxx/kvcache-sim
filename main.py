#!/usr/bin/env python3
"""
kvcache-sim — KV Cache Multi-Tier Storage Simulator

  python main.py                 # single-node (4 workers) demo
  python main.py --cluster       # 万卡集群 + EIC demo
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


# ======================================================================
# Table printers
# ======================================================================


def _print_table(results: dict, tier_names=None) -> None:
    if tier_names is None:
        tier_names = ["HBM", "DRAM", "SSD"]
    try:
        from tabulate import tabulate
        rows = []
        for name, m in results.items():
            row = [name, f"{m.hit_rate:.2%}"]
            for t in tier_names:
                row.append(f"{m.tier_hit_rate(t):.2%}")
            row += [f"{m.avg_hit_latency_ms:.4f}", m.evictions, m.prefetches]
            rows.append(row)
        headers = ["Policy", "HitRate"] + tier_names + ["AvgLat(ms)", "Evictions", "Prefetches"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        for name, m in results.items():
            parts = [f"{name:<24}", f"hit={m.hit_rate:.2%}"]
            for t in tier_names:
                parts.append(f"{t}={m.tier_hit_rate(t):.2%}")
            parts.append(f"lat={m.avg_hit_latency_ms:.4f}ms")
            parts.append(f"evict={m.evictions}")
            print("  ".join(parts))


def _print_cluster_info(cluster) -> None:
    print(f"\n  {cluster.summary()}")
    eic_utils = cluster.eic_utilizations()
    xgpu = cluster.total_cross_gpu_eic_hits
    print(f"  EIC utilization: {', '.join(f'R{k}={v:.0%}' for k,v in eic_utils.items())}")
    print(f"  Cross-GPU EIC hits (shared prefix reuse): {xgpu}")


# ======================================================================
# Single-node mode
# ======================================================================


def run_single_node(cfg: dict, args) -> None:
    print("=" * 64)
    print("  kvcache-sim  —  Single-Node Multi-Tier Demo")
    print("=" * 64)

    # 1. Config
    print(f"\n[1/5] Loaded config: {args.config}")

    # 2. Trace
    from trace.generator import TraceGenerator
    gen = TraceGenerator.from_config(cfg)
    t0 = time.perf_counter()
    requests = gen.generate()
    elapsed = time.perf_counter() - t0

    num_req = cfg.get("experiments", {}).get("num_requests", len(requests))
    requests = requests[:num_req]
    warmup = cfg.get("experiments", {}).get("warmup_requests", 200)
    print(f"[2/5] Generated {len(requests)} requests ({elapsed:.2f}s)  "
          f"[warmup={warmup}, sessions={gen.num_sessions}]")

    # 3. Learned model
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
            print("      Skipped (insufficient data).")
    else:
        print("[3/5] Training skipped.")

    # 4. Experiments
    print("[4/5] Running 6 policy configurations …")
    from experiments.run_all import ExperimentRunner
    runner = ExperimentRunner(cfg, warmup=warmup)
    results = runner.run_all(requests, learned_model=learned_model)

    # 5. Results
    print("\n[5/5] Results:")
    _print_table(results, ["HBM", "DRAM", "SSD"])

    # 6. Plot
    if not args.no_plot:
        from experiments.plot import plot_results
        out = plot_results(
            results, tier_names=["HBM", "DRAM", "SSD"],
            output_dir=cfg.get("experiments", {}).get("output_dir", "results"),
        )
        print(f"\nPlot saved → {out}")

    print("\nDone.")


# ======================================================================
# Cluster mode  (万卡 + EIC)
# ======================================================================


def run_cluster(cfg: dict, args) -> None:
    cc = cfg.get("cluster", {})
    ct = cfg.get("cluster_trace", cfg.get("trace", {}))
    ce = cfg.get("cluster_experiments", cfg.get("experiments", {}))

    n_racks = cc.get("simulate_racks", 8)
    n_gpus_per_rack = cc.get("simulate_gpus_per_rack", 16)
    total_sim = n_racks * n_gpus_per_rack
    full_cluster = cc.get("total_gpus", 10240)

    print("=" * 64)
    print(f"  kvcache-sim  —  万卡 Cluster + EIC Demo")
    print(f"  Full cluster: {full_cluster:,} GPUs  |  "
          f"Simulating: {total_sim} GPUs ({n_racks} racks × {n_gpus_per_rack})")
    print("=" * 64)

    # ── 1. Generate high-volume trace ────────────────────────────────
    from trace.generator import TraceGenerator
    gen = TraceGenerator(
        num_sessions=ct.get("num_sessions", 5000),
        turns_per_session=ct.get("turns_per_session", 5),
        prompt_tokens_min=ct.get("prompt_tokens_min", 64),
        prompt_tokens_max=ct.get("prompt_tokens_max", 512),
        num_system_prompts=ct.get("num_system_prompts", 20),
        qps=ct.get("qps", 500.0),
        block_size_bytes=cfg.get("cache", {}).get("block_size_bytes", 4096),
        seed=ct.get("seed", 42),
    )
    t0 = time.perf_counter()
    requests = gen.generate()
    elapsed = time.perf_counter() - t0

    num_req = ce.get("num_requests", len(requests))
    requests = requests[:num_req]
    warmup = ce.get("warmup_requests", 1000)

    print(f"\n[1/4] Generated {len(requests)} requests ({elapsed:.2f}s)  "
          f"[sessions={gen.num_sessions}, sys_prompts={gen.num_system_prompts}]")

    # ── 2. EIC sizing experiments ────────────────────────────────────
    print(f"\n[2/4] EIC Sizing Experiments (warmup={warmup}):")
    from experiments.run_all import ClusterExperimentRunner
    runner = ClusterExperimentRunner(cfg, warmup=warmup)
    eic_results = runner.run_eic_sizing(requests)

    print("\n  EIC Sizing Results:")
    _print_table(eic_results, ["HBM", "EIC", "Remote"])

    # ── 3. Eviction policy comparison at cluster scale ───────────────
    print(f"\n[3/4] Eviction Policies at Cluster Scale:")
    eviction_results = runner.run_eviction_at_scale(requests)

    print("\n  Eviction Policy Results (cluster):")
    _print_table(eviction_results, ["HBM", "EIC", "Remote"])

    # ── 4. Build final cluster and show stats ────────────────────────
    print(f"\n[4/4] Final Cluster Topology:")
    from sim.cluster import build_cluster
    from trace.cluster_replay import ClusterReplayer
    cluster = build_cluster(cfg)
    replayer = ClusterReplayer(cluster, warmup_count=warmup)
    replayer.run(requests)
    _print_cluster_info(cluster)

    # ── Plot ─────────────────────────────────────────────────────────
    if not args.no_plot:
        from experiments.plot import plot_results
        out_dir = ce.get("output_dir", "results")
        out1 = plot_results(
            eic_results, tier_names=["HBM", "EIC", "Remote"],
            output_dir=out_dir, filename="cluster_eic_sizing.png",
        )
        out2 = plot_results(
            eviction_results, tier_names=["HBM", "EIC", "Remote"],
            output_dir=out_dir, filename="cluster_eviction.png",
        )
        print(f"\nPlots saved → {out1}, {out2}")

    print("\nDone.")


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache Multi-Tier Simulator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cluster", action="store_true", help="Run 万卡 cluster + EIC mode")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.cluster:
        run_cluster(cfg, args)
    else:
        run_single_node(cfg, args)


if __name__ == "__main__":
    main()
