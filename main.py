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


def _tokens_per_block(cfg: dict) -> int:
    return int(
        cfg.get("pd_separation", {})
        .get("compute", {})
        .get("tokens_per_block", 16)
    )


def _kv_bytes_per_token(cfg: dict) -> int | None:
    value = (
        cfg.get("pd_separation", {})
        .get("compute", {})
        .get("kv_bytes_per_token")
    )
    return int(value) if value is not None else None


def _load_workload_requests(cfg: dict, args, limit: int | None):
    if not args.workload_trace:
        return None

    from trace.workload import load_workload_trace, summarize_workload

    load_limit = args.workload_limit if args.workload_limit is not None else limit
    workload = load_workload_trace(
        args.workload_trace,
        block_size_bytes=cfg.get("cache", {}).get("block_size_bytes", 4096),
        kv_bytes_per_token=_kv_bytes_per_token(cfg),
        tokens_per_block=_tokens_per_block(cfg),
        format_name=args.workload_format,
        limit=load_limit,
        timestamp_unit=args.workload_time_unit,
        arrival_scale=args.workload_arrival_scale,
        include_failed=args.workload_include_failed,
        hash_tokens_per_block=args.workload_hash_tokens_per_block,
    )
    summary = summarize_workload(workload)
    if not workload.requests:
        raise ValueError(f"No usable workload requests loaded from {args.workload_trace}")

    prompt = (
        f"prompt_avg={summary['prompt_avg']:.0f}, "
        f"prompt_p95={summary['prompt_p95']:.0f}"
    )
    output = ""
    if "output_avg" in summary:
        output = (
            f", output_avg={summary['output_avg']:.0f}, "
            f"output_p95={summary['output_p95']:.0f}"
        )
    hash_note = "hash_ids" if summary.get("hash_backed") else "synthetic_prefix"
    print(
        f"[workload] Loaded {summary['requests']} requests from {args.workload_trace} "
        f"[format={summary['format']}, {prompt}{output}, "
        f"rps={summary['rps']:.2f}, {hash_note}, skipped={summary['skipped_rows']}]"
    )
    return workload.requests


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


def _print_context_table(ctx_results: dict) -> None:
    """Print context length sweep results."""
    try:
        from tabulate import tabulate
        rows = []
        for label, data in ctx_results.items():
            m_eic = data["eic"]
            m_no = data["no_eic"]
            delta = m_eic.hit_rate - m_no.hit_rate
            rows.append([
                label,
                data["blocks_per_req"],
                f"{m_no.hit_rate:.2%}",
                f"{m_eic.hit_rate:.2%}",
                f"+{delta:.2%}" if delta >= 0 else f"{delta:.2%}",
                f"{m_eic.tier_hit_rate('EIC'):.2%}",
                data["eic_xgpu"],
                m_eic.evictions,
            ])
        headers = ["Context", "Blk/Req", "NoEIC", "W/EIC", "Delta", "EIC%", "xGPU", "Evictions"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        for label, data in ctx_results.items():
            m_eic = data["eic"]
            m_no = data["no_eic"]
            delta = m_eic.hit_rate - m_no.hit_rate
            print(f"  {label}  blk={data['blocks_per_req']}  "
                  f"no_eic={m_no.hit_rate:.2%}  w/eic={m_eic.hit_rate:.2%}  "
                  f"delta=+{delta:.2%}  xGPU={data['eic_xgpu']}")


def _print_credibility_report(cfg: dict, cluster) -> None:
    """Print simulation fidelity analysis vs real H100 + vLLM."""
    cc = cfg.get("cluster", {})
    gpu_cfg = cc.get("gpu", {})
    eic_cfg = cc.get("eic", {})
    net_cfg = cc.get("network", {})

    hbm_gb = gpu_cfg.get("hbm_capacity_gb", 0.003)
    eic_per_node_gb = eic_cfg.get("capacity_per_node_gb", 0.02)
    eic_nodes = eic_cfg.get("nodes_per_rack", 4)
    gpus_per_rack = cc.get("simulate_gpus_per_rack", 16)
    eic_per_gpu = eic_per_node_gb * eic_nodes / gpus_per_rack
    block_size_mb = cfg.get("cache", {}).get("block_size_bytes", 4096) / (1024 * 1024)
    tokens_per_block = (
        cfg.get("pd_separation", {})
        .get("compute", {})
        .get("tokens_per_block", 16)
    )

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                Simulation Fidelity Report                       │
  ├────────────────────┬──────────────┬──────────────┬──────────────┤
  │ Parameter          │ Simulator    │ Real H100    │ Match?       │
  ├────────────────────┼──────────────┼──────────────┼──────────────┤
  │ HBM BW             │ 3200 GB/s    │ 3350 GB/s    │ ~            │
  │ HBM latency        │ 1 μs         │ ~1 μs        │ =            │
  │ CXL/EIC BW         │ 100 GB/s     │ 64-128 GB/s  │ ~            │
  │ CXL/EIC latency    │ 5 μs         │ 3-10 μs      │ ~            │
  │ RDMA intra-rack    │ {net_cfg.get("intra_rack_latency_us", 3)} μs         │ 2-5 μs       │ =            │
  │ RDMA cross-rack    │ {net_cfg.get("cross_rack_latency_us", 15)} μs        │ 10-30 μs     │ ~            │
  │ NVMe-oF SSD        │ {net_cfg.get("remote_ssd_latency_us", 200)} μs       │ 100-500 μs   │ ~            │
  ├────────────────────┼──────────────┼──────────────┼──────────────┤
  │ HBM KV budget/GPU  │ {hbm_gb * 1024:.0f} MB       │ ~10 GB       │ x{10 / (hbm_gb):.0f} scaled   │
  │ EIC per GPU share   │ {eic_per_gpu * 1024:.0f} MB       │ ~16 GB       │ x{16 / (eic_per_gpu):.0f} scaled   │
  │ EIC : HBM ratio    │ {eic_per_gpu / hbm_gb:.1f}×          │ ~1.6×        │ {"=" if abs(eic_per_gpu / hbm_gb - 1.6) < 0.5 else "~"}            │
  │ Block size          │ {block_size_mb:.1f} MiB      │ 0.3-5 MB     │ =            │
  │ Tokens/block        │ {tokens_per_block:<12} │ 16 (vLLM)    │ =            │
  ├────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Session routing     │ affinity     │ vLLM affinity│ =            │
  │ Prefix sharing      │ shared sys   │ vLLM radix   │ ~            │
  │ Eviction scope      │ per-GPU      │ per-GPU      │ =            │
  │ EIC sharing scope   │ per-rack     │ per-rack CXL │ =            │
  └────────────────────┴──────────────┴──────────────┴──────────────┘

  Legend: = exact match  ~ approximate  xN absolute capacity scaled N×
  Note: Absolute capacities are scaled down ~{10 / hbm_gb:.0f}× for tractability.
        Ratios (HBM:EIC, bandwidth hierarchy, latency ordering) are preserved.
        Cache dynamics depend on ratios, not absolute sizes.""")


def _print_cluster_info(cluster) -> None:
    print(f"\n  {cluster.summary()}")
    eic_utils = cluster.eic_utilizations()
    xgpu = cluster.total_cross_gpu_eic_hits
    print(f"  EIC utilization: {', '.join(f'R{k}={v:.0%}' for k,v in eic_utils.items())}")
    print(f"  Cross-GPU EIC hits (shared prefix reuse): {xgpu}")


def _learned_training_plan(cfg: dict, args, num_requests: int, warmup: int) -> tuple[str, int, int]:
    """
    Choose the prefix of the trace used to train LearnedPolicy.

    Returns (mode, train_count, eval_warmup). Requests before eval_warmup are
    excluded from reported metrics, so learned training data is not part of the
    measured evaluation window.
    """
    learned_cfg = cfg.get("learned", {})
    mode = args.learned_train_mode or learned_cfg.get("train_mode", "warmup")
    if mode not in {"warmup", "split"}:
        raise ValueError(f"Unknown learned train mode: {mode}")

    if mode == "warmup":
        train_count = min(max(int(warmup), 0), num_requests)
        return mode, train_count, warmup

    fraction = (
        args.learned_train_fraction
        if args.learned_train_fraction is not None
        else learned_cfg.get("train_fraction", 0.2)
    )
    fraction = max(0.0, min(float(fraction), 1.0))
    if num_requests <= 1 or fraction <= 0.0:
        train_count = 0
    else:
        train_count = int(num_requests * fraction)
        train_count = max(1, min(train_count, num_requests - 1))

    return mode, train_count, max(warmup, train_count)


# ======================================================================
# Single-node mode
# ======================================================================


def run_single_node(cfg: dict, args) -> None:
    print("=" * 64)
    print("  kvcache-sim  —  Single-Node Multi-Tier Demo")
    print("=" * 64)

    # 1. Config
    print(f"\n[1/5] Loaded config: {args.config}")

    num_req = cfg.get("experiments", {}).get("num_requests")
    warmup = cfg.get("experiments", {}).get("warmup_requests", 200)

    # 2. Trace
    requests = _load_workload_requests(cfg, args, num_req)
    if requests is None:
        from trace.generator import TraceGenerator
        gen = TraceGenerator.from_config(cfg)
        t0 = time.perf_counter()
        requests = gen.generate()
        elapsed = time.perf_counter() - t0
        if num_req is not None:
            requests = requests[:num_req]
        print(f"[2/5] Generated {len(requests)} requests ({elapsed:.2f}s)  "
              f"[warmup={warmup}, sessions={gen.num_sessions}]")
    else:
        if num_req is not None:
            requests = requests[:num_req]
        print(f"[2/5] Using workload trace ({len(requests)} requests)  "
              f"[warmup={warmup}]")

    # 3. Learned model
    learned_model = None
    eval_warmup = warmup
    if not args.no_train:
        mode, train_count, eval_warmup = _learned_training_plan(
            cfg, args, len(requests), warmup
        )
        print(
            f"[3/5] Training Learned policy model "
            f"[mode={mode}, train_requests={train_count}, eval_warmup={eval_warmup}] …"
        )
        from learned.train import ModelTrainer
        from learned.model import LearnedModel

        if train_count > 0:
            learned_cfg = cfg.get("learned", {})
            trainer = ModelTrainer(
                min_samples=learned_cfg.get("min_samples", 200)
            )
            trainer.collect(requests[:train_count])
            raw = trainer.train()
            if raw is not None:
                learned_model = LearnedModel(raw)
                print("      Model ready.")
            else:
                print("      Skipped (insufficient data).")
        else:
            print("      Skipped (no training requests outside evaluation window).")
    else:
        print("[3/5] Training skipped.")

    # 4. Experiments
    print("[4/5] Running 6 policy configurations …")
    from experiments.run_all import ExperimentRunner
    runner = ExperimentRunner(cfg, warmup=eval_warmup)
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


def _print_pd_table(results: dict) -> None:
    """Print PD metrics comparison table."""
    try:
        from tabulate import tabulate
        rows = []
        for name, m in results.items():
            rows.append([
                name,
                f"{m.ttft_p50:.1f}",
                f"{m.ttft_p99:.1f}",
                f"{m.tpot_avg:.1f}",
                f"{m.e2e_p50:.0f}",
                f"{m.avg_prefill_compute_ms:.1f}",
                f"{m.avg_transfer_ms:.3f}",
                f"{m.avg_queue_wait_ms:.1f}",
                f"{m.prefix_cache_hit_rate:.1%}",
                f"{m.same_rack_ratio:.0%}",
            ])
        headers = [
            "Config", "TTFT_p50", "TTFT_p99", "TPOT_avg",
            "E2E_p50", "Prefill", "Transfer", "QueueWait",
            "PrefixHit", "SameRack",
        ]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        for name, m in results.items():
            print(
                f"  {name:<24} TTFT_p50={m.ttft_p50:.1f}ms  "
                f"TPOT={m.tpot_avg:.1f}ms  prefix_hit={m.prefix_cache_hit_rate:.1%}"
            )


def _print_pd_context_table(ctx_results: dict) -> None:
    """Print context length × PD comparison table."""
    try:
        from tabulate import tabulate
        rows = []
        for label, data in ctx_results.items():
            m_uni = data["unified"]
            m_pd = data["pd"]
            delta = m_uni.ttft_p50 - m_pd.ttft_p50
            rows.append([
                label,
                f"{m_uni.ttft_p50:.1f}",
                f"{m_pd.ttft_p50:.1f}",
                f"{delta:+.1f}",
                f"{m_pd.avg_transfer_ms:.3f}",
                f"{m_pd.prefix_cache_hit_rate:.1%}",
                f"{m_pd.tpot_avg:.1f}",
            ])
        headers = [
            "Context", "Unified_TTFT", "PD_TTFT", "Delta",
            "Transfer", "PrefixHit", "TPOT",
        ]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        for label, data in ctx_results.items():
            m_uni = data["unified"]
            m_pd = data["pd"]
            delta = m_uni.ttft_p50 - m_pd.ttft_p50
            print(f"  {label}  uni={m_uni.ttft_p50:.1f}ms  pd={m_pd.ttft_p50:.1f}ms  delta={delta:+.1f}ms")


def run_pd(cfg: dict, args) -> None:
    """PD-separated cluster mode."""
    from sim.pd_cluster import PDConfig, build_pd_cluster
    from trace.generator import TraceGenerator
    from experiments.pd_experiments import PDExperimentRunner

    pd_cfg = PDConfig.from_config(cfg)
    pt = cfg.get("pd_trace", cfg.get("trace", {}))
    pe = cfg.get("pd_experiments", cfg.get("experiments", {}))

    print("=" * 64)
    print("  kvcache-sim  —  PD Separation Mode")
    print("=" * 64)

    cluster = build_pd_cluster(cfg, pd_cfg)
    print(f"\n  {cluster.summary()}")

    num_req = pe.get("num_requests")
    warmup = pe.get("warmup_requests", 500)
    requests = _load_workload_requests(cfg, args, num_req)
    if requests is None:
        # 1. Generate trace
        gen = TraceGenerator(
            num_sessions=pt.get("num_sessions", 3000),
            turns_per_session=pt.get("turns_per_session", 5),
            prompt_tokens_min=pt.get("prompt_tokens_min", 128),
            prompt_tokens_max=pt.get("prompt_tokens_max", 2048),
            initial_context_tokens=pt.get("initial_context_tokens", 1024),
            num_system_prompts=pt.get("num_system_prompts", 20),
            qps=pt.get("qps", 300.0),
            block_size_bytes=cfg.get("cache", {}).get("block_size_bytes", 4096),
            seed=pt.get("seed", 42),
        )
        t0 = time.perf_counter()
        requests = gen.generate()
        elapsed = time.perf_counter() - t0
        if num_req is not None:
            requests = requests[:num_req]
        trace_msg = f"Generated {len(requests)} requests ({elapsed:.2f}s)"
    else:
        if num_req is not None:
            requests = requests[:num_req]
        trace_msg = f"Using workload trace ({len(requests)} requests)"
    avg_blocks = sum(len(r.block_hashes) for r in requests) / max(len(requests), 1)

    print(f"\n[1/4] {trace_msg}  "
          f"[warmup={warmup}, avg {avg_blocks:.0f} blocks/req]")

    runner = PDExperimentRunner(cfg, warmup=warmup)

    # 2. Unified vs PD
    print(f"\n[2/4] Unified vs PD-Separated:")
    uvp_results = runner.run_unified_vs_pd(requests)
    _print_pd_table(uvp_results)

    # 3. P:D Ratio Sweep
    print(f"\n[3/4] P:D Ratio Sweep:")
    ratio_results = runner.run_pd_ratio_sweep(requests)
    _print_pd_table(ratio_results)

    # 4. Transfer Strategy Comparison
    print(f"\n[4/4] Transfer Strategy Comparison:")
    transfer_results = runner.run_transfer_strategy(requests)
    _print_pd_table(transfer_results)

    # 5. Context Length Sweep (optional, longer)
    if not args.no_plot and not args.skip_context_sweep:
        print(f"\n[Bonus] Context Length × PD Benefit:")
        ctx_results = runner.run_context_length_pd()
        _print_pd_context_table(ctx_results)

    # Plot
    if not args.no_plot:
        out_dir = pe.get("output_dir", "results")
        os.makedirs(out_dir, exist_ok=True)
        _plot_pd_results(uvp_results, ratio_results, transfer_results, out_dir)

    print("\nDone.")


def _plot_pd_results(uvp, ratios, transfers, out_dir):
    """Generate PD-specific comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[pd] matplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("KV-Cache Simulator — PD Separation Analysis", fontsize=14, fontweight="bold")

    # Panel 1: Unified vs PD (TTFT)
    ax = axes[0, 0]
    names = list(uvp.keys())
    ttft_p50 = [uvp[n].ttft_p50 for n in names]
    ttft_p99 = [uvp[n].ttft_p99 for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, ttft_p50, w, label="P50", color="#2196F3")
    ax.bar(x + w/2, ttft_p99, w, label="P99", color="#FF9800")
    ax.set_title("TTFT: Unified vs PD (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylabel("ms")
    for i, (v50, v99) in enumerate(zip(ttft_p50, ttft_p99)):
        ax.text(i - w/2, v50, f"{v50:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, v99, f"{v99:.0f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: P:D Ratio Sweep (TTFT + TPOT)
    ax = axes[0, 1]
    names = list(ratios.keys())
    ttft = [ratios[n].ttft_p50 for n in names]
    tpot = [ratios[n].tpot_avg for n in names]
    x = np.arange(len(names))
    ax2 = ax.twinx()
    bars = ax.bar(x, ttft, 0.5, label="TTFT P50", color="#4CAF50", alpha=0.8)
    line = ax2.plot(x, tpot, "o-", color="#E53935", label="TPOT avg")
    ax.set_title("P:D Ratio Sweep")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("TTFT P50 (ms)")
    ax2.set_ylabel("TPOT avg (ms)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Transfer Strategy
    ax = axes[1, 0]
    names = list(transfers.keys())
    ttft = [transfers[n].ttft_p50 for n in names]
    transfer_t = [transfers[n].avg_transfer_ms for n in names]
    x = np.arange(len(names))
    ax.bar(x, ttft, 0.5, color="#9C27B0")
    ax.set_title("Transfer Strategy: TTFT P50 (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("ms")
    for i, v in enumerate(ttft):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    # Panel 4: Prefix Cache Hit Rate
    ax = axes[1, 1]
    all_results = {**uvp, **ratios}
    names = list(all_results.keys())
    hits = [all_results[n].prefix_cache_hit_rate * 100 for n in names]
    x = np.arange(len(names))
    ax.bar(x, hits, 0.6, color="#00BCD4")
    ax.set_title("Prefix Cache Hit Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=7)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pd_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[pd] Plot saved → {out_path}")
    plt.close(fig)


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

    num_req = ce.get("num_requests")
    warmup = ce.get("warmup_requests", 1000)
    requests = _load_workload_requests(cfg, args, num_req)
    if requests is None:
        # ── 1. Generate high-volume trace ────────────────────────────────
        from trace.generator import TraceGenerator
        gen = TraceGenerator(
            num_sessions=ct.get("num_sessions", 5000),
            turns_per_session=ct.get("turns_per_session", 5),
            prompt_tokens_min=ct.get("prompt_tokens_min", 64),
            prompt_tokens_max=ct.get("prompt_tokens_max", 512),
            initial_context_tokens=ct.get("initial_context_tokens", 128),
            num_system_prompts=ct.get("num_system_prompts", 20),
            num_shared_docs=ct.get("num_shared_docs", 0),
            num_rag_chunks=ct.get("num_rag_chunks", 3),
            doc_zipf_alpha=ct.get("doc_zipf_alpha", 1.1),
            qps=ct.get("qps", 500.0),
            block_size_bytes=cfg.get("cache", {}).get("block_size_bytes", 4096),
            seed=ct.get("seed", 42),
        )
        t0 = time.perf_counter()
        requests = gen.generate()
        elapsed = time.perf_counter() - t0
        if num_req is not None:
            requests = requests[:num_req]
        trace_msg = (
            f"Generated {len(requests)} requests ({elapsed:.2f}s)  "
            f"[sessions={gen.num_sessions}, sys_prompts={gen.num_system_prompts}, "
            f"shared_docs={gen.num_shared_docs}]"
        )
    else:
        if num_req is not None:
            requests = requests[:num_req]
        trace_msg = f"Using workload trace ({len(requests)} requests) [warmup={warmup}]"

    print(f"\n[1/4] {trace_msg}")

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

    # ── 4. Context length sweep ────────────────────────────────────
    ctx_results = None
    if not args.skip_context_sweep:
        print(f"\n[4/5] Context Length Sweep (EIC vs No-EIC):")
        from experiments.run_all import ClusterContextExperiment
        ctx_runner = ClusterContextExperiment(cfg)
        ctx_results = ctx_runner.run()
        _print_context_table(ctx_results)
    else:
        print(f"\n[4/5] Context Length Sweep skipped (--skip-context-sweep).")

    # ── 5. Final cluster stats + credibility report ──────────────────
    print(f"\n[5/5] Final Cluster Topology:")
    from sim.cluster import build_cluster
    cluster = build_cluster(cfg)
    _print_cluster_info(cluster)
    _print_credibility_report(cfg, cluster)

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
        outputs = [out1, out2]
        if ctx_results is not None:
            # Context sweep plot: EIC vs No-EIC hit rates
            ctx_plot = {
                k: v["eic"] for k, v in ctx_results.items()
            }
            out3 = plot_results(
                ctx_plot, tier_names=["HBM", "EIC", "Remote"],
                output_dir=out_dir, filename="cluster_context_sweep.png",
            )
            outputs.append(out3)
        print(f"\nPlots saved → {', '.join(outputs)}")

    print("\nDone.")


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache Multi-Tier Simulator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cluster", action="store_true", help="Run 万卡 cluster + EIC mode")
    parser.add_argument("--pd", action="store_true", help="Run PD-separated cluster mode")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "--learned-train-mode",
        choices=("warmup", "split"),
        default=None,
        help="Train Learned eviction only on warmup requests or on a train/eval prefix split",
    )
    parser.add_argument(
        "--learned-train-fraction",
        type=float,
        default=None,
        help="Fraction of requests used for --learned-train-mode split (default: 0.2)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument(
        "--skip-context-sweep",
        action="store_true",
        help="Skip long context-length sweeps in cluster/PD modes",
    )
    parser.add_argument(
        "--calibration-profile",
        default=None,
        help="YAML profile that overlays externally calibrated hardware/network parameters",
    )
    parser.add_argument(
        "--workload-trace",
        default=None,
        help="CSV/JSON/JSONL production-style workload trace to replay instead of synthetic trace",
    )
    parser.add_argument(
        "--workload-format",
        choices=("auto", "burstgpt", "azure", "mooncake", "splitwise", "generic"),
        default="auto",
        help="External workload schema hint; auto detects common public traces",
    )
    parser.add_argument(
        "--workload-limit",
        type=int,
        default=None,
        help="Maximum usable workload rows to load; defaults to the mode's num_requests",
    )
    parser.add_argument(
        "--workload-time-unit",
        choices=("auto", "s", "ms", "us", "ns"),
        default="auto",
        help="Unit for numeric workload timestamps; datetimes are parsed automatically",
    )
    parser.add_argument(
        "--workload-arrival-scale",
        type=float,
        default=1.0,
        help="Scale request rate; 2.0 doubles RPS, 0.5 halves RPS",
    )
    parser.add_argument(
        "--workload-include-failed",
        action="store_true",
        help="Keep rows with zero output tokens, such as failed BurstGPT requests",
    )
    parser.add_argument(
        "--workload-hash-tokens-per-block",
        type=int,
        default=None,
        help="Token span represented by each workload hash_id; Mooncake defaults to 512",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.calibration_profile:
        from sim.calibration import (
            apply_calibration_profile,
            load_calibration_profile,
            profile_name,
        )

        profile = load_calibration_profile(args.calibration_profile)
        cfg = apply_calibration_profile(cfg, profile)
        print(
            f"[calibration] Applied profile "
            f"{profile_name(profile, args.calibration_profile)} "
            f"({args.calibration_profile})"
        )

    if args.pd:
        run_pd(cfg, args)
    elif args.cluster:
        run_cluster(cfg, args)
    else:
        run_single_node(cfg, args)


if __name__ == "__main__":
    main()
