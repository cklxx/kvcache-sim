# kvcache-sim

> A configurable multi-tier KV-cache simulator for LLM serving — eviction, prefetch, and learned policies.

Simulates how a production vLLM-style KV cache behaves across **HBM → DRAM → SSD** storage tiers under realistic multi-turn conversation workloads. Compares six cache policies — from simple LRU to an offline Belady Oracle — and visualises the trade-offs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        kvcache-sim                              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │    Trace     │    │    Router    │    │   Experiments    │  │
│  │  Generator   │───▶│  (prefix     │    │   run_all.py     │  │
│  │ (synthetic   │    │   trie +     │    │   plot.py        │  │
│  │  multi-turn) │    │   worker     │    └────────┬─────────┘  │
│  └──────┬───────┘    │   pool)      │             │            │
│         │            └──────┬───────┘             │            │
│         │                   │                     │            │
│         ▼                   ▼                     ▼            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      CacheManager                        │  │
│  │                                                          │  │
│  │   read() ──▶  HBM ──▶ DRAM ──▶ SSD ──▶ miss             │  │
│  │   write() ──▶ HBM  (async backup → DRAM/SSD)            │  │
│  │   tick()  ──▶ prefetch + background eviction            │  │
│  │                                                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │  │
│  │  │Eviction  │  │Prefetch  │  │    StorageTiers        │  │  │
│  │  │Policy    │  │Policy    │  │  HBM / DRAM / SSD      │  │  │
│  │  │LRU|ARC   │  │NoPrefetch│  │  (metadata only,       │  │  │
│  │  │Learned   │  │Session   │  │   latency simulated)   │  │  │
│  │  │Belady    │  │Aware     │  └───────────────────────┘  │  │
│  │  └──────────┘  └──────────┘                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Learned Policy (LightGBM)                  │  │
│  │   features.py → train.py → model.py → LearnedPolicy      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/cklxx/kvcache-sim.git
cd kvcache-sim

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

The demo will:
- Generate a synthetic multi-turn conversation trace
- Train a LightGBM model on the trace
- Run all 6 policy configurations
- Print a results table
- Save `results/policy_comparison.png`

---

## File Structure

```
kvcache-sim/
├── sim/
│   ├── storage.py        # StorageTier, KVBlock
│   ├── policies.py       # LRU, ARC, Learned, BeladyOracle, NoPrefetch, SessionAwarePrefetch
│   ├── cache_manager.py  # Multi-tier cache orchestrator
│   ├── router.py         # Prefix trie + worker pool + request router
│   └── metrics.py        # Counters, KPIs, matplotlib visualiser
├── trace/
│   ├── generator.py      # Synthetic multi-turn trace generation
│   └── replay.py         # Trace replay loop
├── learned/
│   ├── features.py       # 8-dim feature engineering
│   ├── train.py          # LightGBM training pipeline
│   └── model.py          # Online inference wrapper
├── experiments/
│   ├── run_all.py        # Run all 6 policy configs
│   └── plot.py           # matplotlib comparison plots
├── config.yaml           # Hardware + workload configuration
├── requirements.txt
└── main.py               # Entry point
```

---

## Policy Descriptions

| # | Policy | Eviction | Prefetch | Notes |
|---|--------|----------|----------|-------|
| 1 | **Baseline LRU** | Least Recently Used | None | Classic FIFO-based |
| 2 | **+ARC** | Adaptive Replacement Cache | None | Balances recency & frequency |
| 3 | **+SessionPrefetch** | LRU | Session-Aware | Predicts next blocks from session patterns |
| 4 | **+SelectiveWrite** | LRU | None | Only caches shallow-prefix blocks (depth ≤ 3) |
| 5 | **+Learned** | LightGBM reuse predictor | None | Trained on trace; predicts reuse distance |
| 6 | **Belady Oracle** | Optimal (offline) | None | Upper bound — evicts farthest-future block |

---

## Tier Hardware Parameters (defaults)

| Tier | Capacity | Read BW | Write BW | Read Latency |
|------|----------|---------|----------|--------------|
| HBM (GPU) | 80 GB | 3 200 GB/s | 3 200 GB/s | 0.001 ms |
| DRAM | 512 GB | 200 GB/s | 200 GB/s | 0.1 ms |
| SSD | 4 000 GB | 7 GB/s | 4 GB/s | 0.5 ms |

All parameters are tunable in `config.yaml`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `hit_rate` | Fraction of requests served from any cache tier |
| `hit_rate_HBM/DRAM/SSD` | Per-tier contribution to overall hit rate |
| `avg_hit_latency_ms` | Mean latency for cache-hit requests |
| `evictions` | Total block evictions across all tiers |
| `prefetches` | Proactive loads triggered by prefetch policy |

---

## Example Results

```
╭──────────────────────┬──────────┬─────────┬─────────┬─────────┬──────────────────┬─────────────┬────────────╮
│ Policy               │ HitRate  │   HBM   │  DRAM   │   SSD   │ AvgLatency(ms)   │   Evictions │  Prefetches│
├──────────────────────┼──────────┼─────────┼─────────┼─────────┼──────────────────┼─────────────┼────────────┤
│ Baseline LRU         │  72.14%  │  68.30% │   2.90% │   0.94% │         0.0041   │        1842 │          0 │
│ +ARC                 │  74.61%  │  71.10% │   2.70% │   0.81% │         0.0038   │        1765 │          0 │
│ +SessionPrefetch     │  76.83%  │  70.50% │   5.10% │   1.23% │         0.0052   │        1901 │        387 │
│ +SelectiveWrite      │  69.42%  │  66.80% │   1.90% │   0.72% │         0.0034   │         912 │          0 │
│ +Learned             │  75.29%  │  72.40% │   2.10% │   0.79% │         0.0037   │        1634 │          0 │
│ Belady Oracle        │  84.17%  │  80.20% │   2.90% │   1.07% │         0.0038   │         973 │          0 │
╰──────────────────────┴──────────┴─────────┴─────────┴─────────┴──────────────────┴─────────────┴────────────╯
```

*(Actual numbers vary with config and random seed.)*

---

## Configuration

Edit `config.yaml` to tune:

```yaml
hardware:
  hbm:
    capacity_gb: 80          # GPU HBM capacity
    read_bw_gbps: 3200

trace:
  num_sessions: 200          # number of conversations
  turns_per_session: 6       # turns per conversation
  qps: 20.0                  # requests per second

experiments:
  num_requests: 2000         # total requests to simulate
  warmup_requests: 200       # warm-up phase (metrics excluded)
```

---

## License

MIT © 2025 cklxx
