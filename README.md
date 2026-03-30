# kvcache-sim

> Multi-tier KV-cache simulator for LLM serving — from single-node to 万卡 cluster with EIC disaggregated memory and Prefill-Decode separation.

Three simulation modes:
- **Single-node**: 4 workers, HBM → DRAM → SSD hierarchy, 6 eviction/prefetch policies
- **Cluster** (万卡): 10,240 GPUs across 160 racks, shared EIC (CXL/RDMA) per rack, prefix-aware routing
- **PD Separated**: Prefill-Decode disaggregated serving with radix tree KV cache, continuous batching, KV transfer modeling

---

## Architecture

### Single-Node Mode

```
  TraceGenerator ──▶ Router (prefix trie) ──▶ Worker[0..3]
                                                  │
                                            CacheManager
                                     HBM ──▶ DRAM ──▶ SSD
                                                  │
                                        EvictionPolicy (LRU/ARC/Learned/Belady)
                                        PrefetchPolicy (None/SessionAware)
```

### Cluster Mode (万卡 + EIC)

```
  Cluster: 10,240 GPUs  (simulating 128 = 8 racks × 16 GPUs)
  ┌──────────────────────────────────────────────────────────────────┐
  │  ClusterRouter (session affinity + prefix scoring)               │
  ├──────────────────────────────────────────────────────────────────┤
  │  Rack 0                          Rack 1              ...  Rack 7│
  │  ┌─────────────────────┐        ┌────────────────┐              │
  │  │ GPU 0  GPU 1 ... 15 │        │ GPU 16 ... 31  │              │
  │  │ ┌───┐  ┌───┐       │        │ ┌───┐          │              │
  │  │ │HBM│  │HBM│  ...  │        │ │HBM│   ...    │              │
  │  │ └─┬─┘  └─┬─┘       │        │ └─┬─┘          │              │
  │  │   └───┬───┘         │        │   └─────┬──────│              │
  │  │  ┌────▼────────┐    │        │  ┌──────▼─────┐│              │
  │  │  │  EIC Pool   │    │        │  │  EIC Pool  ││              │
  │  │  │ (shared CXL)│    │        │  │ (shared)   ││              │
  │  │  └─────────────┘    │        │  └────────────┘│              │
  │  └─────────────────────┘        └────────────────┘              │
  │  Network: intra-rack 3μs (RDMA) │ cross-rack 15μs │ SSD 200μs  │
  └──────────────────────────────────────────────────────────────────┘
```

### PD Separation Mode

```
  PDCluster: 128 GPUs (32 Prefill + 96 Decode, P:D = 1:3)
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Request ──▶ PrefillRouter ──▶ PrefillNode                      │
  │              (prefix match      │                                │
  │               + load balance)   │ RadixTree lookup               │
  │                                 │ (prefix sharing, ref counting) │
  │                                 │ Compute new KV blocks          │
  │                                 │ SessionAware prefetch          │
  │                                 ▼                                │
  │                          KV Transfer (RDMA push)                 │
  │                          0.01ms @ 100 Gbps                      │
  │                                 │                                │
  │              DecodeRouter ◀─────┘                                │
  │              (same-rack pref    │                                │
  │               + capacity)       ▼                                │
  │                           DecodeNode                             │
  │                           │ Continuous batching                  │
  │                           │ All active sequences per step        │
  │                           │ Memory-bandwidth bound               │
  │                           ▼                                      │
  │                     Output tokens                                │
  │                                                                  │
  │  Per rack: [P P P P | D D D D D D D D D D D D] + shared EIC     │
  └──────────────────────────────────────────────────────────────────┘

  TTFT = queue_wait + prefill_compute + kv_transfer + first_decode
  Key insight: Unified GPU is blocked for prefill + ALL decode steps.
               PD separation frees the prefill GPU after compute only.
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Single-node demo (6 policies, HBM → DRAM → SSD)
python main.py

# 万卡 cluster + EIC demo
python main.py --cluster

# PD separation analysis (unified vs PD, P:D ratio sweep, transfer strategies)
python main.py --pd
```

---

## File Structure

```
kvcache-sim/
├── sim/
│   ├── storage.py        # StorageTier, KVBlock
│   ├── policies.py       # LRU, ARC, Learned, BeladyOracle, prefetch policies
│   ├── cache_manager.py  # Single-node multi-tier cache orchestrator
│   ├── router.py         # Prefix trie + worker pool router
│   ├── metrics.py        # Counters, KPIs, matplotlib visualiser
│   ├── network.py        # Network latency model (intra/cross-rack, P2P RDMA)
│   ├── cluster.py        # GPUNode, EICPool, Rack, Cluster, ClusterRouter
│   ├── radix_tree.py     # KV cache radix tree (prefix sharing, ref counting)
│   ├── pd_nodes.py       # PrefillNode, DecodeNode, compute models
│   ├── pd_router.py      # PrefillRouter, DecodeRouter, PDOrchestrator
│   ├── pd_cluster.py     # PDCluster, PDConfig, build_pd_cluster
│   ├── pd_metrics.py     # TTFT/TPOT distributions, transfer stats
│   └── kv_transfer.py    # KV transfer protocol (push/pull/pipeline)
├── trace/
│   ├── generator.py      # Synthetic multi-turn trace (shared system prompts)
│   ├── replay.py         # Single-node trace replay
│   ├── cluster_replay.py # Cluster-scale trace replay
│   └── pd_replay.py      # PD-separated trace replay
├── learned/
│   ├── features.py       # 8-dim feature engineering
│   ├── train.py          # LightGBM training pipeline
│   └── model.py          # Online inference wrapper
├── experiments/
│   ├── run_all.py        # Single-node + cluster experiments
│   ├── pd_experiments.py # PD separation experiments
│   └── plot.py           # matplotlib comparison plots
├── config.yaml           # Full configuration (all three modes)
├── requirements.txt
└── main.py               # Entry point (--cluster / --pd)
```

---

## PD Separation: Key Concepts

### Why PD Separation?

In **unified** serving, a GPU does prefill (process prompt) then decode (generate tokens) sequentially. The decode phase (128 tokens × 8.75ms = 1.12s for 7B) **blocks** the GPU from accepting new prefill requests — this is head-of-line blocking.

**PD separation** dedicates GPUs to each phase:
- **Prefill nodes**: Compute-bound, process prompts, free immediately after
- **Decode nodes**: Memory-bandwidth-bound, generate tokens via continuous batching
- **KV transfer**: RDMA push of KV cache from prefill → decode node

### Components

| Component | Description |
|-----------|-------------|
| **RadixTree** | Prefix-sharing block tree with reference counting and leaf-only eviction |
| **PrefillNode** | RadixTree-backed cache + session-aware prefetch + continuous batching |
| **DecodeNode** | Receives KV via RDMA, continuous batching of active sequences |
| **KVTransferModel** | Push/pull strategies, pipeline support, bandwidth modeling |
| **PrefillRouter** | Prefix cache hit scoring + queue-aware load balancing |
| **DecodeRouter** | Same-rack preference (fast transfer) + capacity-aware |

### Compute Model (H100, 7B)

| Phase | Formula | Value |
|-------|---------|-------|
| Prefill | `2 × params / TFLOPS` | 0.035 ms/token |
| Decode | `2 × params / HBM_BW` | 8.75 ms/token |
| Decode (64 seq batch) | base + marginal KV overhead | ~9.8 ms/step |
| KV transfer (1K tokens) | `bytes / RDMA_BW` | 0.005 ms |
| KV transfer (128K tokens) | `bytes / RDMA_BW` | 0.65 ms |

---

## PD Separation: Example Results

```
================================================================
  kvcache-sim  —  PD Separation Mode
  PDCluster: 128 GPUs (32P + 96D, ratio 1:3) × 8 racks
================================================================

Unified vs PD-Separated:
╭──────────────┬──────────┬──────────┬──────────┬──────────┬───────────╮
│ Config       │ TTFT_p50 │ TPOT_avg │ Prefill  │ Transfer │ QueueWait │
├──────────────┼──────────┼──────────┼──────────┼──────────┼───────────┤
│ Unified      │ 15840 ms │   8.8 ms │  12.7 ms │     0 ms │  15353 ms │
│ PD Separated │   847 ms │   8.9 ms │  12.6 ms │  0.01 ms │    853 ms │
╰──────────────┴──────────┴──────────┴──────────┴──────────┴───────────╯

  PD separation: 18.7× lower TTFT
  Root cause: Unified GPU blocked by decode (128 × 8.75ms = 1120ms per request)
  Transfer overhead: 0.01ms — negligible at 100 Gbps RDMA

P:D Ratio Sweep:
╭──────────┬──────────┬───────────┬───────────╮
│ P:D      │ TTFT_p50 │ PrefixHit │ SameRack  │
├──────────┼──────────┼───────────┼───────────┤
│ 1:1      │   427 ms │    65.3%  │     92%   │
│ 1:2      │   701 ms │    60.8%  │     80%   │
│ 1:3      │   853 ms │    66.7%  │     97%   │
│ 1:4      │  1258 ms │    52.4%  │     59%   │
│ 1:7      │  2302 ms │    34.6%  │     32%   │
╰──────────┴──────────┴───────────┴───────────╯

  Fewer prefill nodes → higher TTFT (queue buildup)
  Fewer prefill nodes → lower prefix cache hit (less cache capacity)
  Fewer prefill nodes → more cross-rack transfers (fewer co-located P-D pairs)
```

---

## Single-Node: Policy Comparison

| # | Policy | Eviction | Prefetch | Notes |
|---|--------|----------|----------|-------|
| 1 | Baseline LRU | Least Recently Used | None | Classic, near-optimal for sequential workloads |
| 2 | +ARC | Adaptive Replacement Cache | None | Balances recency & frequency (T1/T2 + ghost lists) |
| 3 | +SessionPrefetch | LRU | Session-Aware | Predicts next blocks from session patterns |
| 4 | +SelectiveWrite | LRU | None | Only caches shallow-prefix blocks (depth <= 3) |
| 5 | +Learned | LightGBM reuse predictor | None | Trained on trace; predicts reuse distance |
| 6 | Belady Oracle | Optimal (offline) | None | Upper bound — evicts farthest-future block |

---

## Configuration

All parameters in `config.yaml`. Key PD separation settings:

```yaml
pd_separation:
  pd_ratio: [1, 3]              # Prefill:Decode GPU ratio
  compute:
    prefill_tflops: 800          # H100 FP16 effective TFLOPS
    decode_memory_bw_gbps: 3200  # H100 HBM bandwidth
    model_params_b: 7            # Model size (billions)
    prefill_batch_efficiency: 0.85
    decode_kv_overhead_factor: 0.02
  transfer:
    strategy: push               # push | pull | pull_on_demand
    rdma_bw_gbps: 100
    pipelining: true
```

---

## What You Can Optimize With This Simulator

1. **P:D Ratio Selection** — Find optimal prefill/decode GPU split for your QPS and prompt lengths
2. **Prefix Cache Capacity Planning** — How much HBM/EIC to allocate for KV cache vs model weights
3. **Interconnect Bandwidth ROI** — Compare 25/50/100/200 Gbps for KV transfer overhead
4. **Eviction Policy Selection** — LRU vs ARC vs Learned under different workload patterns
5. **EIC Sizing** — How much shared CXL memory per rack for cross-GPU prefix reuse
6. **Context Length Impact** — How 4K vs 32K vs 128K contexts affect cache dynamics and PD benefit

---

## License

MIT
