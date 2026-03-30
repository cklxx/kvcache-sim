# kvcache-sim

> Multi-tier KV-cache simulator for LLM serving — from single-node to 万卡 cluster with EIC disaggregated memory.

Two simulation modes:
- **Single-node**: 4 workers, HBM → DRAM → SSD hierarchy, 6 eviction/prefetch policies
- **Cluster** (万卡): 10,240 GPUs across 160 racks, shared EIC (CXL/RDMA) per rack, prefix-aware routing

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
  │  │  │ (4 nodes    │    │        │  │ (4 nodes   ││              │
  │  │  │  shared CXL)│    │        │  │  shared)   ││              │
  │  │  └─────────────┘    │        │  └────────────┘│              │
  │  └─────────────────────┘        └────────────────┘              │
  │                                                                  │
  │  Network: intra-rack 3μs (RDMA) │ cross-rack 15μs │ SSD 200μs  │
  └──────────────────────────────────────────────────────────────────┘
```

**EIC (External Interconnect Cache)** = disaggregated CXL/RDMA memory shared across all GPUs in a rack. When GPU A evicts a block from HBM, it lands in the shared EIC. GPU B in the same rack can hit on that block — enabling cross-GPU prefix reuse without recomputation.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/cklxx/kvcache-sim.git && cd kvcache-sim

# 2. Install
pip install -r requirements.txt

# 3a. Single-node demo (6 policies)
python main.py

# 3b. 万卡 cluster + EIC demo
python main.py --cluster
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
│   ├── network.py        # Network latency model (intra-rack / cross-rack / SSD)
│   └── cluster.py        # GPUNode, EICPool, Rack, Cluster, ClusterRouter
├── trace/
│   ├── generator.py      # Synthetic multi-turn trace (shared system prompts)
│   ├── replay.py         # Single-node trace replay
│   └── cluster_replay.py # Cluster-scale trace replay
├── learned/
│   ├── features.py       # 8-dim feature engineering
│   ├── train.py          # LightGBM training pipeline
│   └── model.py          # Online inference wrapper
├── experiments/
│   ├── run_all.py        # Single-node + cluster experiments
│   └── plot.py           # matplotlib comparison plots
├── config.yaml           # Full configuration (single-node + cluster)
├── requirements.txt
└── main.py               # Entry point (--cluster for 万卡 mode)
```

---

## Policy Descriptions

| # | Policy | Eviction | Prefetch | Notes |
|---|--------|----------|----------|-------|
| 1 | Baseline LRU | Least Recently Used | None | Classic, near-optimal for sequential workloads |
| 2 | +ARC | Adaptive Replacement Cache | None | Balances recency & frequency (T1/T2 + ghost lists) |
| 3 | +SessionPrefetch | LRU | Session-Aware | Predicts next blocks from session patterns |
| 4 | +SelectiveWrite | LRU | None | Only caches shallow-prefix blocks (depth <= 3) |
| 5 | +Learned | LightGBM reuse predictor | None | Trained on trace; predicts reuse distance |
| 6 | Belady Oracle | Optimal (offline) | None | Upper bound — evicts farthest-future block |

---

## Cluster Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_gpus` | 10,240 | Full cluster GPU count |
| `simulate_racks` | 8 | Racks simulated in detail |
| `simulate_gpus_per_rack` | 16 | GPUs per simulated rack |
| `eic.nodes_per_rack` | 4 | EIC memory nodes per rack |
| `eic.capacity_per_node_gb` | 0.02 | Capacity per EIC node (scaled) |
| `network.intra_rack_latency_us` | 3 | GPU ↔ EIC latency (CXL/RDMA) |
| `network.cross_rack_latency_us` | 15 | Spine fabric latency |
| `network.remote_ssd_latency_us` | 200 | Disaggregated NVMe-oF |

---

## Example: Cluster EIC Sizing Results

```
================================================================
  kvcache-sim  —  万卡 Cluster + EIC Demo
  Full cluster: 10,240 GPUs  |  Simulating: 128 GPUs (8 racks × 16)
================================================================

╭───────────────────┬──────────┬────────┬───────┬──────────┬────────────┬───────────╮
│ Policy            │ HitRate  │ HBM    │ EIC   │ Remote   │ AvgLat(ms) │ Evictions │
├───────────────────┼──────────┼────────┼───────┼──────────┼────────────┼───────────┤
│ No EIC (HBM only) │ 71.20%   │ 71.20% │ 0.00% │ 0.00%   │ 0.001      │     5,257 │
│ EIC 2×20 MB       │ 71.39%   │ 71.17% │ 0.22% │ 0.00%   │ 0.001      │    25,238 │
│ EIC 4×20 MB       │ 71.39%   │ 71.18% │ 0.21% │ 0.00%   │ 0.001      │    21,167 │
│ EIC 4×50 MB       │ 71.39%   │ 71.20% │ 0.20% │ 0.00%   │ 0.001      │    22,031 │
│ EIC 8×50 MB       │ 71.39%   │ 71.19% │ 0.20% │ 0.00%   │ 0.001      │    21,710 │
╰───────────────────┴──────────┴────────┴───────┴──────────┴────────────┴───────────╯

Cluster Topology:
  128 GPUs × 8 racks, 32 EIC nodes
  Total HBM: 0.4 GB  |  Total EIC: 0.7 GB
  EIC utilization: R0=83%, R1=88%, R2=68%, R3=42%, R4=36%, R5=21%
  Cross-GPU EIC hits (shared prefix reuse): 512
```

Key findings:
- **EIC adds +0.19% hit rate** by catching HBM evictions and enabling cross-GPU prefix sharing
- **512 cross-GPU EIC hits** show different GPUs in the same rack reusing shared system prompt blocks via CXL
- **EIC utilization is skewed** across racks (83% → 0%) due to session-affinity routing concentrating traffic
- **ARC pushes 5.93% of hits to EIC** (vs 0.21% for LRU) — worse latency but same hit rate

---

## License

MIT
