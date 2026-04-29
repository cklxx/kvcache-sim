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
  │                          pipelined first-chunk latency           │
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

# Replay a production-style Azure LLM trace instead of a synthetic trace
python main.py --pd \
  --workload-trace /path/to/AzureLLMInferenceTrace_code.csv \
  --workload-format azure \
  --no-plot --skip-context-sweep
```

---

## Real Workload Traces

`trace/workload.py` loads public production-style LLM traces into the same
`Request` objects used by the synthetic generator. Timestamps are normalized to
milliseconds, input tokens become prompt KV blocks, and per-request output
tokens drive PD decode length when present.

Supported schemas:

| Source | Useful fields | Notes |
|--------|---------------|-------|
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | `Timestamp`, `Session ID`, `Model`, `Request tokens`, `Response tokens` | Real Azure-powered ChatGPT/GPT-4 workload; session IDs preserve conversation prefix reuse when present. |
| [Azure LLM Inference 2023/2024](https://github.com/Azure/AzurePublicDataset) | `TIMESTAMP`, `ContextTokens`, `GeneratedTokens` | Production Azure traces used by Splitwise and DynamoLLM; no prompt text, only token counts. |
| [Mooncake traces](https://huggingface.co/datasets/valeriol29/mooncake-traces) | `timestamp`, `input_length`, `output_length`, `hash_ids` | Best fit for KV-cache experiments because `hash_ids` preserve prefix-sharing relationships. |
| [SplitwiseSim](https://github.com/mutinifni/splitwise-sim) | `arrival_timestamp`, `prompt_size`, `token_size` | Compatible with Splitwise-style generated traces. |

Examples:

```bash
# BurstGPT CSV, excluding failed rows with zero response tokens by default
python main.py --pd \
  --workload-trace /path/to/BurstGPT_1.csv \
  --workload-format burstgpt \
  --workload-limit 4000 \
  --no-plot --skip-context-sweep

# Mooncake-style JSONL/CSV with hash_ids
python main.py --cluster \
  --workload-trace /path/to/mooncake.jsonl \
  --workload-format mooncake \
  --workload-time-unit ms \
  --no-plot --skip-context-sweep
```

If a trace has `hash_ids`, the loader uses them as the actual KV block IDs.
Otherwise it synthesizes deterministic block IDs from `session_id` when present
and falls back to unique per-request blocks. This keeps token-count-only traces
useful without pretending they contain exact prefix-sharing metadata.

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
│   ├── calibration.py    # External benchmark/simulator profile overlays
│   ├── cluster.py        # GPUNode, EICPool, Rack, Cluster, ClusterRouter
│   ├── radix_tree.py     # KV cache radix tree (prefix sharing, ref counting)
│   ├── pd_nodes.py       # PrefillNode, DecodeNode, compute models
│   ├── pd_router.py      # PrefillRouter, DecodeRouter, PDOrchestrator
│   ├── pd_cluster.py     # PDCluster, PDConfig, build_pd_cluster
│   ├── pd_metrics.py     # TTFT/TPOT distributions, transfer stats
│   └── kv_transfer.py    # KV transfer protocol (push/pull/pipeline)
├── trace/
│   ├── generator.py      # Synthetic multi-turn trace (shared system prompts)
│   ├── workload.py       # Public CSV/JSONL workload loader
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
│   ├── network_variance.py # Jitter/contention sensitivity experiments
│   └── plot.py           # matplotlib comparison plots
├── profiles/
│   └── h100_70b_reference.yaml # Example calibrated overlay
├── config.yaml           # Full configuration (all three modes)
├── requirements.txt
└── main.py               # Entry point (--cluster / --pd)
```

---

## PD Separation: Key Concepts

### Why PD Separation?

In **unified** serving, a GPU does prefill (process prompt) then decode (generate tokens) sequentially. With the default 70B profile, the decode phase (128 tokens × ~83.6ms) can occupy a GPU for ~10.7s, blocking new prefill work — this is head-of-line blocking.

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

### Compute Model (H100, 70B default profile)

| Phase | Formula | Value |
|-------|---------|-------|
| Prefill | `2 × params / TFLOPS` | ~0.35 ms/token |
| Decode | `2 × params / HBM_BW` | ~83.6 ms/token |
| Decode (64 seq batch) | base + marginal KV overhead | ~93.6 ms/step |
| KV transfer first chunk | `16 blocks × 5 MiB / 12.5 GB/s` | ~6.7 ms |
| KV transfer full 8K prompt | `512 blocks × 5 MiB / 12.5 GB/s` | ~215 ms |

---

## PD Separation: Example Output

```
================================================================
  kvcache-sim  —  PD Separation Mode
  PDCluster: 128 GPUs (32P + 96D, ratio 1:3) × 8 racks
================================================================

The exact numbers are workload-dependent. The PD replayer now keeps decode
sequences active across requests, so TTFT/TPOT reflect queueing, decode overlap,
P:D ratio, prompt length, output length, and KV transfer settings.

Key interpretation:
- Unified serving blocks a GPU for prefill plus the full decode phase.
- PD serving frees prefill GPUs after prefill, then admits transferred KV into decode nodes.
- Transfer reports both full KV movement and pipelined first-chunk latency for TTFT.
- P:D sweeps are meaningful only for the configured workload and hardware profile.
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
    model_params_b: 70           # Model size (billions)
    kv_bytes_per_token: 327680   # 320 KiB for Llama-3-70B GQA
    tokens_per_block: 16
    prefill_batch_efficiency: 0.85
    decode_kv_overhead_factor: 0.02
  transfer:
    strategy: push               # push | pull | pull_on_demand
    rdma_bw_gbps: 12.5           # effective GB/s for 100 Gbps RDMA
    pipelining: true
```

### Calibration Profiles

This simulator is intentionally system-level. It should not embed slow
cycle-level GPU, DRAM, or SSD simulators in the main replay loop. Instead, run
microbenchmarks or external simulators offline, then overlay their calibrated
parameters:

```bash
python main.py --config config.yaml \
  --calibration-profile profiles/h100_70b_reference.yaml --pd

python -m experiments.network_variance \
  --config config.yaml \
  --calibration-profile profiles/h100_70b_reference.yaml
```

Supported overlay sections are `hardware`, `cache`, `cluster`, and
`pd_separation`. Unknown sections are rejected so typoed calibration keys do not
silently change nothing.

Recommended calibration sources:

| Layer | Use in this repo | External source |
|-------|------------------|-----------------|
| LLM operator timing | `pd_separation.compute.*` | Vidur-style profiling or real serving traces |
| GPU/HBM bandwidth | `hardware.hbm`, `cluster.gpu` | Accel-Sim for kernels, plus bandwidth microbenchmarks |
| CPU DRAM/HBM details | `hardware.dram`, EIC params | Ramulator2 or DRAMsim3 |
| SSD/NVMe | `hardware.ssd` | MQSim or SimpleSSD |
| Network/RDMA/NVLink | `cluster.network`, transfer params | RDMA/NVLink microbenchmarks, ASTRA-sim/ns-3 for topology effects |

The default profile in `profiles/h100_70b_reference.yaml` mirrors the default
configuration and documents the expected shape for calibrated values.

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
