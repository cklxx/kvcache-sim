# Architecture: Full Computation Graph

## End-to-End Request Flow (PD Separated)

```
User Request (prompt: 10K tokens)
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│  PrefillRouter                                                      │
│  Score = prefix_match - queue_penalty + eic_bonus                   │
│  Session affinity (soft) — re-route if queue > 50ms                 │
│                                                                     │
│  Input:  block_hashes[625]  (10K tokens / 16 tok per block)         │
│  Output: selected PrefillNode                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PrefillNode (GPU)                                                  │
│                                                                     │
│  ┌─── 1. RadixTree Lookup ──────────────────────────────┐           │
│  │  Walk tree matching block_hashes[0..624]              │           │
│  │  Stop at first miss                                   │           │
│  │                                                       │           │
│  │  Result: 450 blocks HIT (prefix sharing)              │           │
│  │          175 blocks MISS → need compute                │           │
│  │  Time: O(1) per block (hash index lookup)             │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌─── 2. EIC Check (for blocks after radix miss) ───────┐           │
│  │  For each miss block, check shared EIC via RDMA       │           │
│  │  If hit: promote to local RadixTree, skip compute     │           │
│  │  Latency: 5μs per EIC lookup                          │           │
│  │                                                       │           │
│  │  Result: 10 blocks from EIC (cross-GPU prefix reuse)  │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌─── 3. Prefill Compute ───────────────────────────────┐           │
│  │  new_tokens = 165 blocks × 16 tok = 2,640 tokens     │           │
│  │  FLOPs = 2 × 140GB × 2,640 = 739 TFLOPs              │           │
│  │  Time = 739T / 800T = 924 ms                          │           │
│  │                                                       │           │
│  │  With continuous batching (batch_size=3):              │           │
│  │  Time = 924 × 0.85 = 785 ms                          │           │
│  │  (GPU utilisation improves with larger batches)        │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌─── 4. Insert New Blocks ─────────────────────────────┐           │
│  │  RadixTree.insert_sequence(165 new blocks)            │           │
│  │  RadixTree.acquire_sequence(625 blocks) → ref_count++ │           │
│  │  Async EIC backup: write 165 blocks to shared EIC     │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌─── 5. Prefetch (SessionAware) ───────────────────────┐           │
│  │  Record: prefetch.record_sequence(session, hashes)    │           │
│  │  Predict: next-turn blocks from pattern history       │           │
│  │  Pre-load: pull predicted blocks from EIC → RadixTree │           │
│  │                                                       │           │
│  │  Result: 5 blocks prefetched for next turn            │           │
│  │  Next turn will see higher prefix cache hit           │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  Output: PrefillResult                                              │
│    - block_hashes: [625 blocks]                                     │
│    - cached: 460, new: 165                                          │
│    - compute_time: 785 ms                                           │
│    - kv_bytes: 625 × 5MiB = 3,125 MiB                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KV Transfer (RDMA Push)                                            │
│                                                                     │
│  625 blocks × 5 MiB = 3,125 MiB                                    │
│                                                                     │
│  Same-rack (intra-rack RDMA):                                       │
│    base = 3μs (RDMA) + 5μs (latency) = 8μs                        │
│    transfer = 3,125 MiB / 12.5 GB/s ≈ 262 ms                       │
│    total ≈ 262.008 ms                                               │
│                                                                     │
│  Cross-rack:                                                        │
│    base = 15μs + 5μs = 20μs                                        │
│    transfer ≈ 262 ms (same bandwidth)                               │
│    total ≈ 262.020 ms                                               │
│                                                                     │
│  With pipelining (16 blocks per chunk):                             │
│    first_chunk = 16 × 5MiB / 12.5GB/s ≈ 6.7 ms                    │
│    → decode can start after ~6.7 ms (not full transfer)            │
│    → remaining transfer overlaps with decode                        │
│                                                                     │
│  Pull strategy adds +1 RTT (8-20 μs) for request message           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DecodeRouter                                                       │
│  Score = -active_sequences + same_rack_bonus(5) + kv_cached(0.5)   │
│                                                                     │
│  Prefer: same rack as prefill (fast transfer)                       │
│  Avoid:  nodes near max_concurrent_sequences (64)                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DecodeNode (GPU)                                                   │
│                                                                     │
│  ┌─── 1. Receive KV ────────────────────────────────────┐           │
│  │  Insert 625 blocks into local RadixTree               │           │
│  │  Acquire ref_count (prevent eviction during decode)   │           │
│  │  Register as active sequence                          │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌─── 2. Decode Steps (Continuous Batching) ────────────┐           │
│  │                                                       │           │
│  │  N = active_sequences on this GPU (e.g., 32)          │           │
│  │                                                       │           │
│  │  Per step: GPU reads ALL model weights once            │           │
│  │    base = 2 × 140GB / 3350 GB/s = 83.6 ms            │           │
│  │    kv_overhead = 83.6 × 0.02 × log2(32) = 8.36 ms   │           │
│  │    step_time = 83.6 + 8.36 = 92.0 ms                 │           │
│  │                                                       │           │
│  │  ALL 32 sequences advance 1 token per step            │           │
│  │  Effective TPOT = 92.0 ms/token (per sequence)        │           │
│  │  Throughput = 32 tokens / 92.0 ms = 348 tok/s         │           │
│  │                                                       │           │
│  │  128 output tokens × 92.0 ms = 11,776 ms total        │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌─── 3. Finish ────────────────────────────────────────┐           │
│  │  Release ref_count on 625 blocks                      │           │
│  │  Remove from active_sequences                         │           │
│  │  Blocks become evictable (leaf nodes with ref=0)      │           │
│  └───────────────────────────────────────────────────────┘           │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Timing Breakdown (10K token request, 128 output tokens)            │
│                                                                     │
│  queue_wait:     ~850 ms  (depends on prefill node load)            │
│  prefill_compute: 785 ms (2,640 new tokens × 0.35 ms/tok × 0.85)   │
│  kv_transfer:      6.7 ms (pipelined first chunk)                  │
│  first_decode:     92 ms (one step with 32 concurrent seqs)        │
│  ─────────────────────────                                          │
│  TTFT:           ~1,734 ms                                          │
│                                                                     │
│  remaining_decode: 11,684 ms (127 × 92.0 ms)                      │
│  ─────────────────────────                                          │
│  E2E:            ~13,418 ms                                         │
│                                                                     │
│  Compare UNIFIED mode (same GPU does both):                         │
│    queue_wait:     ~15,000 ms  (blocked by other requests' decode)  │
│    prefill_compute: 785 ms                                          │
│    decode:          11,776 ms                                       │
│    ─────────────────────────                                        │
│    TTFT:           ~15,878 ms  (queue + prefill + first decode)     │
│    GPU blocked for prefill+decode = 12,561 ms per request           │
└─────────────────────────────────────────────────────────────────────┘
```

## KV Cache Lifecycle

```
Block Birth → Active Use → Eviction → EIC → Cross-GPU Reuse

  1. COMPUTE: PrefillNode computes KV for new tokens
     Block created in RadixTree, ref_count=1

  2. ACTIVE: DecodeNode uses block during generation
     ref_count > 0 → cannot be evicted

  3. RELEASE: Decode completes, ref_count decremented
     Block becomes evictable leaf in RadixTree

  4. EVICT: When HBM is full, LRU evicts leaf nodes
     ┌─────────────┐
     │  RadixTree   │    Only leaves with ref_count=0
     │              │    are eligible for eviction.
     │   A          │    Internal nodes (shared prefixes)
     │  / \         │    are protected as long as they
     │ B   C (leaf) │ ← evict C first (LRU leaf)
     │ |            │    B is protected (has child D)
     │ D (leaf)     │
     └─────────────┘
     Evicted block → async write to EIC

  5. EIC: Block lives in shared rack memory
     Other GPUs in same rack can read it via CXL/RDMA
     EIC has its own LRU eviction when full

  6. REUSE: Another GPU needs same prefix
     PrefillNode.prefill() → RadixTree miss → EIC hit!
     Promote back to local RadixTree, skip compute
     This is cross-GPU prefix sharing via EIC
```

## Memory Hierarchy (70B model on H100)

```
                    Capacity    Latency    Bandwidth    Blocks
                    ─────────   ─────────  ──────────   ──────
  GPU HBM           5 GB       1 μs       3,350 GB/s     953
       ↓ evict
  EIC (rack CXL)   128 GB      5 μs       100 GB/s    24,414
       ↓ evict
  Remote SSD        3.2 TB     200 μs     7 GB/s      610,351
       ↓ miss
  Recompute         ∞          workload   (GPU bound)  —

  Each block = 5 MiB (16 tokens × 320 KiB/token)

  Working set for 10K context request = 625 blocks ≈ 3.1 GiB
  HBM holds ~1 full 10K request per GPU
  EIC holds ~40 full 10K requests per rack
```

## Unified vs PD: Why the Difference

```
UNIFIED MODE (one GPU does both):
  ┌──────────────────────────────────────────────────┐
  │ GPU Timeline:                                     │
  │                                                   │
  │ |--prefill--|--------decode (128 tokens)--------| │
  │ | 785 ms   |          11,776 ms                 | │
  │ |  GPU computing KV  |  GPU reading weights     | │
  │ |  (compute bound)   |  (memory BW bound)       | │
  │                                                   │
  │ Total GPU occupation: 12,561 ms per request       │
  │                                                   │
  │ Next prefill must wait 1,255 ms → HEAD-OF-LINE    │
  │ BLOCKING. At 1 req/s/GPU, queue grows rapidly.    │
  └──────────────────────────────────────────────────┘

PD SEPARATED MODE:
  ┌──────────────────────────────────────────────────┐
  │ Prefill GPU:                                      │
  │ |--prefill--|--prefill--|--prefill--|  ...         │
  │ | 785 ms  | 600 ms  | 900 ms  |                 │
  │ GPU free after each prefill!                      │
  │ Throughput: ~12 req/s/GPU                         │
  │                                                   │
  │ Decode GPU (continuous batching, 32 seqs):        │
  │ |step|step|step|step|step|  ...                   │
  │ |92ms|92ms|92ms|92ms|92ms|                        │
  │ Each step advances ALL 32 sequences               │
  │ Throughput: 32 × (1000/92) = 348 tok/s           │
  │                                                   │
  │ KV Transfer (RDMA, pipelined):                    │
  │ Overlaps with decode startup                      │
  │ First chunk arrives in ~6.7 ms                    │
  └──────────────────────────────────────────────────┘
```
