"""
PD Experiments — compare unified vs PD-separated, sweep P:D ratios,
transfer strategies, and chunked prefill.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple


from sim.cluster import build_cluster
from sim.kv_transfer import TransferConfig
from sim.pd_cluster import PDCluster, PDConfig, build_pd_cluster
from sim.pd_metrics import PDMetrics
from sim.pd_nodes import ComputeConfig
from trace.cluster_replay import ClusterReplayer
from trace.generator import Request, TraceGenerator
from trace.pd_replay import PDReplayer


# ======================================================================
# Unified baseline — simulate as if P+D on same GPU
# ======================================================================


def _unified_baseline_metrics(
    cfg: dict,
    requests: List[Request],
    warmup: int,
    compute_cfg: ComputeConfig,
    max_output_tokens: int,
) -> PDMetrics:
    """
    Unified mode with realistic queuing.

    Key difference from PD: each GPU does prefill + ALL decode steps
    before it can accept the next request.  This is the head-of-line
    blocking that PD separation eliminates.

    Service time per request = prefill_compute + decode_total
    In PD mode, the prefill GPU is free after prefill_compute alone.
    """
    # Run cache simulation for hit rate
    cluster = build_cluster(cfg)
    replayer = ClusterReplayer(cluster, warmup_count=warmup)
    base_metrics = replayer.run(requests)

    n_gpus = len(cluster.all_gpus)

    # Model queuing: each GPU is busy for (prefill + all decode) per request
    # Use earliest_available_time per GPU with session-affinity routing
    gpu_available = [0.0] * n_gpus
    session_gpu: Dict[str, int] = {}
    rr = 0

    pd = PDMetrics()
    pd.total_requests = base_metrics.total_requests
    pd.prefill_cache = base_metrics

    for req in requests[warmup:]:
        n_blocks = len(req.block_hashes)
        current_time = req.timestamp

        # Route: session affinity or round-robin
        if req.session_id in session_gpu:
            gid = session_gpu[req.session_id]
        else:
            # Pick least-loaded GPU
            gid = min(range(n_gpus), key=lambda g: gpu_available[g])
            session_gpu[req.session_id] = gid

        # Queue wait: GPU busy with previous request's decode
        effective_time = max(current_time, gpu_available[gid])
        queue_wait = effective_time - current_time

        # Compute
        cached = int(n_blocks * base_metrics.hit_rate)
        new_tokens = (n_blocks - cached) * compute_cfg.tokens_per_block
        prefill_ms = new_tokens * compute_cfg.prefill_ms_per_token
        # Unified decode: no continuous batching benefit (GPU is dedicated)
        first_decode_ms = compute_cfg.decode_ms_per_token
        decode_total_ms = max_output_tokens * first_decode_ms

        # GPU is busy for prefill + ALL decode (head-of-line blocking!)
        service_time = prefill_ms + decode_total_ms
        gpu_available[gid] = effective_time + service_time

        ttft = queue_wait + prefill_ms + first_decode_ms
        e2e = queue_wait + service_time

        pd.ttft_ms.append(ttft)
        pd.e2e_ms.append(e2e)
        pd.prefill_compute_ms.append(prefill_ms)
        pd.transfer_ms.append(0.0)
        pd.queue_wait_ms.append(queue_wait)
        pd.tpot_ms.append(first_decode_ms)
        pd.total_prefix_cache_hits += cached
        pd.total_blocks_processed += n_blocks

    return pd


# ======================================================================
# Experiment Runner
# ======================================================================


class PDExperimentRunner:
    """Experiments comparing unified vs PD-separated architectures."""

    def __init__(self, config: dict, warmup: int = 500) -> None:
        self.config = config
        self.warmup = warmup
        pd_cfg = PDConfig.from_config(config)
        self.compute_cfg = pd_cfg.compute
        self.max_output_tokens = pd_cfg.max_output_tokens

    # ── Experiment 1: Unified vs PD ──────────────────────────────────

    def run_unified_vs_pd(self, requests: List[Request]) -> Dict[str, PDMetrics]:
        """Compare unified cluster vs PD-separated with same total GPU count."""
        results: Dict[str, PDMetrics] = {}

        # Unified baseline
        print("  [Unified] …", end=" ", flush=True)
        m_uni = _unified_baseline_metrics(
            self.config, requests, self.warmup,
            self.compute_cfg, self.max_output_tokens,
        )
        results["Unified"] = m_uni
        print(f"TTFT_p50={m_uni.ttft_p50:.1f}ms  prefix_hit={m_uni.prefix_cache_hit_rate:.1%}")

        # PD separated
        print("  [PD Separated] …", end=" ", flush=True)
        pd_config = PDConfig.from_config(self.config)
        cluster = build_pd_cluster(self.config, pd_config)
        replayer = PDReplayer(cluster, warmup_count=self.warmup)
        m_pd = replayer.run(requests)
        results["PD Separated"] = m_pd
        print(
            f"TTFT_p50={m_pd.ttft_p50:.1f}ms  "
            f"prefix_hit={m_pd.prefix_cache_hit_rate:.1%}  "
            f"same_rack={m_pd.same_rack_ratio:.0%}"
        )

        return results

    # ── Experiment 2: P:D Ratio Sweep ────────────────────────────────

    def run_pd_ratio_sweep(self, requests: List[Request]) -> Dict[str, PDMetrics]:
        """Sweep P:D ratios to find optimal split."""
        ratios = self.config.get("pd_experiments", {}).get(
            "pd_ratios", [[1, 1], [1, 2], [1, 3], [1, 4], [1, 7]]
        )
        results: Dict[str, PDMetrics] = {}

        for p, d in ratios:
            name = f"P:D={p}:{d}"
            print(f"  [{name}] …", end=" ", flush=True)
            cfg = copy.deepcopy(self.config)
            cfg.setdefault("pd_separation", {})["pd_ratio"] = [p, d]
            pd_config = PDConfig.from_config(cfg)
            cluster = build_pd_cluster(cfg, pd_config)
            replayer = PDReplayer(cluster, warmup_count=self.warmup)
            m = replayer.run(requests)
            results[name] = m
            print(
                f"TTFT_p50={m.ttft_p50:.1f}ms  TPOT={m.tpot_avg:.1f}ms  "
                f"P={cluster.total_prefill_gpus} D={cluster.total_decode_gpus}"
            )

        return results

    # ── Experiment 3: Transfer Strategy ──────────────────────────────

    def run_transfer_strategy(self, requests: List[Request]) -> Dict[str, PDMetrics]:
        """Compare push vs pull, with/without pipelining."""
        configs = [
            ("Push + Pipeline", "push", True),
            ("Push (no pipeline)", "push", False),
            ("Pull + Pipeline", "pull", True),
            ("Pull (no pipeline)", "pull", False),
        ]
        results: Dict[str, PDMetrics] = {}

        for name, strategy, pipeline in configs:
            print(f"  [{name}] …", end=" ", flush=True)
            cfg = copy.deepcopy(self.config)
            tc = cfg.setdefault("pd_separation", {}).setdefault("transfer", {})
            tc["strategy"] = strategy
            tc["pipelining"] = pipeline
            pd_config = PDConfig.from_config(cfg)
            cluster = build_pd_cluster(cfg, pd_config)
            replayer = PDReplayer(cluster, warmup_count=self.warmup)
            m = replayer.run(requests)
            results[name] = m
            print(
                f"TTFT_p50={m.ttft_p50:.1f}ms  "
                f"avg_transfer={m.avg_transfer_ms:.3f}ms"
            )

        return results

    # ── Experiment 4: Context Length × PD Benefit ────────────────────

    def run_context_length_pd(self) -> Dict[str, Dict[str, PDMetrics]]:
        """
        Sweep context length: short → very long.
        For each, compare unified vs PD.
        """
        contexts = [
            ("Short (512 tok)",    256,  3000, 5, 5000, 500,  64,  128),
            ("Medium (4K tok)",   3072,  1500, 5, 4000, 400, 128,  512),
            ("Long (32K tok)",   30720,   400, 4, 1600, 200, 128,  512),
            ("VLong (128K tok)", 126976,  120, 3,  360,  40, 256, 1024),
        ]
        results: Dict[str, Dict[str, PDMetrics]] = {}

        for label, init_ctx, sessions, turns, num_req, warmup, min_t, max_t in contexts:
            print(f"  [{label}]", flush=True)
            gen = TraceGenerator(
                num_sessions=sessions,
                turns_per_session=turns,
                prompt_tokens_min=min_t,
                prompt_tokens_max=max_t,
                initial_context_tokens=init_ctx,
                num_system_prompts=20,
                qps=300.0,
                seed=42,
            )
            reqs = gen.generate()[:num_req]
            avg_blocks = sum(len(r.block_hashes) for r in reqs) / max(len(reqs), 1)
            print(f"    trace: {len(reqs)} reqs, avg {avg_blocks:.0f} blocks/req")

            # Unified
            m_uni = _unified_baseline_metrics(
                self.config, reqs, warmup,
                self.compute_cfg, self.max_output_tokens,
            )

            # PD
            pd_config = PDConfig.from_config(self.config)
            cluster = build_pd_cluster(self.config, pd_config)
            replayer = PDReplayer(cluster, warmup_count=warmup)
            m_pd = replayer.run(reqs)

            delta_ttft = m_uni.ttft_p50 - m_pd.ttft_p50
            print(
                f"    Unified TTFT_p50={m_uni.ttft_p50:.1f}ms  "
                f"PD TTFT_p50={m_pd.ttft_p50:.1f}ms  "
                f"delta={delta_ttft:+.1f}ms  "
                f"prefix_hit={m_pd.prefix_cache_hit_rate:.1%}"
            )

            results[label] = {"unified": m_uni, "pd": m_pd}

        return results
