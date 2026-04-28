"""
Network variance experiments.

Runs three focused checks:
  1. Transfer microbenchmark: same-link KV bursts with/without jitter/contention.
  2. PD replay: end-to-end TTFT/transfer sensitivity under the same variants.
  3. Transfer-sensitive PD replay: forced RDMA and no pipelining.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from sim.kv_transfer import KVTransferModel, TransferConfig
from sim.network import NetworkModel
from sim.pd_cluster import PDConfig, build_pd_cluster
from trace.generator import TraceGenerator
from trace.pd_replay import PDReplayer


VARIANTS: Dict[str, dict] = {
    "deterministic": {
        "jitter_cv": 0.0,
        "tail_jitter_prob": 0.0,
        "tail_jitter_multiplier": 1.0,
        "contention_enabled": False,
    },
    "jitter": {
        "jitter_cv": 0.25,
        "tail_jitter_prob": 0.02,
        "tail_jitter_multiplier": 4.0,
        "contention_enabled": False,
    },
    "contention": {
        "jitter_cv": 0.0,
        "tail_jitter_prob": 0.0,
        "tail_jitter_multiplier": 1.0,
        "contention_enabled": True,
    },
    "jitter+contention": {
        "jitter_cv": 0.25,
        "tail_jitter_prob": 0.02,
        "tail_jitter_multiplier": 4.0,
        "contention_enabled": True,
    },
}


def percentile(values: Iterable[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = (len(vals) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def with_network_variant(cfg: dict, variant: dict) -> dict:
    out = copy.deepcopy(cfg)
    net = out.setdefault("cluster", {}).setdefault("network", {})
    net.update(variant)
    net.setdefault("seed", cfg.get("trace", {}).get("seed", 42))
    return out


def compact_pd_config(cfg: dict, num_requests: int) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("cluster", {}).update(
        {
            "simulate_racks": 2,
            "simulate_gpus_per_rack": 8,
        }
    )
    out.setdefault("pd_separation", {})["max_output_tokens"] = 48
    out.setdefault("pd_trace", {}).update(
        {
            "num_sessions": max(200, num_requests // 2),
            "turns_per_session": 3,
            "prompt_tokens_min": 256,
            "prompt_tokens_max": 1024,
            "initial_context_tokens": 4096,
            "num_system_prompts": 12,
            "qps": 250.0,
            "seed": 42,
        }
    )
    return out


def run_transfer_microbenchmark(cfg: dict) -> None:
    print("\n[1/3] Transfer microbenchmark: 64 simultaneous same-rack RDMA KV transfers")
    print("variant              p50_ms     p95_ms     p99_ms    max_ms")
    for name, variant in VARIANTS.items():
        exp_cfg = with_network_variant(cfg, variant)
        net = NetworkModel.from_config(exp_cfg)
        model = KVTransferModel(
            TransferConfig.from_config(exp_cfg),
            net,
        )
        transfer_ms: List[float] = []
        for _ in range(64):
            timing = model.transfer_timing_ms(
                num_blocks=128,
                block_size=exp_cfg.get("cache", {}).get("block_size_bytes", 5_242_880),
                same_rack=True,
                src_gpu=0,
                dst_gpu=8,
                start_time_ms=0.0,
            )
            transfer_ms.append(timing.full_ms)
        print(
            f"{name:<18} "
            f"{percentile(transfer_ms, 50):9.3f} "
            f"{percentile(transfer_ms, 95):9.3f} "
            f"{percentile(transfer_ms, 99):9.3f} "
            f"{max(transfer_ms):9.3f}"
        )


def run_pd_variants(cfg: dict, num_requests: int, warmup: int) -> None:
    print("\n[2/3] PD end-to-end replay under network variants")
    base_cfg = compact_pd_config(cfg, num_requests)
    pt = base_cfg["pd_trace"]
    requests = TraceGenerator(
        num_sessions=pt["num_sessions"],
        turns_per_session=pt["turns_per_session"],
        prompt_tokens_min=pt["prompt_tokens_min"],
        prompt_tokens_max=pt["prompt_tokens_max"],
        initial_context_tokens=pt["initial_context_tokens"],
        qps=pt["qps"],
        block_size_bytes=base_cfg.get("cache", {}).get("block_size_bytes", 4096),
        num_system_prompts=pt["num_system_prompts"],
        seed=pt["seed"],
    ).generate()[:num_requests]

    print(f"trace: {len(requests)} requests, warmup={warmup}")
    print(
        "variant              ttft_p50  ttft_p95  ttft_p99  "
        "xfer_avg  xfer_p99  same_rack"
    )
    for name, variant in VARIANTS.items():
        exp_cfg = with_network_variant(base_cfg, variant)
        pd_config = PDConfig.from_config(exp_cfg)
        cluster = build_pd_cluster(exp_cfg, pd_config)
        metrics = PDReplayer(cluster, warmup_count=warmup).run(requests)
        print(
            f"{name:<18} "
            f"{metrics.ttft_p50:9.1f} "
            f"{percentile(metrics.ttft_ms, 95):9.1f} "
            f"{metrics.ttft_p99:9.1f} "
            f"{metrics.avg_transfer_ms:9.3f} "
            f"{percentile(metrics.transfer_ms, 99):9.3f} "
            f"{metrics.same_rack_ratio:9.1%}"
        )


def run_transfer_sensitive_pd(cfg: dict) -> None:
    print("\n[3/3] Transfer-sensitive PD replay: forced RDMA, no pipeline")
    base_cfg = compact_pd_config(cfg, num_requests=500)
    base_cfg["cluster"]["network"]["gpus_per_node"] = 1
    base_cfg["pd_separation"]["max_output_tokens"] = 8
    base_cfg["pd_separation"]["transfer"]["pipelining"] = False
    base_cfg["pd_separation"]["compute"]["prefill_tflops"] = 3000
    base_cfg["pd_separation"]["compute"]["model_params_b"] = 7
    pt = base_cfg["pd_trace"]
    pt.update(
        {
            "num_sessions": 220,
            "turns_per_session": 3,
            "qps": 20.0,
            "seed": 44,
        }
    )
    requests = TraceGenerator(
        num_sessions=pt["num_sessions"],
        turns_per_session=pt["turns_per_session"],
        prompt_tokens_min=pt["prompt_tokens_min"],
        prompt_tokens_max=pt["prompt_tokens_max"],
        initial_context_tokens=pt["initial_context_tokens"],
        qps=pt["qps"],
        block_size_bytes=base_cfg.get("cache", {}).get("block_size_bytes", 4096),
        num_system_prompts=pt["num_system_prompts"],
        seed=pt["seed"],
    ).generate()[:500]
    warmup = 50

    print(f"trace: {len(requests)} requests, warmup={warmup}")
    print(
        "variant              ttft_p50  ttft_p95  ttft_p99  "
        "xfer_avg  xfer_p99"
    )
    for name, variant in VARIANTS.items():
        exp_cfg = with_network_variant(base_cfg, variant)
        pd_config = PDConfig.from_config(exp_cfg)
        cluster = build_pd_cluster(exp_cfg, pd_config)
        metrics = PDReplayer(cluster, warmup_count=warmup).run(requests)
        print(
            f"{name:<18} "
            f"{metrics.ttft_p50:9.1f} "
            f"{percentile(metrics.ttft_ms, 95):9.1f} "
            f"{metrics.ttft_p99:9.1f} "
            f"{metrics.avg_transfer_ms:9.3f} "
            f"{percentile(metrics.transfer_ms, 99):9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pd-requests", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=80)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_transfer_microbenchmark(cfg)
    run_pd_variants(cfg, args.pd_requests, args.warmup)
    run_transfer_sensitive_pd(cfg)


if __name__ == "__main__":
    main()
