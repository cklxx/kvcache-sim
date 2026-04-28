from sim.pd_cluster import PDConfig, build_pd_cluster
from sim.pd_nodes import DecodeNode
from trace.generator import Request
from trace.pd_replay import PDReplayer


def _small_pd_config() -> dict:
    return {
        "cluster": {
            "simulate_racks": 1,
            "simulate_gpus_per_rack": 4,
            "gpu": {
                "hbm_capacity_gb": 1,
                "hbm_read_bw_gbps": 1000,
                "hbm_write_bw_gbps": 1000,
                "hbm_latency_ms": 0.001,
            },
            "eic": {
                "nodes_per_rack": 1,
                "capacity_per_node_gb": 1,
                "read_bw_gbps": 100,
                "write_bw_gbps": 100,
                "access_latency_ms": 0.005,
            },
            "network": {
                "intra_rack_latency_us": 3,
                "cross_rack_latency_us": 15,
                "remote_ssd_latency_us": 200,
                "nvlink_bw_gbps": 900,
                "nvlink_latency_us": 1,
                "gpus_per_node": 8,
            },
        },
        "pd_separation": {
            "enabled": True,
            "pd_ratio": [1, 1],
            "max_output_tokens": 8,
            "compute": {
                "prefill_tflops": 1000,
                "decode_memory_bw_gbps": 1000,
                "model_params_b": 1,
                "kv_bytes_per_token": 1024,
                "tokens_per_block": 16,
                "overhead_factor": 1.0,
                "prefill_batch_efficiency": 0.85,
                "decode_kv_overhead_factor": 0.02,
            },
            "transfer": {
                "strategy": "push",
                "rdma_bw_gbps": 12.5,
                "rdma_latency_us": 5,
                "pipelining": True,
                "pipeline_chunk_blocks": 2,
                "compression_ratio": 1.0,
            },
        },
    }


def test_pd_replay_keeps_overlapping_decode_sequences(monkeypatch) -> None:
    seen_active_counts = []
    original = DecodeNode._run_one_decode_step

    def wrapped(self):
        if self.active_count:
            seen_active_counts.append(self.active_count)
        return original(self)

    monkeypatch.setattr(DecodeNode, "_run_one_decode_step", wrapped)

    cfg = _small_pd_config()
    cluster = build_pd_cluster(cfg, PDConfig.from_config(cfg))
    requests = [
        Request(
            session_id=f"s{i}",
            turn_id=0,
            timestamp=i * 0.01,
            block_hashes=[f"s{i}:b{j}" for j in range(4)],
            block_size=1024,
            prompt_tokens=64,
        )
        for i in range(12)
    ]

    metrics = PDReplayer(cluster).run(requests)

    assert metrics.total_requests == len(requests)
    assert max(seen_active_counts) > 1
    assert all(node.active_count == 0 for node in cluster.decode_nodes)


def test_pd_replay_keeps_duplicate_session_turn_requests() -> None:
    cfg = _small_pd_config()
    cluster = build_pd_cluster(cfg, PDConfig.from_config(cfg))
    requests = [
        Request(
            session_id="same-session",
            turn_id=0,
            timestamp=0.0,
            block_hashes=[f"shared:b{j}" for j in range(4)],
            block_size=1024,
            prompt_tokens=64,
        ),
        Request(
            session_id="same-session",
            turn_id=0,
            timestamp=0.001,
            block_hashes=[f"shared:b{j}" for j in range(4)],
            block_size=1024,
            prompt_tokens=64,
        ),
    ]

    metrics = PDReplayer(cluster).run(requests)

    assert metrics.total_requests == len(requests)
    assert all(node.active_count == 0 for node in cluster.decode_nodes)
