from sim.cluster import ClusterRouter, build_cluster
from sim.policies import LearnedPolicy
from sim.storage import KVBlock
from trace.cluster_replay import ClusterReplayer
from trace.generator import TraceGenerator


def _cluster_config() -> dict:
    return {
        "trace": {"seed": 123},
        "cluster": {
            "simulate_racks": 2,
            "simulate_gpus_per_rack": 20,
            "routing_seed": 99,
            "gpu": {
                "hbm_capacity_gb": 0.001,
                "hbm_read_bw_gbps": 1000,
                "hbm_write_bw_gbps": 1000,
                "hbm_latency_ms": 0.001,
            },
            "eic": {
                "nodes_per_rack": 1,
                "capacity_per_node_gb": 0.01,
                "read_bw_gbps": 100,
                "write_bw_gbps": 100,
                "access_latency_ms": 0.005,
            },
            "network": {
                "intra_rack_latency_us": 3,
                "cross_rack_latency_us": 15,
                "remote_ssd_latency_us": 200,
            },
        },
    }


def test_cluster_router_sampling_is_seeded() -> None:
    choices = []
    for _ in range(3):
        cluster = build_cluster(_cluster_config())
        request_hashes = []
        for gpu in cluster.all_gpus:
            block_hash = f"gpu:{gpu.gpu_id}"
            request_hashes.append(block_hash)
            gpu.hbm.insert(
                KVBlock(
                    block_hash=block_hash,
                    size_bytes=16,
                    prefix_depth=0,
                    last_access_time=0.0,
                    access_count=1,
                )
            )

        choices.append(
            ClusterRouter(cluster).route(request_hashes, session_id="s").gpu_id
        )

    assert choices == [choices[0]] * len(choices)


def test_cluster_replay_metrics_are_reproducible() -> None:
    requests = TraceGenerator(
        num_sessions=16,
        turns_per_session=3,
        prompt_tokens_min=32,
        prompt_tokens_max=64,
        initial_context_tokens=64,
        qps=20,
        block_size_bytes=1024,
        num_system_prompts=4,
        seed=7,
    ).generate()

    signatures = []
    for _ in range(3):
        cluster = build_cluster(_cluster_config())
        metrics = ClusterReplayer(cluster).run(requests)
        signatures.append(
            (
                metrics.total_requests,
                metrics.total_hits,
                metrics.total_misses,
                round(metrics.hit_rate, 6),
                cluster.total_cross_gpu_eic_hits,
            )
        )

    assert signatures == [signatures[0]] * len(signatures)


def test_learned_policy_sampling_is_seeded() -> None:
    blocks = {f"b{i}": object() for i in range(100)}
    first = LearnedPolicy(seed=17)
    second = LearnedPolicy(seed=17)

    assert [first.evict_candidate(blocks) for _ in range(3)] == [
        second.evict_candidate(blocks) for _ in range(3)
    ]
