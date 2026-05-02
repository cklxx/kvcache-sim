from sim.cluster import build_cluster
from sim.diagnostics import summarize_cluster_health


def _small_cluster():
    return build_cluster(
        {
            "cluster": {
                "simulate_racks": 2,
                "simulate_gpus_per_rack": 2,
                "routing_seed": 1,
            }
        }
    )


def test_cluster_health_counts_active_gpus_and_racks() -> None:
    cluster = _small_cluster()
    cluster.all_gpus[0].process_request(["a"], 128, "s0", 0.0)
    cluster.all_gpus[2].process_request(["b"], 128, "s1", 1.0)

    summary = summarize_cluster_health(cluster)

    assert summary["total_racks"] == 2
    assert summary["total_gpus"] == 4
    assert summary["active_racks"] == 2
    assert summary["active_gpus"] == 2
    assert summary["request_distribution"]["per_gpu"] == {0: 1, 1: 0, 2: 1, 3: 0}
    assert summary["request_distribution"]["per_rack"] == {0: 1, 1: 1}
    assert summary["request_distribution"]["total_processed_requests"] == 2
    assert summary["utilization"]["hbm"]["max"] > 0.0
    assert summary["utilization"]["eic"]["max"] == 0.0


def test_cluster_health_warns_on_request_skew() -> None:
    cluster = _small_cluster()
    for i in range(8):
        cluster.all_gpus[0].process_request([f"hot:{i}"], 128, "hot", float(i))

    summary = summarize_cluster_health(cluster)

    assert summary["active_gpus"] == 1
    assert summary["active_racks"] == 1
    assert summary["warnings"]["request_skew"] is True
    assert summary["warnings"]["gpu_request_skew"] is True
    assert summary["warnings"]["gpu_hot_id"] == 0
    assert summary["warnings"]["gpu_hot_requests"] == 8
    assert summary["warnings"]["inactive_gpus"] == 3
