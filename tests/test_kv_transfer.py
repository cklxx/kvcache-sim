from sim.kv_transfer import KVTransferModel, TransferConfig
from sim.network import NetworkModel


def test_transfer_config_default_uses_effective_100gbps_bandwidth() -> None:
    cfg = TransferConfig.from_config({})

    assert cfg.rdma_bw_gbps == 12.5


def test_same_node_nvlink_is_not_used_across_racks() -> None:
    model = KVTransferModel(
        TransferConfig(rdma_bw_gbps=12.5),
        NetworkModel(gpus_per_node=8, nvlink_bw_gbps=900),
    )

    same_rack_ms = model.transfer_latency_ms(
        num_blocks=1,
        block_size=1_000_000_000,
        same_rack=True,
        src_gpu=0,
        dst_gpu=4,
    )
    cross_rack_ms = model.transfer_latency_ms(
        num_blocks=1,
        block_size=1_000_000_000,
        same_rack=False,
        src_gpu=0,
        dst_gpu=4,
    )

    assert cross_rack_ms > same_rack_ms * 10
