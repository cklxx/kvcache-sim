from sim.cache_manager import CacheManager
from sim.policies import LRUPolicy, SessionAwarePrefetch
from sim.router import Router, Worker
from sim.storage import StorageTier
from trace.generator import Request
from trace.replay import TraceReplayer


class CountingPrefetch(SessionAwarePrefetch):
    def __init__(self) -> None:
        super().__init__()
        self.records = 0

    def record_sequence(self, session_id, block_hashes):
        self.records += 1
        super().record_sequence(session_id, block_hashes)

    def candidates(self, block_hash, session_id):
        assert self.records <= 1
        return super().candidates(block_hash, session_id)


def test_session_prefetch_learns_online_not_from_future_trace() -> None:
    prefetch = CountingPrefetch()
    cache = CacheManager(
        tiers=[
            StorageTier("HBM", 1024, 1000, 1000, 0.001),
            StorageTier("DRAM", 1024, 100, 100, 0.1),
        ],
        eviction_policy=LRUPolicy(),
        prefetch_policy=prefetch,
    )
    router = Router([Worker(0, cache)])
    requests = [
        Request("s", 0, 0.0, ["hot"], block_size=16, prompt_tokens=16),
        Request("s", 1, 1.0, ["hot"], block_size=16, prompt_tokens=16),
    ]

    TraceReplayer(router).run(requests)

    assert prefetch.records == 2
