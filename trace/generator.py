"""
Synthetic trace generator.

Produces realistic multi-turn conversation workloads without any external
dataset dependency.  Conversations follow these rules:

  - Each session starts with a shared system prompt (creating hot common blocks)
  - Sessions also have an *initial context* of configurable length (simulating
    long documents, few-shot examples, or RAG context).  This is the main
    knob for controlling per-request cache footprint.
  - Subsequent turns append new tokens to the cumulative prefix.
  - Block hashes use incremental chaining (O(N) total, O(1) per block):
      hash_k = MD5(hash_{k-1} || tokens[k*B : (k+1)*B])
    This preserves prefix sharing: two sessions that share the first K
    blocks will produce identical hashes for those blocks.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Request:
    session_id: str
    turn_id: int
    timestamp: float
    block_hashes: List[str]
    block_size: int = 4096
    prompt_tokens: int = 0


# ── Block hashing ────────────────────────────────────────────────────


def _incremental_block_hashes(
    tokens: List[int], tokens_per_block: int = 16
) -> List[str]:
    """
    Build block hashes using incremental chaining — O(N) total.

    hash_k = MD5( hash_{k-1} + "|" + chunk_k )

    where chunk_k = str(tokens[k*B : (k+1)*B]).

    This mimics vLLM's radix-tree hashing: each node's identity depends
    on its parent hash + its own token chunk, so prefix sharing is
    preserved across sessions that share token prefixes.
    """
    hashes: List[str] = []
    prev = ""
    n = len(tokens)
    for start in range(0, n - tokens_per_block + 1, tokens_per_block):
        end = start + tokens_per_block
        chunk = tokens[start:end]
        payload = f"{prev}|{chunk}"
        h = hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()
        hashes.append(h)
        prev = h
    return hashes


# ── Trace Generator ──────────────────────────────────────────────────


class TraceGenerator:
    """
    Generates a list of :class:`Request` objects.

    Parameters
    ----------
    num_sessions            : conversation sessions
    turns_per_session       : turns per session
    prompt_tokens_min/max   : new tokens added per turn
    initial_context_tokens  : fixed document/RAG context prepended to each
                              session (controls per-request cache footprint;
                              set to 0 for short Q&A, 32768 for long-doc RAG)
    num_system_prompts      : shared prompt templates (hot prefix blocks)
    qps                     : average requests / simulated second
    block_size_bytes        : bytes per KV block
    seed                    : reproducibility
    """

    def __init__(
        self,
        num_sessions: int = 100,
        turns_per_session: int = 5,
        prompt_tokens_min: int = 64,
        prompt_tokens_max: int = 256,
        initial_context_tokens: int = 128,
        qps: float = 10.0,
        block_size_bytes: int = 4096,
        num_system_prompts: int = 8,
        num_shared_docs: int = 0,
        seed: int = 42,
    ) -> None:
        self.num_sessions = num_sessions
        self.turns_per_session = turns_per_session
        self.prompt_tokens_min = prompt_tokens_min
        self.prompt_tokens_max = prompt_tokens_max
        self.initial_context_tokens = initial_context_tokens
        self.qps = qps
        self.block_size = block_size_bytes
        self.num_system_prompts = num_system_prompts
        self.num_shared_docs = num_shared_docs
        self.rng = random.Random(seed)
        self._tokens_per_block = 16  # vLLM default page size
        # Pre-build shared system prompts (128 tokens each, deterministic)
        sp_rng = random.Random(seed + 999)
        self._system_prompts: List[List[int]] = [
            [sp_rng.randint(0, 50000) for _ in range(128)]
            for _ in range(num_system_prompts)
        ]
        # Shared document pool: multiple sessions reference the same docs
        # Models RAG over a common knowledge base or shared code repos
        self._shared_docs: List[List[int]] = []
        if num_shared_docs > 0:
            doc_rng = random.Random(seed + 777)
            for _ in range(num_shared_docs):
                doc_len = max(16, initial_context_tokens)
                self._shared_docs.append(
                    [doc_rng.randint(0, 50000) for _ in range(doc_len)]
                )

    def _new_tokens(self, n: int) -> List[int]:
        return [self.rng.randint(0, 50000) for _ in range(n)]

    def _session_context_tokens(self, session_id: str, length: int) -> List[int]:
        """Deterministic per-session context (document, RAG chunks, etc.)."""
        r = random.Random(session_id)
        return [r.randint(0, 50000) for _ in range(length)]

    def generate(self) -> List[Request]:
        """Build and return a sorted-by-timestamp request list."""
        requests: List[Request] = []
        current_time = 0.0

        for s_idx in range(self.num_sessions):
            session_id = f"session_{s_idx:04d}"

            # 1) Shared system prompt (hot blocks, shared across sessions)
            sp_idx = s_idx % self.num_system_prompts
            session_tokens = list(self._system_prompts[sp_idx])

            # 2) Initial context: shared doc (if pool exists) or per-session
            if self._shared_docs:
                # Pick a shared document — multiple sessions hit the same doc
                doc_idx = s_idx % self.num_shared_docs
                session_tokens += list(self._shared_docs[doc_idx])
            elif self.initial_context_tokens > 0:
                session_tokens += self._session_context_tokens(
                    session_id, self.initial_context_tokens
                )

            for turn_id in range(self.turns_per_session):
                # 3) New turn tokens
                new_len = self.rng.randint(
                    self.prompt_tokens_min, self.prompt_tokens_max
                )
                session_tokens = session_tokens + self._new_tokens(new_len)

                # 4) Incremental block hashing (O(N) total, preserves prefix sharing)
                block_hashes = _incremental_block_hashes(
                    session_tokens, self._tokens_per_block
                )
                if not block_hashes:
                    continue

                inter_arrival = self.rng.expovariate(self.qps)
                current_time += inter_arrival

                requests.append(
                    Request(
                        session_id=session_id,
                        turn_id=turn_id,
                        timestamp=round(current_time, 6),
                        block_hashes=block_hashes,
                        block_size=self.block_size,
                        prompt_tokens=len(session_tokens),
                    )
                )

        requests.sort(key=lambda r: r.timestamp)
        return requests

    @staticmethod
    def from_config(cfg: dict) -> "TraceGenerator":
        tc = cfg.get("trace", {})
        cc = cfg.get("cache", {})
        return TraceGenerator(
            num_sessions=tc.get("num_sessions", 100),
            turns_per_session=tc.get("turns_per_session", 5),
            prompt_tokens_min=tc.get("prompt_tokens_min", 64),
            prompt_tokens_max=tc.get("prompt_tokens_max", 256),
            initial_context_tokens=tc.get("initial_context_tokens", 128),
            qps=tc.get("qps", 10.0),
            block_size_bytes=cc.get("block_size_bytes", 4096),
            num_system_prompts=tc.get("num_system_prompts", 8),
            num_shared_docs=tc.get("num_shared_docs", 0),
            seed=tc.get("seed", 42),
        )
