"""
Synthetic trace generator.

Produces realistic multi-turn conversation workloads without any external
dataset dependency.  Three workload types:

  1. **Standard**: Multi-turn chat with shared system prompts
  2. **RAG**: Zipf-distributed document retrieval, multi-chunk per query
  3. **Agent**: Long-context tool-use sessions (20-50+ turns, growing context)

Block hashes use incremental chaining:
  hash_k = MD5(hash_{k-1} || tokens[k*B : (k+1)*B])
This preserves prefix sharing: sessions that share token prefixes
produce identical hashes for those blocks.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Request:
    """Simulator request. ``timestamp`` is milliseconds from trace start."""

    session_id: str
    turn_id: int
    timestamp: float
    block_hashes: List[str]
    block_size: int = 4096
    prompt_tokens: int = 0
    output_tokens: int = 0


# ── Block hashing ────────────────────────────────────────────────────


def _incremental_block_hashes(
    tokens: List[int], tokens_per_block: int = 16
) -> List[str]:
    """
    Build block hashes using incremental chaining — O(N) total.
    Mimics vLLM's radix-tree hashing.
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


# ── Zipf distribution helper ────────────────────────────────────────


def _zipf_sample(rng: random.Random, n: int, alpha: float = 1.1) -> int:
    """Sample from Zipf distribution over [0, n). alpha > 1."""
    weights = [1.0 / (i + 1) ** alpha for i in range(n)]
    return rng.choices(range(n), weights=weights, k=1)[0]


# ── Trace Generator ──────────────────────────────────────────────────


class TraceGenerator:
    """
    Generates a list of :class:`Request` objects.

    Parameters
    ----------
    num_sessions            : conversation sessions
    turns_per_session       : turns per session
    prompt_tokens_min/max   : new tokens added per turn
    initial_context_tokens  : fixed context prepended to each session
    num_system_prompts      : shared prompt templates (hot prefix blocks)
    num_shared_docs         : shared document pool size (0 = per-session context)
    num_rag_chunks          : chunks retrieved per query from doc pool
    doc_zipf_alpha          : Zipf exponent for doc popularity (1.1 = moderate skew)
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
        num_rag_chunks: int = 3,
        doc_zipf_alpha: float = 1.1,
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
        self.num_rag_chunks = num_rag_chunks
        self.doc_zipf_alpha = doc_zipf_alpha
        self.rng = random.Random(seed)
        self._tokens_per_block = 16  # vLLM default page size

        # Pre-build shared system prompts (128 tokens each)
        sp_rng = random.Random(seed + 999)
        self._system_prompts: List[List[int]] = [
            [sp_rng.randint(0, 50000) for _ in range(128)]
            for _ in range(num_system_prompts)
        ]

        # Shared document pool with varying doc lengths
        # Models a RAG knowledge base: hot docs queried by many sessions
        self._shared_docs: List[List[int]] = []
        if num_shared_docs > 0:
            doc_rng = random.Random(seed + 777)
            chunk_size = max(16, initial_context_tokens // max(1, num_rag_chunks))
            for _ in range(num_shared_docs):
                self._shared_docs.append(
                    [doc_rng.randint(0, 50000) for _ in range(chunk_size)]
                )

    def _new_tokens(self, n: int) -> List[int]:
        return [self.rng.randint(0, 50000) for _ in range(n)]

    def _session_context_tokens(self, session_id: str, length: int) -> List[int]:
        """Deterministic per-session context (document, RAG chunks, etc.)."""
        r = random.Random(session_id)
        return [r.randint(0, 50000) for _ in range(length)]

    def _pick_rag_chunks(self) -> List[int]:
        """
        Pick num_rag_chunks docs via Zipf distribution and concatenate.

        Zipf models real RAG: a few hot documents (FAQ, popular APIs)
        are retrieved far more often than long-tail content.
        """
        tokens: List[int] = []
        picked = set()
        for _ in range(self.num_rag_chunks):
            doc_idx = _zipf_sample(self.rng, self.num_shared_docs, self.doc_zipf_alpha)
            if doc_idx not in picked:
                picked.add(doc_idx)
                tokens += self._shared_docs[doc_idx]
        return tokens

    def generate(self) -> List[Request]:
        """Build and return a sorted-by-timestamp request list."""
        requests: List[Request] = []
        current_time = 0.0

        for s_idx in range(self.num_sessions):
            session_id = f"session_{s_idx:04d}"

            # 1) Shared system prompt
            sp_idx = s_idx % self.num_system_prompts
            session_tokens = list(self._system_prompts[sp_idx])

            # 2) Initial context: RAG chunks (Zipf) or per-session
            if self._shared_docs:
                session_tokens += self._pick_rag_chunks()
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

                # 4) Incremental block hashing
                block_hashes = _incremental_block_hashes(
                    session_tokens, self._tokens_per_block
                )
                if not block_hashes:
                    continue

                inter_arrival = self.rng.expovariate(self.qps) * 1000.0
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
            num_rag_chunks=tc.get("num_rag_chunks", 3),
            doc_zipf_alpha=tc.get("doc_zipf_alpha", 1.1),
            seed=tc.get("seed", 42),
        )


# ======================================================================
# Agent Trace Generator
# ======================================================================


class AgentTraceGenerator:
    """
    Generates long-context agent workloads.

    Agent sessions model tool-use workflows:
      [system_prompt | tool_defs | turn_0_query | tool_call_0 | tool_result_0
       | turn_1_reasoning | tool_call_1 | tool_result_1 | ... | final_answer]

    Key characteristics:
      - Many turns (20-50+) per session
      - Context grows monotonically (no truncation by default)
      - Tool definitions are shared across sessions (hot prefix)
      - Tool results vary in size (small API calls vs large code/doc)
      - Earlier turns become less relevant but retain prefix dependency

    Parameters
    ----------
    num_sessions          : number of agent sessions
    turns_per_session     : range (min, max) of turns
    tool_def_tokens       : shared tool definitions (all sessions)
    tool_result_min/max   : tokens per tool result (varies by tool type)
    reasoning_tokens      : tokens for agent reasoning per turn
    num_tool_sets         : number of shared tool definition sets
    qps                   : arrival rate
    """

    def __init__(
        self,
        num_sessions: int = 500,
        turns_min: int = 15,
        turns_max: int = 50,
        tool_def_tokens: int = 2048,
        tool_result_min: int = 128,
        tool_result_max: int = 4096,
        reasoning_tokens: int = 256,
        num_tool_sets: int = 5,
        qps: float = 20.0,
        block_size_bytes: int = 2097152,
        num_system_prompts: int = 5,
        seed: int = 42,
    ) -> None:
        self.num_sessions = num_sessions
        self.turns_min = turns_min
        self.turns_max = turns_max
        self.tool_result_min = tool_result_min
        self.tool_result_max = tool_result_max
        self.reasoning_tokens = reasoning_tokens
        self.num_tool_sets = num_tool_sets
        self.qps = qps
        self.block_size = block_size_bytes
        self.num_system_prompts = num_system_prompts
        self.rng = random.Random(seed)
        self._tokens_per_block = 16

        # Shared system prompts
        sp_rng = random.Random(seed + 999)
        self._system_prompts = [
            [sp_rng.randint(0, 50000) for _ in range(256)]
            for _ in range(num_system_prompts)
        ]

        # Shared tool definition sets (hot prefix, shared across agent sessions)
        td_rng = random.Random(seed + 555)
        self._tool_defs = [
            [td_rng.randint(0, 50000) for _ in range(tool_def_tokens)]
            for _ in range(num_tool_sets)
        ]

    def _new_tokens(self, n: int) -> List[int]:
        return [self.rng.randint(0, 50000) for _ in range(n)]

    def generate(self) -> List[Request]:
        """Generate agent workload with long, growing context."""
        requests: List[Request] = []
        current_time = 0.0

        for s_idx in range(self.num_sessions):
            session_id = f"agent_{s_idx:04d}"
            num_turns = self.rng.randint(self.turns_min, self.turns_max)

            # 1) System prompt + tool definitions (shared hot prefix)
            sp_idx = s_idx % self.num_system_prompts
            tool_idx = s_idx % self.num_tool_sets
            session_tokens = (
                list(self._system_prompts[sp_idx])
                + list(self._tool_defs[tool_idx])
            )

            # 2) Initial user query
            query_len = self.rng.randint(128, 1024)
            session_tokens += self._new_tokens(query_len)

            for turn_id in range(num_turns):
                # Agent reasoning (thinking tokens)
                session_tokens += self._new_tokens(self.reasoning_tokens)

                # Tool call + tool result (variable size)
                # Some tools return small results (API), some large (code search)
                tool_result_len = self.rng.randint(
                    self.tool_result_min, self.tool_result_max
                )
                session_tokens += self._new_tokens(tool_result_len)

                # Block hashing
                block_hashes = _incremental_block_hashes(
                    session_tokens, self._tokens_per_block
                )
                if not block_hashes:
                    continue

                inter_arrival = self.rng.expovariate(self.qps) * 1000.0
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
    def from_config(cfg: dict) -> "AgentTraceGenerator":
        ac = cfg.get("agent_trace", {})
        return AgentTraceGenerator(
            num_sessions=ac.get("num_sessions", 500),
            turns_min=ac.get("turns_min", 15),
            turns_max=ac.get("turns_max", 50),
            tool_def_tokens=ac.get("tool_def_tokens", 2048),
            tool_result_min=ac.get("tool_result_min", 128),
            tool_result_max=ac.get("tool_result_max", 4096),
            reasoning_tokens=ac.get("reasoning_tokens", 256),
            num_tool_sets=ac.get("num_tool_sets", 5),
            qps=ac.get("qps", 20.0),
            block_size_bytes=cfg.get("cache", {}).get("block_size_bytes", 2097152),
            num_system_prompts=ac.get("num_system_prompts", 5),
            seed=ac.get("seed", 42),
        )
