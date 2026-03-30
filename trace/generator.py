"""
Synthetic trace generator.

Produces realistic multi-turn conversation workloads without any external
dataset dependency.  Conversations follow these rules:

  • Each session has a fixed "topic context" — a shared prefix that all
    turns in that session will build on (simulating the system prompt +
    prior conversation history).
  • Turn-N's prefix = session_prefix + turn0_prefix + … + turn(N-1)_prefix.
  • Block hashes are derived by hashing the token sequence at each
    prefix boundary, so the same prefix is always the same hash.

Request fields
--------------
  session_id   : str
  turn_id      : int
  timestamp    : float  (seconds since epoch 0)
  block_hashes : List[str]   (one hash per cache block in the prefix)
  block_size   : int    (bytes, uniform across blocks)
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


def _token_hash(tokens: List[int]) -> str:
    """Deterministic hex digest of a token sequence."""
    import struct
    raw = struct.pack(f">{len(tokens)}I", *tokens)
    h = hashlib.md5(raw, usedforsecurity=False)
    return h.hexdigest()


class TraceGenerator:
    """
    Generates a list of :class:`Request` objects.

    Parameters
    ----------
    num_sessions          : number of distinct conversation sessions
    turns_per_session     : turns per session (each turn = one cache read)
    prompt_tokens_min/max : range of *new* tokens added each turn
    qps                   : average requests per simulated second
    block_size_bytes      : bytes per KV block (default 4 KiB)
    num_system_prompts    : number of shared system prompt templates
                            (sessions sharing the same prompt share prefix blocks,
                             creating the hot/cold working set separation needed
                             to differentiate eviction policies)
    seed                  : random seed for reproducibility
    """

    def __init__(
        self,
        num_sessions: int = 100,
        turns_per_session: int = 5,
        prompt_tokens_min: int = 64,
        prompt_tokens_max: int = 256,
        qps: float = 10.0,
        block_size_bytes: int = 4096,
        num_system_prompts: int = 8,
        seed: int = 42,
    ) -> None:
        self.num_sessions = num_sessions
        self.turns_per_session = turns_per_session
        self.prompt_tokens_min = prompt_tokens_min
        self.prompt_tokens_max = prompt_tokens_max
        self.qps = qps
        self.block_size = block_size_bytes
        self.num_system_prompts = num_system_prompts
        self.rng = random.Random(seed)
        # Tokens per block boundary (simulate a fixed block-token size)
        self._tokens_per_block = 32
        # Pre-build shared system prompts (deterministic, based on prompt index)
        sp_rng = random.Random(seed + 999)
        self._system_prompts: List[List[int]] = [
            [sp_rng.randint(0, 50000) for _ in range(128)]
            for _ in range(num_system_prompts)
        ]

    def _new_tokens(self, n: int) -> List[int]:
        return [self.rng.randint(0, 50000) for _ in range(n)]

    def _session_context_tokens(self, session_id: str, length: int) -> List[int]:
        """Deterministic per-session topic tokens (appended after shared system prompt)."""
        r = random.Random(session_id)
        return [r.randint(0, 50000) for _ in range(length)]

    def _prefix_blocks(self, token_sequence: List[int]) -> List[str]:
        """
        Split *token_sequence* into blocks of _tokens_per_block and return
        cumulative-prefix hashes (like vLLM's prefix-tree entries).
        """
        blocks = []
        for end in range(self._tokens_per_block, len(token_sequence) + 1, self._tokens_per_block):
            blocks.append(_token_hash(token_sequence[:end]))
        return blocks

    def generate(self) -> List[Request]:
        """Build and return a sorted-by-timestamp request list."""
        requests: List[Request] = []
        current_time = 0.0

        for s_idx in range(self.num_sessions):
            session_id = f"session_{s_idx:04d}"

            # Pick a shared system prompt (many sessions share the same one → hot blocks)
            sp_idx = s_idx % self.num_system_prompts
            shared_prefix = list(self._system_prompts[sp_idx])

            # Append session-specific context tokens
            topic_len = self.rng.randint(self.prompt_tokens_min // 4, self.prompt_tokens_max // 4)
            session_tokens = shared_prefix + self._session_context_tokens(session_id, topic_len)

            for turn_id in range(self.turns_per_session):
                # Add new turn tokens
                new_len = self.rng.randint(self.prompt_tokens_min, self.prompt_tokens_max)
                session_tokens = session_tokens + self._new_tokens(new_len)

                block_hashes = self._prefix_blocks(session_tokens)
                if not block_hashes:
                    continue

                # Poisson inter-arrival
                inter_arrival = self.rng.expovariate(self.qps)
                current_time += inter_arrival

                requests.append(
                    Request(
                        session_id=session_id,
                        turn_id=turn_id,
                        timestamp=round(current_time, 6),
                        block_hashes=block_hashes,
                        block_size=self.block_size,
                    )
                )

        # Sort by arrival time (sessions may interleave)
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
            qps=tc.get("qps", 10.0),
            block_size_bytes=cc.get("block_size_bytes", 4096),
            num_system_prompts=tc.get("num_system_prompts", 8),
            seed=tc.get("seed", 42),
        )
