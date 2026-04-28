"""
Feature engineering for the learned eviction policy.

Each training example represents a (block, decision_point) pair.
The label is the *reuse distance* (time until next access, or ∞ for
never-accessed-again blocks).

Feature vector (8 dims)
-----------------------
  0  log(1 + recency)         — time since last access
  1  log(1 + access_count)    — total access count so far
  2  log(1 + mean_interval)   — mean inter-access interval
  3  log(1 + std_interval)    — standard deviation of intervals
  4  log(1 + min_interval)    — minimum inter-access interval
  5  log(1 + max_interval)    — maximum inter-access interval
  6  access_count / horizon   — normalised frequency
  7  session_turn_depth       — prefix depth encoded as float
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURE_DIM = 8


def extract_features(
    block_hash: str,
    access_times: Dict[str, List[float]],
    current_time: float,
    prefix_depth: float = 0.0,
    horizon: float = 1000.0,
) -> List[float]:
    """
    Build the 8-dim feature vector for one block at *current_time*.
    """
    # Only use history up to current_time (build_feature_matrix passes the full
    # accumulated history, so future accesses must be excluded)
    hist = [t for t in access_times.get(block_hash, []) if t <= current_time]
    if not hist:
        return [0.0] * FEATURE_DIM

    recency = current_time - hist[-1]
    count = float(len(hist))

    if len(hist) >= 2:
        intervals = [hist[i + 1] - hist[i] for i in range(len(hist) - 1)]
        mean_iv = float(np.mean(intervals))
        std_iv = float(np.std(intervals))
        min_iv = float(min(intervals))
        max_iv = float(max(intervals))
    else:
        mean_iv = std_iv = min_iv = max_iv = 0.0

    return [
        math.log1p(recency),
        math.log1p(count),
        math.log1p(mean_iv),
        math.log1p(std_iv),
        math.log1p(min_iv),
        math.log1p(max_iv),
        count / max(horizon, 1.0),
        float(prefix_depth),
    ]


def build_feature_matrix(
    events: List[Tuple[str, float, float, float]],
    access_times: Dict[str, List[float]],
    horizon: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for training.

    Parameters
    ----------
    events : list of (block_hash, current_time, prefix_depth, reuse_distance)
    access_times : accumulated access history up to each event's time

    Returns
    -------
    X : (N, 8) float32
    y : (N,) float32  — log(1 + reuse_distance)
    """
    X_rows = []
    y_vals = []

    for block_hash, current_time, depth, reuse_dist in events:
        feats = extract_features(block_hash, access_times, current_time, depth, horizon)
        X_rows.append(feats)
        y_vals.append(math.log1p(reuse_dist))

    if X_rows:
        X = np.array(X_rows, dtype=np.float32)
    else:
        X = np.empty((0, FEATURE_DIM), dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y
