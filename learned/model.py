"""
LearnedModel — thin wrapper for online inference.

Wraps a trained sklearn/LightGBM model and exposes a simple
.predict_reuse_distance(block_hash, access_times, current_time) API
so that LearnedPolicy can call it without knowing the model type.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from learned.features import extract_features


class LearnedModel:
    """
    Wraps a fitted regressor (LightGBM or sklearn-compatible).

    The model predicts log(1 + reuse_distance), so we exponentiate
    the output to get the predicted reuse distance in seconds.
    """

    def __init__(self, raw_model) -> None:
        self._model = raw_model

    def predict_reuse_distance(
        self,
        block_hash: str,
        access_times: Dict[str, List[float]],
        current_time: float,
        prefix_depth: float = 0.0,
    ) -> float:
        """Return predicted reuse distance (seconds) for a block."""
        if self._model is None:
            return float("inf")
        feats = extract_features(block_hash, access_times, current_time, prefix_depth)
        X = np.array([feats], dtype=np.float32)
        log_dist = float(self._model.predict(X)[0])
        return math.expm1(max(log_dist, 0.0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw batch prediction (returns log-distance array)."""
        return self._model.predict(X)

    @classmethod
    def from_trainer(cls, trainer) -> Optional["LearnedModel"]:
        """Train and wrap a model from a ModelTrainer instance."""
        raw = trainer.train()
        if raw is None:
            return None
        return cls(raw)
