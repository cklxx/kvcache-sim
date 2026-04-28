"""
ModelTrainer — collects training examples from a trace replay and
               trains a LightGBM regression model to predict reuse distance.

Usage
-----
  trainer = ModelTrainer()
  trainer.collect(requests)          # build training data from trace
  model   = trainer.train()          # returns a LearnedModel
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from trace.generator import Request


class ModelTrainer:
    def __init__(self, min_samples: int = 200) -> None:
        self.min_samples = min_samples
        self._events: List[Tuple[str, float, float, float]] = []
        self._access_times: Dict[str, List[float]] = defaultdict(list)

    def collect(self, requests: List[Request], reset: bool = True) -> None:
        """
        Scan the trace and build (block_hash, time, depth, reuse_dist) records.
        We compute reuse distance as the gap to the next occurrence of the same
        block hash within the supplied training slice. Callers should pass only
        warmup/train requests, not the measured evaluation suffix.
        """
        if reset:
            self._events = []
            self._access_times = defaultdict(list)

        # Build next_access map: for each (block_hash, occurrence_idx) → next_time
        occurrences: Dict[str, List[float]] = defaultdict(list)
        for req in requests:
            for depth, bh in enumerate(req.block_hashes):
                occurrences[bh].append(req.timestamp)

        # Now build training events
        seen: Dict[str, int] = defaultdict(int)  # block_hash → occurrence count so far
        access_so_far: Dict[str, List[float]] = defaultdict(list)
        if not reset:
            for bh, times in self._access_times.items():
                access_so_far[bh] = list(times)

        for req in requests:
            for depth, bh in enumerate(req.block_hashes):
                current_time = req.timestamp
                idx = seen[bh]
                all_times = occurrences[bh]

                if idx + 1 < len(all_times):
                    reuse_dist = all_times[idx + 1] - current_time
                else:
                    reuse_dist = 1e6   # never accessed again

                if access_so_far[bh]:   # need at least one prior access for features
                    self._events.append((bh, current_time, float(depth), reuse_dist))

                access_so_far[bh].append(current_time)
                seen[bh] += 1

        self._access_times = {k: list(v) for k, v in access_so_far.items()}

    @property
    def sample_count(self) -> int:
        return len(self._events)

    def train(self):
        """
        Train a LightGBM model. Falls back to a trivial sklearn model if
        LightGBM is not installed.

        Returns
        -------
        A fitted model with a .predict(X) method, or None if insufficient data.
        """
        from learned.features import build_feature_matrix

        if len(self._events) < self.min_samples:
            print(f"[train] Only {len(self._events)} samples — skipping training.")
            return None

        X, y = build_feature_matrix(self._events, self._access_times)
        print(f"[train] Training on {len(X)} samples, {X.shape[1]} features …")

        try:
            import lightgbm as lgb
            import pandas as pd
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
            params = {
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "verbosity": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_df, y)
            print("[train] LightGBM model trained.")
            return model
        except ImportError:
            pass

        # Fallback: gradient boosting from sklearn
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
            model.fit(X, y)
            print("[train] sklearn GradientBoosting model trained (LightGBM not found).")
            return model
        except ImportError:
            pass

        print("[train] Neither LightGBM nor sklearn available — no model trained.")
        return None
