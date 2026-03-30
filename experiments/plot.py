"""
plot_results — convenience wrapper around sim.metrics.plot_comparison.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from sim.metrics import Metrics, plot_comparison


def plot_results(
    results: Dict[str, Metrics],
    tier_names: List[str],
    output_dir: str = "results",
    show: bool = False,
    filename: str = "policy_comparison.png",
) -> str:
    """
    Save comparison plots to *output_dir* and optionally display them.

    Returns the path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plot_comparison(results, tier_names, output_path=out_path, show=show)
    return out_path
