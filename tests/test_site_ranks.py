"""Regression coverage for scenario ranks and tied upper bounds."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from site_ranks import rank_sites


def test_max_overlay_rank_remains_distinct_from_upper_bound():
    sites = pd.DataFrame({
        "composite_equal": [0.1, 0.1, 0.4, 0.8],
        "composite_max": [0.6, 0.4, 0.4, 0.3],
    })
    sensitivity = rank_sites(sites, ["equal", "max"]).set_index("scenario")
    assert sites["rank_equal"].tolist() == [1, 1, 3, 4]
    assert sites["rank_max"].tolist() == [4, 2, 2, 1]
    assert sites["rank_upper_bound"].tolist() == [4, 2, 3, 4]
    assert sites["rank_range"].tolist() == [3, 1, 1, 3]
    assert sensitivity.loc["max", "max_rank_shift"] == 3
    assert sensitivity.loc["max", "mean_rank_shift"] == 2
