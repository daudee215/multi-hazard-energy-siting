"""Assign scenario ranks and keep the upper end of the rank range separate."""

import pandas as pd


def rank_sites(sites, scenarios):
    """Update the supplied site table and return aggregate sensitivity results."""
    for name in scenarios:
        sites[f"rank_{name}"] = sites[f"composite_{name}"].rank(method="min")

    ranks = sites[[f"rank_{name}" for name in scenarios]]
    sites["rank_min"] = ranks.min(axis=1)
    sites["rank_upper_bound"] = ranks.max(axis=1)
    sites["rank_range"] = sites["rank_upper_bound"] - sites["rank_min"]
    sites["rank_instability_flag"] = sites["rank_range"] > len(sites) * 0.1

    baseline = ranks["rank_equal"]
    results = []
    for name in scenarios:
        scenario = ranks[f"rank_{name}"]
        shift = (scenario - baseline).abs()
        results.append({
            "scenario": name,
            "n_sites": len(sites),
            "spearman_vs_equal": baseline.corr(scenario, method="spearman"),
            "max_rank_shift": shift.max(),
            "mean_rank_shift": shift.mean(),
        })
    return pd.DataFrame(results)
