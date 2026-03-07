"""
aggregator.py
─────────────────────────────────────────────────────────────────────────────
Aggregates classified conflicts into per-cycle statistics and prints a
formatted summary table to the console.
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Dict

import pandas as pd

from config import SiteConfig, TTC_THRESHOLDS, DRAC_THRESHOLDS


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def aggregate_results(
    conflict_df: pd.DataFrame,
    site: SiteConfig,
    n_cycles: int,
) -> Dict:
    """
    Compute per-cycle conflict frequencies and severity distributions.

    Parameters
    ----------
    conflict_df : pd.DataFrame
        Output of ``conflict_classifier.classify_conflicts``.
    site : SiteConfig
        Signal-timing configuration for the study site.
    n_cycles : int
        Number of complete red–amber–green cycles in the observation video.

    Returns
    -------
    dict
        Keys:
        ``site_name``, ``n_cycles``, ``cycle_length``,
        ``total_conflicts``, ``conflicts_per_cycle``,
        ``ttc_distribution``, ``drac_distribution``,
        ``by_vehicle_type``, ``conflict_df`` (critical rows only).
    """
    critical = conflict_df[conflict_df["is_critical"]].copy()
    n_safe   = max(n_cycles, 1)

    ttc_dist  = critical["TTC_severity"].value_counts().to_dict()
    drac_dist = critical["DRAC_severity"].value_counts().to_dict()

    # Per-follower-type breakdown
    by_type = (
        critical.groupby("follower_type")
        .size()
        .rename("conflict_count")
        .reset_index()
    )
    by_type["per_cycle"] = (by_type["conflict_count"] / n_safe).round(2)

    return {
        "site_name":           site.name,
        "n_cycles":            n_cycles,
        "cycle_length":        site.cycle_length,
        "total_conflicts":     int(len(critical)),
        "conflicts_per_cycle": round(len(critical) / n_safe, 2),
        "ttc_distribution":    ttc_dist,
        "drac_distribution":   drac_dist,
        "by_vehicle_type":     by_type,
        "conflict_df":         critical,
    }


def print_summary(results: Dict) -> None:
    """Print a formatted conflict-analysis summary to stdout."""
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  CONFLICT ANALYSIS SUMMARY")
    print(f"  Site         : {results['site_name']}")
    print(f"  Signal cycles: {results['n_cycles']}  "
          f"(cycle length = {results['cycle_length']:.0f} s)")
    print(f"  Total critical conflicts : {results['total_conflicts']}")
    print(f"  Per signal cycle         : {results['conflicts_per_cycle']:.2f}")

    print(f"\n  TTC-based severity distribution:")
    ttc_order = [f"TTC ≤ {t} s" for t in TTC_THRESHOLDS]
    for label in ttc_order:
        count = results["ttc_distribution"].get(label, 0)
        print(f"    {label:<20} : {count}")

    print(f"\n  DRAC-based severity distribution:")
    drac_order = [f"DRAC ≥ {d} m/s²" for d in DRAC_THRESHOLDS]
    for label in drac_order:
        count = results["drac_distribution"].get(label, 0)
        print(f"    {label:<25} : {count}")

    print(f"\n  Conflicts by follower vehicle type:")
    print(results["by_vehicle_type"].to_string(index=False))
    print(f"{sep}\n")
