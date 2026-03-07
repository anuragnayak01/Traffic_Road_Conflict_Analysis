"""
05_conflict_estimator.py
PURPOSE : Estimate critical rear-end conflicts per signal cycle.
          Expects ssm_with_cycles.csv (output of 05b_signal_detector.py)
          which already has correct signal_cycle from vehicle motion detection.

SOURCE  : IITB 4-step methodology (handwritten notes Page 3)
          TTC  < 1.5 s      → critical (TRB / Souza 2011)
          DRAC > 3.35 m/s²  → critical rear-end conflict
"""
import argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TTC_THRESHOLD_S     = 1.5
DRAC_THRESHOLD_MPS2 = 3.35
MIN_INTERACTIONS    = 10    # minimum per cycle to report


def estimate_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each (follower, leader) interaction → critical flag."""
    valid = df.dropna(subset=['TTC_s', 'DRAC_mps2']).copy()
    valid = valid[valid['TTC_s'] < 999.0]

    grouped = (
        valid
        .groupby(['signal_cycle', 'phase',
                  'follower_id', 'leader_id',
                  'follower_type', 'leader_type'])
        .agg(
            min_TTC_s              = ('TTC_s',       'min'),
            max_DRAC_mps2          = ('DRAC_mps2',   'max'),
            mean_TTC_s             = ('TTC_s',       'mean'),
            interaction_duration_s = ('timestamp_s',
                                      lambda x: round(x.max()-x.min(), 3)),
            n_timesteps            = ('timestamp_s', 'count'),
        )
        .reset_index()
    )

    grouped['critical_TTC']  = grouped['min_TTC_s']     < TTC_THRESHOLD_S
    grouped['critical_DRAC'] = grouped['max_DRAC_mps2'] > DRAC_THRESHOLD_MPS2
    grouped['is_critical']   = grouped['critical_TTC'] | grouped['critical_DRAC']
    return grouped


def summarise_by_cycle(conflicts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        conflicts
        .groupby(['signal_cycle', 'phase'])
        .agg(
            total_interactions  = ('is_critical',  'count'),
            critical_conflicts  = ('is_critical',  'sum'),
            critical_TTC_only   = ('critical_TTC', 'sum'),
            critical_DRAC_only  = ('critical_DRAC','sum'),
            mean_min_TTC_s      = ('min_TTC_s',    'mean'),
            mean_max_DRAC_mps2  = ('max_DRAC_mps2','mean'),
        )
        .reset_index()
    )
    summary['conflict_rate_pct'] = (
        100.0 * summary['critical_conflicts']
              / summary['total_interactions']
    ).round(2)

    # Drop cycles too sparse to be statistically meaningful
    before  = len(summary)
    summary = summary[
        summary['total_interactions'] >= MIN_INTERACTIONS
    ].reset_index(drop=True)
    if len(summary) < before:
        logger.info(f"Dropped {before - len(summary)} sparse cycles "
                    f"(< {MIN_INTERACTIONS} interactions)")

    return summary


def process(input_csv: str, output_csv: str):
    logger.info(f"Reading SSM with detected cycles: {input_csv}")
    df = pd.read_csv(input_csv)

    # Verify signal_cycle column exists — must come from 05b
    if 'signal_cycle' not in df.columns:
        raise ValueError(
            "signal_cycle column not found. "
            "Run 05b_signal_detector.py before this step."
        )

    if 'phase' not in df.columns:
        df['phase'] = 'UNKNOWN'

    logger.info(f"Cycles present : {sorted(df['signal_cycle'].unique())}")
    logger.info(f"Phases present : {sorted(df['phase'].unique())}")

    conflicts = estimate_conflicts(df)
    summary   = summarise_by_cycle(conflicts)

    conflicts.to_csv(output_csv, index=False)
    sum_path = output_csv.replace('.csv', '_summary.csv')
    summary.to_csv(sum_path, index=False)

    logger.info(f"\nConflicts saved → {output_csv}")
    logger.info(f"Summary   saved → {sum_path}")
    logger.info("\n" + summary.to_string(index=False))

    total_critical = int(conflicts['is_critical'].sum())
    valid_cycles   = int(summary['signal_cycle'].nunique())
    logger.info(f"\nTotal critical conflicts : {total_critical}")
    logger.info(f"Valid signal cycles      : {valid_cycles}")
    if valid_cycles > 0:
        logger.info(
            f"Avg per cycle            : "
            f"{total_critical / valid_cycles:.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  default='results_v2/ssm_with_cycles.csv',
                    help='SSM results WITH signal_cycle from 05b_signal_detector.py')
    ap.add_argument('--output', default='results_v2/conflicts.csv')
    args = ap.parse_args()
    process(args.input, args.output)


if __name__ == '__main__':
    main()