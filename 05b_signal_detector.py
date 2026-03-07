"""
05b_signal_detector.py
PURPOSE : Detect ACTUAL signal cycle boundaries from vehicle motion patterns.
          Uses mean speed across all vehicles per 0.2s time-step as a proxy
          for signal phase — low mean speed = RED, high mean speed = GREEN.

WHY THIS IS CORRECT:
    Arbitrary time-division (timestamp // 90) is wrong because:
        - It assumes the video starts exactly at a cycle boundary
        - It ignores actual vehicle behavior
    This approach reads the traffic itself to find where each cycle starts.

ALGORITHM:
    1. Compute mean speed of ALL vehicles at each 0.2s time-step
    2. Apply rolling smoothing to remove per-frame noise
    3. Classify each time-step: mean_speed < threshold → RED, else → GREEN
    4. Detect RED→GREEN transitions = green phase start = cycle boundary
    5. Filter out spurious transitions shorter than MIN_GREEN_DURATION
    6. If fewer than 2 green phases → single cycle (video too short)
    7. Assign cycle number to every SSM row by timestamp merge

OUTPUT:
    signal_phases.csv      — per-timestep: phase, mean_speed, cycle number
    ssm_with_cycles.csv    — ssm_results.csv with correct signal_cycle column
    cycle_boundaries.csv   — summary of each detected cycle

USAGE:
    python 05b_signal_detector.py \
        --smoothed       results_v2/smoothed.csv \
        --ssm            results_v2/ssm_results.csv \
        --output-phases  results_v2/signal_phases.csv \
        --output-ssm     results_v2/ssm_with_cycles.csv \
        --output-boundaries results_v2/cycle_boundaries.csv \
        --stopped-speed  5.0 \
        --min-cycle      30.0
"""

import argparse
import logging
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants  (can be overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_STOPPED_SPEED_KMH  = 5.0    # below this = RED phase
DEFAULT_MIN_CYCLE_DURATION = 30.0   # minimum seconds for a valid cycle
DEFAULT_MIN_GREEN_DURATION = 5.0    # minimum seconds a green phase must last
DEFAULT_SMOOTH_WINDOW      = 7      # rolling window size (× 0.2s = 1.4s)
DEFAULT_MIN_VEHICLES       = 3      # ignore time-steps with fewer vehicles


# ---------------------------------------------------------------------------
# Step 1 — Build mean-speed time series
# ---------------------------------------------------------------------------
def build_speed_timeseries(smoothed_csv: str,
                            min_vehicles: int = DEFAULT_MIN_VEHICLES
                            ) -> pd.DataFrame:
    """
    Aggregate per-vehicle smoothed speeds into one mean-speed per time-step.
    Time-steps with fewer than min_vehicles are dropped (unreliable).
    """
    logger.info(f"Reading smoothed trajectories: {smoothed_csv}")
    df = pd.read_csv(smoothed_csv)

    required = {'timestamp_s', 'vehicle_id', 'speed_kmh'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"smoothed.csv missing columns: {missing}")

    ts_stats = (
        df.groupby('timestamp_s')['speed_kmh']
        .agg(
            mean_speed_kmh = 'mean',
            median_speed_kmh = 'median',
            n_vehicles     = 'count',
            std_speed_kmh  = 'std',
        )
        .reset_index()
        .sort_values('timestamp_s')
        .reset_index(drop=True)
    )

    # Drop time-steps with too few vehicles — unreliable mean
    before = len(ts_stats)
    ts_stats = ts_stats[ts_stats['n_vehicles'] >= min_vehicles].copy()
    dropped  = before - len(ts_stats)
    if dropped:
        logger.info(f"Dropped {dropped} time-steps with < {min_vehicles} vehicles")

    logger.info(
        f"Time-steps: {len(ts_stats)}  "
        f"Speed range: {ts_stats['mean_speed_kmh'].min():.1f}–"
        f"{ts_stats['mean_speed_kmh'].max():.1f} km/h  "
        f"Mean: {ts_stats['mean_speed_kmh'].mean():.1f} km/h"
    )
    return ts_stats


# ---------------------------------------------------------------------------
# Step 2 — Smooth the speed time series
# ---------------------------------------------------------------------------
def smooth_speed_series(ts_stats: pd.DataFrame,
                         window: int = DEFAULT_SMOOTH_WINDOW
                         ) -> pd.DataFrame:
    """
    Apply uniform (box) filter to mean_speed_kmh.
    uniform_filter1d handles edge effects better than pandas rolling.
    """
    ts_stats = ts_stats.copy()
    ts_stats['smooth_speed_kmh'] = uniform_filter1d(
        ts_stats['mean_speed_kmh'].values,
        size=window,
        mode='nearest'
    )
    return ts_stats


# ---------------------------------------------------------------------------
# Step 3 — Classify phases and detect transitions
# ---------------------------------------------------------------------------
def classify_phases(ts_stats: pd.DataFrame,
                     stopped_speed_kmh: float = DEFAULT_STOPPED_SPEED_KMH
                     ) -> pd.DataFrame:
    """
    Classify each time-step as RED or GREEN based on smooth_speed_kmh.
    Then detect transitions to find cycle boundaries.
    """
    ts_stats = ts_stats.copy()

    ts_stats['phase'] = np.where(
        ts_stats['smooth_speed_kmh'] < stopped_speed_kmh,
        'RED', 'GREEN'
    )

    # Encode phase as integer for diff-based transition detection
    ts_stats['phase_int']    = (ts_stats['phase'] == 'GREEN').astype(int)
    ts_stats['phase_change'] = ts_stats['phase_int'].diff().fillna(0)

    # GREEN start (RED→GREEN): phase_change == +1
    # RED   start (GREEN→RED): phase_change == -1
    ts_stats['is_green_start'] = ts_stats['phase_change'] == 1
    ts_stats['is_red_start']   = ts_stats['phase_change'] == -1

    n_green = ts_stats['is_green_start'].sum()
    n_red   = ts_stats['is_red_start'].sum()
    logger.info(f"Phase transitions detected: "
                f"{n_green} GREEN starts, {n_red} RED starts")

    return ts_stats


# ---------------------------------------------------------------------------
# Step 4 — Filter spurious transitions and build cycle boundaries
# ---------------------------------------------------------------------------
def build_cycle_boundaries(ts_stats: pd.DataFrame,
                            min_cycle_duration: float = DEFAULT_MIN_CYCLE_DURATION,
                            min_green_duration: float = DEFAULT_MIN_GREEN_DURATION
                            ) -> tuple:
    """
    From the detected green starts, filter out spurious short transitions
    and return valid cycle boundary timestamps.

    Returns:
        (green_starts: np.ndarray of valid cycle start times,
         single_cycle: bool,
         reason: str)
    """
    raw_green_starts = ts_stats.loc[
        ts_stats['is_green_start'], 'timestamp_s'
    ].values

    if len(raw_green_starts) == 0:
        # No transitions at all — probably all green (short video, free flow)
        # or all red (very slow traffic)
        dominant = ts_stats['phase'].mode()[0]
        reason = (
            f"No phase transitions found. "
            f"Dominant phase: {dominant}. Treating as single cycle."
        )
        return np.array([]), True, reason

    # Filter: remove green starts that are too close together
    # (spurious flicker caused by momentary speed changes)
    filtered = [raw_green_starts[0]]
    for gs in raw_green_starts[1:]:
        if (gs - filtered[-1]) >= min_green_duration:
            filtered.append(gs)
        else:
            logger.debug(f"  Filtered spurious green start at t={gs:.2f}s "
                         f"(too close to previous at {filtered[-1]:.2f}s)")

    filtered = np.array(filtered)

    # Filter: remove cycles that are too short
    valid = []
    for i, gs in enumerate(filtered):
        next_gs = filtered[i+1] if i+1 < len(filtered) else ts_stats['timestamp_s'].max()
        duration = next_gs - gs
        if duration >= min_cycle_duration:
            valid.append(gs)
        else:
            logger.debug(f"  Filtered short cycle at t={gs:.2f}s "
                         f"(duration {duration:.1f}s < {min_cycle_duration}s)")

    valid = np.array(valid)

    if len(valid) < 2:
        reason = (
            f"Only {len(valid)} valid green phase(s) found after filtering. "
            f"Treating as single cycle."
        )
        return valid, True, reason

    reason = (
        f"{len(valid)} valid signal cycles detected "
        f"from vehicle motion data."
    )
    return valid, False, reason


# ---------------------------------------------------------------------------
# Step 5 — Assign cycle numbers to time-steps
# ---------------------------------------------------------------------------
def assign_cycles_to_timeseries(ts_stats: pd.DataFrame,
                                 green_starts: np.ndarray,
                                 single_cycle: bool
                                 ) -> pd.DataFrame:
    """
    Add signal_cycle column to the time-series DataFrame.
    """
    ts_stats = ts_stats.copy()

    if single_cycle or len(green_starts) == 0:
        ts_stats['signal_cycle'] = 1
        return ts_stats

    # np.searchsorted: for each timestamp, find which cycle interval it falls in
    cycle_nums = np.searchsorted(green_starts,
                                  ts_stats['timestamp_s'].values,
                                  side='right')
    # Clamp minimum to 1
    ts_stats['signal_cycle'] = np.maximum(cycle_nums, 1)
    return ts_stats


# ---------------------------------------------------------------------------
# Step 6 — Build human-readable cycle boundary summary
# ---------------------------------------------------------------------------
def build_boundary_table(ts_stats: pd.DataFrame,
                          green_starts: np.ndarray,
                          single_cycle: bool
                          ) -> pd.DataFrame:
    """Build a summary DataFrame of each cycle's time window."""
    max_ts = ts_stats['timestamp_s'].max()
    min_ts = ts_stats['timestamp_s'].min()

    if single_cycle or len(green_starts) == 0:
        return pd.DataFrame([{
            'cycle':           1,
            'start_s':         round(min_ts, 2),
            'end_s':           round(max_ts, 2),
            'duration_s':      round(max_ts - min_ts, 2),
            'phase_detected':  False,
            'note':            'Single cycle — full video window',
        }])

    rows = []
    for i, gs in enumerate(green_starts):
        end_s = green_starts[i+1] if i+1 < len(green_starts) else max_ts
        # Count RED and GREEN timesteps in this cycle
        mask  = (ts_stats['timestamp_s'] >= gs) & (ts_stats['timestamp_s'] < end_s)
        sub   = ts_stats[mask]
        n_red   = (sub['phase'] == 'RED').sum()
        n_green = (sub['phase'] == 'GREEN').sum()
        rows.append({
            'cycle':           i + 1,
            'start_s':         round(gs,          2),
            'end_s':           round(end_s,        2),
            'duration_s':      round(end_s - gs,   2),
            'phase_detected':  True,
            'n_red_steps':     int(n_red),
            'n_green_steps':   int(n_green),
            'mean_speed_kmh':  round(sub['mean_speed_kmh'].mean(), 2),
            'note':            'Detected from vehicle motion',
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7 — Merge detected cycles onto SSM results
# ---------------------------------------------------------------------------
def merge_cycles_onto_ssm(ssm_csv: str,
                           ts_stats: pd.DataFrame
                           ) -> pd.DataFrame:
    """
    Join signal_cycle and phase onto ssm_results.csv by timestamp.
    Uses nearest-timestamp merge to handle floating-point differences.
    """
    logger.info(f"Reading SSM results: {ssm_csv}")
    ssm = pd.read_csv(ssm_csv)

    # Round to 1 decimal for merge key (matches 0.2s sampling precision)
    ssm['_ts_key']      = ssm['timestamp_s'].round(1)
    phase_lkp           = ts_stats.copy()
    phase_lkp['_ts_key'] = phase_lkp['timestamp_s'].round(1)

    # Keep only what we need from phase data
    phase_lkp = phase_lkp[['_ts_key', 'signal_cycle', 'phase']].drop_duplicates('_ts_key')

    merged = ssm.merge(phase_lkp, on='_ts_key', how='left')
    merged.drop(columns=['_ts_key'], inplace=True)

    # Fill unmatched rows (edge timestamps) with cycle 1
    unmatched = merged['signal_cycle'].isna().sum()
    if unmatched:
        logger.warning(f"{unmatched} SSM rows had no matching time-step — "
                        f"assigned to cycle 1")
    merged['signal_cycle'] = merged['signal_cycle'].fillna(1).astype(int)
    merged['phase']        = merged['phase'].fillna('UNKNOWN')

    return merged


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def detect_and_assign(smoothed_csv:       str,
                       ssm_csv:            str,
                       output_phases_csv:  str,
                       output_ssm_csv:     str,
                       output_bounds_csv:  str,
                       stopped_speed:      float = DEFAULT_STOPPED_SPEED_KMH,
                       min_cycle:          float = DEFAULT_MIN_CYCLE_DURATION,
                       min_green:          float = DEFAULT_MIN_GREEN_DURATION,
                       smooth_window:      int   = DEFAULT_SMOOTH_WINDOW,
                       min_vehicles:       int   = DEFAULT_MIN_VEHICLES):

    logger.info("=" * 60)
    logger.info("  SIGNAL PHASE DETECTION FROM VEHICLE MOTION")
    logger.info("=" * 60)

    # Build speed time series
    ts_stats = build_speed_timeseries(smoothed_csv, min_vehicles)

    # Smooth
    ts_stats = smooth_speed_series(ts_stats, smooth_window)

    # Classify phases
    ts_stats = classify_phases(ts_stats, stopped_speed)

    # Detect cycle boundaries
    green_starts, single_cycle, reason = build_cycle_boundaries(
        ts_stats, min_cycle, min_green
    )

    logger.info(f"\nCycle Detection Result:")
    logger.info(f"  Mode   : {'SINGLE CYCLE' if single_cycle else 'MULTI CYCLE'}")
    logger.info(f"  Reason : {reason}")

    # Assign cycle numbers
    ts_stats = assign_cycles_to_timeseries(ts_stats, green_starts, single_cycle)

    # Build boundary table
    boundaries = build_boundary_table(ts_stats, green_starts, single_cycle)

    # Save phase time series
    ts_stats.drop(
        columns=['phase_int', 'phase_change',
                 'is_green_start', 'is_red_start'],
        errors='ignore'
    ).to_csv(output_phases_csv, index=False)
    logger.info(f"\nPhase time-series → {output_phases_csv}")

    # Save boundary table
    boundaries.to_csv(output_bounds_csv, index=False)
    logger.info(f"Cycle boundaries  → {output_bounds_csv}")
    logger.info("\n" + boundaries.to_string(index=False))

    # Merge onto SSM
    ssm_with_cycles = merge_cycles_onto_ssm(ssm_csv, ts_stats)
    ssm_with_cycles.to_csv(output_ssm_csv, index=False)
    logger.info(f"\nSSM with cycles   → {output_ssm_csv}")

    # Distribution report
    cycle_dist = ssm_with_cycles['signal_cycle'].value_counts().sort_index()
    logger.info("\nSSM rows per detected cycle:")
    for cyc, cnt in cycle_dist.items():
        logger.info(f"  Cycle {cyc:2d} : {cnt:,} rows")

    return ts_stats, boundaries, ssm_with_cycles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Detect signal phases from vehicle motion data"
    )
    ap.add_argument('--smoothed',
                    default='results_v2/smoothed.csv')
    ap.add_argument('--ssm',
                    default='results_v2/ssm_results.csv')
    ap.add_argument('--output-phases',
                    default='results_v2/signal_phases.csv')
    ap.add_argument('--output-ssm',
                    default='results_v2/ssm_with_cycles.csv')
    ap.add_argument('--output-boundaries',
                    default='results_v2/cycle_boundaries.csv')
    ap.add_argument('--stopped-speed',   type=float,
                    default=DEFAULT_STOPPED_SPEED_KMH,
                    help='Speed below which phase = RED (km/h)')
    ap.add_argument('--min-cycle',       type=float,
                    default=DEFAULT_MIN_CYCLE_DURATION,
                    help='Minimum valid cycle duration (seconds)')
    ap.add_argument('--min-green',       type=float,
                    default=DEFAULT_MIN_GREEN_DURATION,
                    help='Minimum green phase duration to avoid spurious detection')
    ap.add_argument('--smooth-window',   type=int,
                    default=DEFAULT_SMOOTH_WINDOW,
                    help='Rolling window size for speed smoothing')
    ap.add_argument('--min-vehicles',    type=int,
                    default=DEFAULT_MIN_VEHICLES,
                    help='Min vehicles per time-step to include in analysis')
    args = ap.parse_args()

    detect_and_assign(
        smoothed_csv      = args.smoothed,
        ssm_csv           = args.ssm,
        output_phases_csv = args.output_phases,
        output_ssm_csv    = args.output_ssm,
        output_bounds_csv = args.output_boundaries,
        stopped_speed     = args.stopped_speed,
        min_cycle         = args.min_cycle,
        min_green         = args.min_green,
        smooth_window     = args.smooth_window,
        min_vehicles      = args.min_vehicles,
    )


if __name__ == '__main__':
    main()