"""
03_leader_follower.py
PURPOSE : Identify critical leader-follower pairs for every subject vehicle
          at every time-step using the IITB virtual-strip + lateral overlap method.

SOURCE  : arXiv:2405.10665 — Leader-Follower Identification for Non-Lane-Based Traffic
          Your handwritten notes (Pages 4, 6, 7):
              Lx = x_L - x_S - l_L          (longitudinal clear gap)
              Ly = |Ys - Yi| - ws/2 - wi/2  (lateral overlap; negative = overlapping)
          DRISHTE-E (IIT Bombay, GeorgeVJose/DRISHTE-Public)

ALGORITHM:
    For each subject vehicle S at each time-step:
        1. Create virtual strip of width = w_S centred on y_S
        2. Find all candidate leaders L where:
             a. x_L > x_S  (L is ahead of S in direction of traffic)
             b. Ly(S,L) <= 0  (lateral overlap exists)
        3. From candidates, select the one with MINIMUM Lx (closest ahead)
        4. Store (subject_id, leader_id, timestamp, Lx, Ly)

NOTE: In mixed traffic a vehicle can have MULTIPLE leaders.
      This script returns ALL overlapping leaders per time-step,
      ranked by Lx (closest first).

USAGE : python 03_leader_follower.py \
            --input smoothed.csv \
            --output leader_follower_pairs.csv
"""
import argparse, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Threshold: Ly <= LY_THRESHOLD counts as lateral overlap
# 0.0 = strict overlap; small positive value = small tolerance for near-misses
LY_THRESHOLD   = 0.1   # metres
MAX_LX_SEARCH  = 50.0  # metres — only look this far ahead


def compute_lx_ly(xs, ys, ws, xl, yl, wl, ll):
    """
    Longitudinal gap and lateral overlap between subject S and leader L.

    From IITB notes (Page 4 & 7):
        Lx = x_L - x_S - l_L
        Ly = |Ys - Yi| - ws/2 - wi/2

    where:
        (xs, ys)  = front-centre of subject vehicle  (metres)
        (xl, yl)  = front-centre of leader vehicle   (metres)
        ws, wl    = widths of subject and leader      (metres)
        ll        = length of leader vehicle          (metres)

    Returns:
        lx (float): longitudinal clear gap (m); negative = overlap
        ly (float): lateral clearance (m); negative = lateral overlap
    """
    lx = (xl - xs) - ll
    ly = abs(ys - yl) - (ws / 2.0) - (wl / 2.0)
    return lx, ly


def identify_pairs(df: pd.DataFrame, min_track_points: int = 5) -> pd.DataFrame:
    pairs     = []

    # Filter out short-lived track fragments — not meaningful for conflict analysis
    track_lengths = df.groupby('vehicle_id')['timestamp_s'].count()
    valid_ids     = track_lengths[track_lengths >= min_track_points].index
    filtered_out  = len(track_lengths) - len(valid_ids)
    df            = df[df['vehicle_id'].isin(valid_ids)].copy()

    logger.info(f"Vehicles total       : {len(track_lengths)}")
    logger.info(f"Vehicles filtered out: {filtered_out} (< {min_track_points} points)")
    logger.info(f"Vehicles kept        : {len(valid_ids)}")

    all_times = sorted(df['timestamp_s'].unique())
    logger.info(f"Identifying pairs across {len(all_times)} time-steps …")

    for ts in all_times:
        snapshot = df[df['timestamp_s'] == ts]
        if len(snapshot) < 2:
            continue

        vids = snapshot['vehicle_id'].values

        for _, subject in snapshot.iterrows():
            xs = float(subject['x_smooth_m'])
            ys = float(subject['y_smooth_m'])
            ws = float(subject['vehicle_width_m'])
            vs = float(subject['speed_mps'])
            sid = int(subject['vehicle_id'])

            candidates = []

            for _, candidate in snapshot.iterrows():
                cid = int(candidate['vehicle_id'])
                if cid == sid:
                    continue

                xl = float(candidate['x_smooth_m'])
                yl = float(candidate['y_smooth_m'])
                wl = float(candidate['vehicle_width_m'])
                ll = float(candidate['vehicle_length_m'])
                vl = float(candidate['speed_mps'])

                # Must be AHEAD in direction of traffic (x_L > x_S)
                if xl <= xs:
                    continue

                lx, ly = compute_lx_ly(xs, ys, ws, xl, yl, wl, ll)

                # Too far ahead — skip
                if lx > MAX_LX_SEARCH:
                    continue

                # Lateral overlap check: Ly <= LY_THRESHOLD
                if ly > LY_THRESHOLD:
                    continue

                candidates.append({
                    'timestamp_s':    ts,
                    'follower_id':    sid,
                    'leader_id':      cid,
                    'follower_type':  subject['vehicle_type'],
                    'leader_type':    candidate['vehicle_type'],
                    'xs':             round(xs, 4),
                    'ys':             round(ys, 4),
                    'xl':             round(xl, 4),
                    'yl':             round(yl, 4),
                    'Lx_m':          round(lx, 4),
                    'Ly_m':          round(ly, 4),
                    'v_follower_mps': round(vs, 4),
                    'v_leader_mps':   round(vl, 4),
                    'leader_length_m': ll,
                    'a_follower_mps2': round(float(subject['accel_mps2']), 4),
                    'a_leader_mps2':   round(float(candidate['accel_mps2']), 4),
                })

            # Sort by Lx and tag rank (1 = closest leader)
            candidates.sort(key=lambda d: d['Lx_m'])
            for rank, c in enumerate(candidates, start=1):
                c['leader_rank'] = rank
                pairs.append(c)

    result = pd.DataFrame(pairs)
    logger.info(f"Found {len(result)} leader-follower pair records.")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',            default='smoothed.csv')
    ap.add_argument('--output',           default='leader_follower_pairs.csv')
    ap.add_argument('--min-track-points', type=int, default=5,
                    help='Minimum observations per vehicle to include in analysis')
    args = ap.parse_args()

    df  = pd.read_csv(args.input)
    out = identify_pairs(df, min_track_points=args.min_track_points)
    out.to_csv(args.output, index=False)
    logger.info(f"Saved → {args.output}")

if __name__ == '__main__':
    main()