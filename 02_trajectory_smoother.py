"""
02_trajectory_smoother.py
PURPOSE : LOESS smooth x,y trajectories per vehicle, then compute
          instantaneous speed (1st derivative) and acceleration (2nd derivative)
          as per IITB methodology notes.

SOURCE  : statsmodels.nonparametric.smoothers_lowess.lowess (official)
          Cleveland W.S. (1979) "Robust Locally Weighted Regression"
          J. American Statistical Association 74(368):829-836
          https://www.statsmodels.org/stable/generated/
               statsmodels.nonparametric.smoothers_lowess.lowess.html

KEY INSIGHT (from your handwritten notes, Page 2):
    Raw trajectory = discrete + noisy
    High-order polynomial = oscillatory + unrealistic (fits ALL points)
    Local smoothing (LOESS) = continuous + meaningful  ← USE THIS
    
    Speed     = 1st derivative of smoothed position
    Accel     = 2nd derivative of smoothed position

USAGE : python 02_trajectory_smoother.py \
            --input  trajectories.csv \
            --output smoothed.csv \
            --frac   0.3
"""
import argparse, logging
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# LOESS bandwidth — fraction of data used for each local regression
# 0.2–0.3 works well for 0.2s-sampled vehicle trajectories
# Larger frac = smoother but loses responsiveness to real maneuvers
DEFAULT_FRAC = 0.3
MIN_POINTS   = 5    # minimum observations per vehicle to attempt smoothing


def loess_smooth_and_derive(t: np.ndarray,
                             x: np.ndarray,
                             y: np.ndarray,
                             frac: float = DEFAULT_FRAC):
    """
    Apply LOESS to x(t) and y(t) trajectories, then compute derivatives.

    Algorithm (Cleveland 1979 via statsmodels):
        For each point i, take frac*N nearest neighbours by t-distance,
        fit a weighted linear regression using tricube weights,
        return smoothed values.

    Speed & acceleration:
        After smoothing, fit a CubicSpline to the LOESS output.
        Speed      = sqrt( (dx/dt)^2 + (dy/dt)^2 )   in m/s → km/h
        Accel      = d(speed)/dt                        in m/s^2

    Returns:
        dict with keys: t, x_smooth, y_smooth,
                        speed_mps, speed_kmh, accel_mps2
    """
    n = len(t)
    if n < MIN_POINTS:
        logger.warning(f"Only {n} points — too few for LOESS. Returning raw.")
        vx        = np.gradient(x, t)
        vy        = np.gradient(y, t)
        speed_mps = np.sqrt(np.maximum(vx**2 + vy**2, 0))
        return {
            't':         t,
            'x_smooth':  x,
            'y_smooth':  y,
            'vx_mps':    vx,           # ← was missing
            'vy_mps':    vy,           # ← was missing
            'speed_mps': speed_mps,
            'speed_kmh': speed_mps * 3.6,
            'accel_mps2': np.gradient(speed_mps, t),
        }

    # --- LOESS smoothing (statsmodels official) ---
    # sm_lowess(endog, exog, frac, it, return_sorted)
    # Note: statsmodels convention: sm_lowess(y, x, ...)
    x_smooth = sm_lowess(x, t, frac=frac, it=3, return_sorted=False)
    y_smooth = sm_lowess(y, t, frac=frac, it=3, return_sorted=False)

    # --- Derivatives via CubicSpline on smoothed output ---
    # CubicSpline gives analytical 1st and 2nd derivatives — more accurate
    # than finite differences on the sparse 0.2s grid
    cs_x = CubicSpline(t, x_smooth)
    cs_y = CubicSpline(t, y_smooth)

    # Instantaneous speed = magnitude of velocity vector
    vx       = cs_x(t, 1)   # 1st derivative dx/dt
    vy       = cs_y(t, 1)   # 1st derivative dy/dt
    speed_mps = np.sqrt(vx**2 + vy**2)
    speed_mps = np.maximum(speed_mps, 0)

    # Instantaneous acceleration = d|v|/dt
    ax        = cs_x(t, 2)   # 2nd derivative d²x/dt²
    ay        = cs_y(t, 2)   # 2nd derivative d²y/dt²
    # Scalar acceleration: project velocity-acceleration vector
    with np.errstate(divide='ignore', invalid='ignore'):
        accel = np.where(
            speed_mps > 0.01,
            (vx * ax + vy * ay) / speed_mps,
            0.0
        )

    return {
        't':         t,
        'x_smooth':  x_smooth,
        'y_smooth':  y_smooth,
        'vx_mps':    vx,
        'vy_mps':    vy,
        'speed_mps': speed_mps,
        'speed_kmh': speed_mps * 3.6,
        'accel_mps2': accel,
    }


def process(input_csv: str, output_csv: str,
            frac: float, pixel_to_meter: float):
    logger.info(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    df = df.sort_values(['vehicle_id', 'timestamp_s']).reset_index(drop=True)

    results = []
    vids    = df['vehicle_id'].unique()
    logger.info(f"Processing {len(vids)} vehicles  frac={frac}")

    for vid in vids:
        sub = df[df['vehicle_id'] == vid].copy()
        t   = sub['timestamp_s'].values
        # Convert pixels to metres using calibration factor
        x   = sub['x_front_px'].values  * pixel_to_meter
        y   = sub['y_front_px'].values  * pixel_to_meter

        if len(t) < 2:
            continue

        res  = loess_smooth_and_derive(t, x, y, frac=frac)

        for i in range(len(t)):
            row = sub.iloc[i].to_dict()
            row.update({
                'x_smooth_m':   round(float(res['x_smooth'][i]), 4),
                'y_smooth_m':   round(float(res['y_smooth'][i]), 4),
                'vx_mps':       round(float(res['vx_mps'][i]),   4),
                'vy_mps':       round(float(res['vy_mps'][i]),   4),
                'speed_mps':    round(float(res['speed_mps'][i]),4),
                'speed_kmh':    round(float(res['speed_kmh'][i]),4),
                'accel_mps2':   round(float(res['accel_mps2'][i]),4),
            })
            results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    logger.info(f"Smoothed data saved → {output_csv}  ({len(out)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',          default='trajectories.csv')
    ap.add_argument('--output',         default='smoothed.csv')
    ap.add_argument('--frac',  type=float, default=DEFAULT_FRAC,
                    help='LOESS bandwidth fraction (0.1–0.5)')
    ap.add_argument('--pixel-to-meter', type=float, default=0.05)
    args = ap.parse_args()
    process(args.input, args.output, args.frac, args.pixel_to_meter)

if __name__ == '__main__':
    main()