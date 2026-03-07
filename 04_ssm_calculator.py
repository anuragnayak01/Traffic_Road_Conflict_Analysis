"""
04_ssm_calculator.py
PURPOSE : Compute TTC and DRAC for every leader-follower pair at every time-step.

SOURCE  : Yiru-Jiao/Two-Dimensional-Time-To-Collision (TU Delft, GitHub)
          https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision
          TRB Paper (Souza 2011) — standard thresholds:
              TTC threshold  = 1.5 s
              DRAC threshold = 3.35 m/s²
          Your handwritten notes (Pages 4, 5):
              TTC  = (x_L - x_F - l_L) / (v_F - v_L)   if v_F > v_L
              DRAC = (v_F - v_L)² / 2(x_L - x_F - l_L)

FORMULAS:
    From notes Page 5:
        TTC  = [ (x_L - x_F - l_L) ] / (v_F - v_L)   only when v_F > v_L
        DRAC = (v_F - v_L)²  /  2*(x_L - x_F - l_L)

    Minimum TTC across all time instants → threshold for conflict severity
    Maximum DRAC across all time instants → if > 3.35 m/s² = critical conflict

USAGE : python 04_ssm_calculator.py \
            --input  leader_follower_pairs.csv \
            --output ssm_results.csv
"""
import argparse, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard thresholds from TRB / literature
TTC_THRESHOLD_S    = 1.5    # seconds  — below this = unsafe
DRAC_THRESHOLD_MPS2 = 3.35  # m/s²    — above this = critical rear-end conflict
TTC_MAX_VALID      = 999.0  # cap for TTC to avoid inf pollution

# Minimum gap to avoid numerical instability (metres)
MIN_GAP_M = 0.1


def compute_ttc(v_follower, v_leader, lx):
    """
    Time-To-Collision (1-D longitudinal, from IITB notes Page 4).

        TTC = Lx / (v_F - v_L)   only when v_F > v_L (follower is faster)

    Returns float:
        > 0       : valid TTC in seconds
        np.inf    : v_F <= v_L (no collision course)
        np.nan    : invalid gap (lx <= 0)
    """
    if lx <= MIN_GAP_M:
        return np.nan           # vehicles overlapping — already in conflict
    dv = v_follower - v_leader
    if dv <= 0:
        return np.inf           # follower not faster — no closing
    ttc = lx / dv
    return min(ttc, TTC_MAX_VALID)


def compute_drac(v_follower, v_leader, lx):
    """
    Deceleration Rate to Avoid Crash (from IITB notes Page 5).

        DRAC = (v_F - v_L)² / 2*Lx

    Returns float:
        >= 0      : required deceleration in m/s²
        np.nan    : invalid gap
    """
    if lx <= MIN_GAP_M:
        return np.nan
    dv = v_follower - v_leader
    if dv <= 0:
        return 0.0              # follower not faster — no required decel
    return (dv ** 2) / (2.0 * lx)


def process(input_csv, output_csv):
    logger.info(f"Reading pairs: {input_csv}")
    df = pd.read_csv(input_csv)

    # Compute TTC and DRAC for each row (each pair at each time-step)
    ttc_vals  = []
    drac_vals = []

    for _, row in df.iterrows():
        vf  = float(row['v_follower_mps'])
        vl  = float(row['v_leader_mps'])
        lx  = float(row['Lx_m'])

        ttc_vals.append( compute_ttc(vf, vl, lx)  )
        drac_vals.append(compute_drac(vf, vl, lx) )

    df['TTC_s']     = ttc_vals
    df['DRAC_mps2'] = drac_vals

    # Tag individual time-step as unsafe
    df['TTC_unsafe']  = df['TTC_s']    < TTC_THRESHOLD_S
    df['DRAC_unsafe'] = df['DRAC_mps2'] > DRAC_THRESHOLD_MPS2

    df.to_csv(output_csv, index=False)
    logger.info(f"SSM results saved → {output_csv}  ({len(df)} rows)")

    # Summary stats
    total   = len(df.dropna(subset=['TTC_s']))
    unsafe_ttc  = df['TTC_unsafe'].sum()
    unsafe_drac = df['DRAC_unsafe'].sum()
    logger.info(f"  Valid TTC rows      : {total}")
    logger.info(f"  TTC unsafe (<1.5s)  : {unsafe_ttc}")
    logger.info(f"  DRAC unsafe (>3.35) : {unsafe_drac}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  default='leader_follower_pairs.csv')
    ap.add_argument('--output', default='ssm_results.csv')
    args = ap.parse_args()
    process(args.input, args.output)

if __name__ == '__main__':
    main()