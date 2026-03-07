"""
06_run_pipeline.py  — Full traffic conflict analysis pipeline.

PIPELINE:
    Step 1  : 01_data_extractor.py       → trajectories.csv
    Step 2  : 02_trajectory_smoother.py  → smoothed.csv
    Step 3  : 03_leader_follower.py      → leader_follower_pairs.csv
    Step 4  : 04_ssm_calculator.py       → ssm_results.csv
    Step 4b : 05b_signal_detector.py     → ssm_with_cycles.csv  ← NEW
    Step 5  : 05_conflict_estimator.py   → conflicts.csv
"""
import argparse, logging, subprocess, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
PYTHON = sys.executable


def run(cmd: list, step_name: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP: {step_name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    subprocess.run([PYTHON] + cmd, check=True)
    logger.info(f"  ✓ {step_name} done in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(
        description="Traffic Conflict Analysis Pipeline — full end-to-end"
    )
    ap.add_argument('--video',             required=True)
    ap.add_argument('--model',             default='yolov8s.pt')
    ap.add_argument('--conf',              type=float, default=0.4)
    ap.add_argument('--video-fps',         type=float, default=0)
    ap.add_argument('--pixel-to-meter',    type=float, default=0.05)
    ap.add_argument('--loess-frac',        type=float, default=0.25)
    ap.add_argument('--min-track-points',  type=int,   default=5)
    ap.add_argument('--output-dir',        default='./results')

    # Signal detector parameters
    ap.add_argument('--stopped-speed',     type=float, default=5.0,
                    help='Speed below which phase = RED (km/h)')
    ap.add_argument('--min-cycle',         type=float, default=30.0,
                    help='Minimum valid cycle duration (seconds)')
    ap.add_argument('--min-green',         type=float, default=5.0,
                    help='Minimum green phase duration (seconds)')
    ap.add_argument('--smooth-window',     type=int,   default=7,
                    help='Speed smoothing window for phase detection')
    ap.add_argument('--min-vehicles',      type=int,   default=3,
                    help='Min vehicles per time-step for phase detection')

    args = ap.parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # File paths
    traj       = str(out / 'trajectories.csv')
    smooth     = str(out / 'smoothed.csv')
    pairs      = str(out / 'leader_follower_pairs.csv')
    ssm        = str(out / 'ssm_results.csv')
    phases     = str(out / 'signal_phases.csv')
    ssm_cyc    = str(out / 'ssm_with_cycles.csv')
    boundaries = str(out / 'cycle_boundaries.csv')
    conf       = str(out / 'conflicts.csv')

    t_total = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Extract raw trajectories
    # ------------------------------------------------------------------
    run([
        '01_data_extractor.py',
        '--input',   args.video,
        '--output',  traj,
        '--model',   args.model,
        '--conf',    str(args.conf),
        '--video-fps', str(args.video_fps),
    ], "Data Extraction  (YOLO + ByteTrack → CSV)")

    # ------------------------------------------------------------------
    # Step 2 — LOESS smooth + speed + acceleration
    # ------------------------------------------------------------------
    run([
        '02_trajectory_smoother.py',
        '--input',          traj,
        '--output',         smooth,
        '--frac',           str(args.loess_frac),
        '--pixel-to-meter', str(args.pixel_to_meter),
    ], "Trajectory Smoothing  (LOESS + derivatives)")

    # ------------------------------------------------------------------
    # Step 3 — Leader-follower identification
    # ------------------------------------------------------------------
    run([
        '03_leader_follower.py',
        '--input',            smooth,
        '--output',           pairs,
        '--min-track-points', str(args.min_track_points),
    ], "Leader-Follower Pair Identification")

    # ------------------------------------------------------------------
    # Step 4 — TTC & DRAC computation
    # ------------------------------------------------------------------
    run([
        '04_ssm_calculator.py',
        '--input',  pairs,
        '--output', ssm,
    ], "Surrogate Safety Measures  (TTC + DRAC)")

    # ------------------------------------------------------------------
    # Step 4b — Signal phase detection from vehicle motion  ← NEW
    # ------------------------------------------------------------------
    run([
        '05b_signal_detector.py',
        '--smoothed',           smooth,
        '--ssm',                ssm,
        '--output-phases',      phases,
        '--output-ssm',         ssm_cyc,
        '--output-boundaries',  boundaries,
        '--stopped-speed',      str(args.stopped_speed),
        '--min-cycle',          str(args.min_cycle),
        '--min-green',          str(args.min_green),
        '--smooth-window',      str(args.smooth_window),
        '--min-vehicles',       str(args.min_vehicles),
    ], "Signal Phase Detection  (from vehicle motion)")

    # ------------------------------------------------------------------
    # Step 5 — Critical conflict estimation using detected cycles
    # ------------------------------------------------------------------
    run([
        '05_conflict_estimator.py',
        '--input',  ssm_cyc,   # ← now uses ssm_with_cycles.csv, not ssm_results.csv
        '--output', conf,
    ], "Critical Conflict Estimation per Signal Cycle")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  PIPELINE COMPLETE  ({time.time()-t_total:.1f}s total)")
    logger.info(f"  Results in: {args.output_dir}")
    logger.info(f"\n  Key outputs:")
    logger.info(f"    {traj}")
    logger.info(f"    {smooth}")
    logger.info(f"    {pairs}")
    logger.info(f"    {ssm}")
    logger.info(f"    {phases}         ← phase per timestep")
    logger.info(f"    {boundaries}     ← cycle start/end times")
    logger.info(f"    {ssm_cyc}        ← SSM with real cycle labels")
    logger.info(f"    {conf}")
    logger.info(f"    {conf.replace('.csv','_summary.csv')}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()


    # python 06_run_pipeline.py `
    # --video intersection.mp4 `
    # --model yolov8s.pt `
    # --pixel-to-meter 0.05 `
    # --loess-frac 0.25 `
    # --output-dir ./results_v3/ `
    # --stopped-speed 5.0 `
    # --min-cycle 30.0 `
    # --min-green 5.0