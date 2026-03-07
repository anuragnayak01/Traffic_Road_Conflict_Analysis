"""
07_conflict_annotator.py
PURPOSE : Extract and annotate video frames where critical conflicts occur.
          For each conflict, draws both vehicles with:
            - Bounding boxes (RED for follower, ORANGE for leader)
            - Vehicle ID, type, speed (km/h), acceleration (m/s²)
            - Longitudinal gap Lx (metres)
            - TTC and DRAC values
            - Connecting line between vehicles with distance label

APPROACH:
    1. Load conflicts.csv → get all critical (follower, leader, timestamp) triples
    2. Look up original pixel bounding boxes from trajectories.csv by
       (vehicle_id, frame_number) — no need to re-run YOLO
    3. Also load speed/accel from smoothed.csv
    4. Seek video to exact frame, annotate, save

OUTPUT:
    - Individual annotated frames as PNG (one per conflict event)
    - Compiled conflict highlight video (conflict_highlights.mp4)
    - Summary overlay showing TTC, DRAC, Lx, speeds on frame

USAGE:
    python 07_conflict_annotator.py \
        --video          intersection.mp4 \
        --trajectories   results_v2/trajectories.csv \
        --smoothed       results_v2/smoothed.csv \
        --conflicts      results_v2/conflicts.csv \
        --ssm            results_v2/ssm_results.csv \
        --output-dir     results_v2/conflict_frames/ \
        --highlight-video results_v2/conflict_highlights.mp4
"""

import cv2
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------
COLOR_FOLLOWER   = (0,   0,   255)   # Red   — the vehicle at risk
COLOR_LEADER     = (0,  140,  255)   # Orange — the vehicle ahead
COLOR_SAFE       = (0,  200,   0)    # Green — non-conflicting vehicles
COLOR_LINE       = (0,  255,  255)   # Yellow — connecting line
COLOR_TEXT_BG    = (20,  20,  20)    # Dark background for text
COLOR_TEXT       = (255, 255, 255)   # White text
COLOR_PANEL_BG   = (30,  30,  30)    # Info panel background
COLOR_CRITICAL   = (0,   0,  255)    # Red for critical values
COLOR_WARNING    = (0,  165, 255)    # Orange for warning values
COLOR_OK         = (0,  200,  0)     # Green for safe values

FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL       = 0.45
FONT_MEDIUM      = 0.55
FONT_LARGE       = 0.75
THICKNESS        = 2
PANEL_WIDTH      = 340   # right-side info panel width in pixels
CONTEXT_FRAMES   = 3     # extra frames before/after conflict to include


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_label_box(frame, text_lines, x, y, font_scale=FONT_SMALL,
                   bg_color=COLOR_TEXT_BG, text_color=COLOR_TEXT,
                   padding=5):
    """Draw a multi-line text box with background at (x, y)."""
    line_h = int(font_scale * 30) + padding
    w_max  = max(
        cv2.getTextSize(t, FONT, font_scale, 1)[0][0]
        for t in text_lines
    ) + padding * 2

    h_total = line_h * len(text_lines) + padding
    # Clamp to frame bounds
    x = max(0, min(x, frame.shape[1] - w_max))
    y = max(0, min(y, frame.shape[0] - h_total))

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w_max, y + h_total),
                  bg_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i, line in enumerate(text_lines):
        ty = y + padding + (i + 1) * line_h - padding
        cv2.putText(frame, line, (x + padding, ty),
                    FONT, font_scale, text_color, 1, cv2.LINE_AA)
    return frame


def draw_vehicle_box(frame, x1, y1, x2, y2, color, label_lines,
                     thickness=THICKNESS):
    """Draw bounding box + label for one vehicle."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Corner accent marks
    corner_len = 12
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),
                            (x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), color, thickness+1)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), color, thickness+1)

    label_y = y1 - 5 if y1 > 60 else y2 + 5
    draw_label_box(frame, label_lines, x1, label_y,
                   bg_color=color, text_color=COLOR_TEXT)
    return frame


def draw_gap_line(frame, follower_box, leader_box, lx_m, color=COLOR_LINE):
    """Draw line between vehicle fronts showing the gap."""
    fx1,fy1,fx2,fy2 = follower_box
    lx1,ly1,lx2,ly2 = leader_box

    # Front-centre of follower (right edge) and rear of leader (left edge)
    f_front = (int(fx2), int((fy1 + fy2) / 2))
    l_rear  = (int(lx1), int((ly1 + ly2) / 2))

    cv2.line(frame, f_front, l_rear, color, 2, cv2.LINE_AA)
    # Distance label at midpoint
    mid_x = (f_front[0] + l_rear[0]) // 2
    mid_y = (f_front[1] + l_rear[1]) // 2
    draw_label_box(frame, [f"Lx={lx_m:.2f}m"],
                   mid_x - 30, mid_y - 10,
                   font_scale=FONT_SMALL,
                   bg_color=(50, 50, 50))
    return frame


def draw_info_panel(frame, conflict_info: dict):
    """
    Draw a right-side information panel with all conflict metrics.
    Returns frame with panel attached on the right.
    """
    h, w = frame.shape[:2]
    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = COLOR_PANEL_BG

    # Separator line
    cv2.line(panel, (0, 0), (0, h), (80, 80, 80), 2)

    y   = 20
    gap = 28

    def put(text, color=COLOR_TEXT, scale=FONT_MEDIUM, bold=False):
        nonlocal y
        thick = 2 if bold else 1
        cv2.putText(panel, text, (10, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += gap

    def separator():
        nonlocal y
        cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), (70, 70, 70), 1)
        y += 10

    # Header
    put("CONFLICT DETECTED", COLOR_CRITICAL, FONT_LARGE, bold=True)
    separator()

    # Cycle & timestamp
    put(f"Signal Cycle : {conflict_info.get('signal_cycle', '?')}")
    put(f"Timestamp    : {conflict_info.get('timestamp_s', 0):.2f}s")
    put(f"Frame        : {conflict_info.get('frame_number', '?')}")
    separator()

    # Follower info
    put("[ FOLLOWER ]", COLOR_FOLLOWER, bold=True)
    put(f"  ID     : {conflict_info.get('follower_id', '?')}")
    put(f"  Type   : {conflict_info.get('follower_type', '?')}")
    spd_f = conflict_info.get('speed_follower_kmh', 0)
    acc_f = conflict_info.get('accel_follower', 0)
    put(f"  Speed  : {spd_f:.1f} km/h")
    acc_color = COLOR_CRITICAL if abs(acc_f) > 3 else COLOR_TEXT
    put(f"  Accel  : {acc_f:+.2f} m/s2", acc_color)
    separator()

    # Leader info
    put("[ LEADER ]", COLOR_LEADER, bold=True)
    put(f"  ID     : {conflict_info.get('leader_id', '?')}")
    put(f"  Type   : {conflict_info.get('leader_type', '?')}")
    spd_l = conflict_info.get('speed_leader_kmh', 0)
    acc_l = conflict_info.get('accel_leader', 0)
    put(f"  Speed  : {spd_l:.1f} km/h")
    put(f"  Accel  : {acc_l:+.2f} m/s2")
    separator()

    # Gap
    lx = conflict_info.get('Lx_m', 0)
    lx_color = COLOR_CRITICAL if lx < 2 else (COLOR_WARNING if lx < 5 else COLOR_OK)
    put(f"Gap Lx   : {lx:.2f} m", lx_color, bold=True)
    separator()

    # SSM values
    put("[ SAFETY MEASURES ]", COLOR_TEXT, bold=True)
    ttc  = conflict_info.get('TTC_s', float('inf'))
    drac = conflict_info.get('DRAC_mps2', 0)

    ttc_color  = COLOR_CRITICAL if ttc  < 1.5  else COLOR_OK
    drac_color = COLOR_CRITICAL if drac > 3.35 else COLOR_OK

    ttc_str = f"{ttc:.2f}s" if ttc < 999 else "inf"
    put(f"  TTC    : {ttc_str}", ttc_color, bold=(ttc < 1.5))
    put(f"  Thresh : <1.50s", (100,100,100), FONT_SMALL)
    put(f"  DRAC   : {drac:.2f} m/s2", drac_color, bold=(drac > 3.35))
    put(f"  Thresh : >3.35 m/s2", (100,100,100), FONT_SMALL)
    separator()

    # Conflict type
    c_ttc  = conflict_info.get('critical_TTC',  False)
    c_drac = conflict_info.get('critical_DRAC', False)
    if c_ttc and c_drac:
        ctype = "TTC + DRAC"
    elif c_ttc:
        ctype = "TTC only"
    elif c_drac:
        ctype = "DRAC only"
    else:
        ctype = "None"
    put(f"Conflict : {ctype}", COLOR_CRITICAL, bold=True)

    # Watermark
    cv2.putText(panel, "IITB Methodology", (10, h - 15),
                FONT, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

    return np.hstack([frame, panel])


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_data(trajectories_csv, smoothed_csv, conflicts_csv, ssm_csv):
    logger.info("Loading data files …")

    traj  = pd.read_csv(trajectories_csv)
    smooth = pd.read_csv(smoothed_csv)
    conf  = pd.read_csv(conflicts_csv)
    ssm   = pd.read_csv(ssm_csv)

    # Keep only critical conflicts
    critical_ssm = ssm[ssm['is_critical'] == True].copy() \
        if 'is_critical' in ssm.columns \
        else ssm[(ssm['TTC_s'] < 1.5) | (ssm['DRAC_mps2'] > 3.35)].copy()

    logger.info(f"  Trajectories : {len(traj):,} rows")
    logger.info(f"  Smoothed     : {len(smooth):,} rows")
    logger.info(f"  Critical SSM : {len(critical_ssm):,} rows")

    # Build lookup: (vehicle_id, timestamp_s) → bounding box pixels
    traj['ts_key'] = traj['timestamp_s'].round(1)
    bbox_lookup = traj.set_index(['vehicle_id', 'ts_key'])[
        ['frame_number', 'x_min', 'y_min', 'x_max', 'y_max']
    ].to_dict('index')

    # Build lookup: (vehicle_id, timestamp_s) → speed_kmh, accel
    smooth['ts_key'] = smooth['timestamp_s'].round(1)
    smooth_lookup = smooth.set_index(['vehicle_id', 'ts_key'])[
        ['speed_kmh', 'accel_mps2', 'frame_number']
    ].to_dict('index')

    return critical_ssm, bbox_lookup, smooth_lookup


def get_bbox(lookup, vid, ts):
    """Get pixel bounding box for a vehicle at a timestamp."""
    key = (int(vid), round(float(ts), 1))
    return lookup.get(key, None)


def get_smooth(lookup, vid, ts):
    """Get smoothed speed/accel for a vehicle at a timestamp."""
    key = (int(vid), round(float(ts), 1))
    return lookup.get(key, None)


# ---------------------------------------------------------------------------
# Core annotator
# ---------------------------------------------------------------------------
def annotate_conflicts(video_path, critical_ssm, bbox_lookup, smooth_lookup,
                       output_dir, highlight_video_path, fps, max_conflicts=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Highlight video writer (frame + panel)
    highlight_writer = None
    if highlight_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        highlight_writer = cv2.VideoWriter(
            highlight_video_path, fourcc, fps,
            (W + PANEL_WIDTH, H)
        )
        logger.info(f"Highlight video: {highlight_video_path}")

    # Sort conflicts by timestamp for efficient frame seeking
    critical_ssm = critical_ssm.sort_values('timestamp_s').reset_index(drop=True)
    if max_conflicts:
        critical_ssm = critical_ssm.head(max_conflicts)

    saved_frames = set()   # avoid duplicate frame saves
    annotated_count = 0

    logger.info(f"Annotating {len(critical_ssm)} critical conflict records …")

    for _, row in critical_ssm.iterrows():
        ts       = float(row['timestamp_s'])
        fid      = int(row['follower_id'])
        lid      = int(row['leader_id'])

        # Look up bounding boxes
        f_bbox_data = get_bbox(bbox_lookup, fid, ts)
        l_bbox_data = get_bbox(bbox_lookup, lid, ts)

        if f_bbox_data is None or l_bbox_data is None:
            continue   # no pixel data for this pair at this timestamp

        frame_no = int(f_bbox_data['frame_number'])

        if frame_no in saved_frames:
            continue
        saved_frames.add(frame_no)

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - CONTEXT_FRAMES))
        ok, frame = cap.read()
        if not ok:
            continue

        # Seek to exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        # Smooth data
        f_smooth = get_smooth(smooth_lookup, fid, ts) or {}
        l_smooth = get_smooth(smooth_lookup, lid, ts) or {}

        spd_f_kmh  = float(f_smooth.get('speed_kmh',  0))
        acc_f      = float(f_smooth.get('accel_mps2', 0))
        spd_l_kmh  = float(l_smooth.get('speed_kmh',  0))
        acc_l      = float(l_smooth.get('accel_mps2', 0))

        f_box = (f_bbox_data['x_min'], f_bbox_data['y_min'],
                 f_bbox_data['x_max'], f_bbox_data['y_max'])
        l_box = (l_bbox_data['x_min'], l_bbox_data['y_min'],
                 l_bbox_data['x_max'], l_bbox_data['y_max'])

        ttc   = float(row.get('TTC_s',     999))
        drac  = float(row.get('DRAC_mps2',  0))
        lx    = float(row.get('Lx_m',       0))

        # --- Draw all detections in frame in green (background context) ---
        # Find all vehicles at this timestamp
        ts_key = round(ts, 1)
        for key, bdata in bbox_lookup.items():
            vid_k, ts_k = key
            if ts_k != ts_key:
                continue
            if vid_k in (fid, lid):
                continue
            bx1 = int(bdata['x_min']); by1 = int(bdata['y_min'])
            bx2 = int(bdata['x_max']); by2 = int(bdata['y_max'])
            cv2.rectangle(frame, (bx1,by1), (bx2,by2), COLOR_SAFE, 1)

        # --- Draw gap line ---
        draw_gap_line(frame, f_box, l_box, lx)

        # --- Draw follower (RED) ---
        f_labels = [
            f"ID:{fid} [{row.get('follower_type','?')}]",
            f"v:{spd_f_kmh:.1f}km/h",
            f"a:{acc_f:+.2f}m/s2",
            "FOLLOWER",
        ]
        draw_vehicle_box(frame,
                         f_box[0], f_box[1], f_box[2], f_box[3],
                         COLOR_FOLLOWER, f_labels)

        # --- Draw leader (ORANGE) ---
        l_labels = [
            f"ID:{lid} [{row.get('leader_type','?')}]",
            f"v:{spd_l_kmh:.1f}km/h",
            f"a:{acc_l:+.2f}m/s2",
            "LEADER",
        ]
        draw_vehicle_box(frame,
                         l_box[0], l_box[1], l_box[2], l_box[3],
                         COLOR_LEADER, l_labels)

        # --- Timestamp watermark ---
        cv2.putText(frame, f"t={ts:.2f}s  frame={frame_no}",
                    (10, 25), FONT, FONT_MEDIUM, COLOR_TEXT, 1, cv2.LINE_AA)

        # --- Build info panel ---
        conflict_info = {
            'signal_cycle':        row.get('signal_cycle', '?'),
            'timestamp_s':         ts,
            'frame_number':        frame_no,
            'follower_id':         fid,
            'follower_type':       row.get('follower_type', '?'),
            'leader_id':           lid,
            'leader_type':         row.get('leader_type', '?'),
            'speed_follower_kmh':  spd_f_kmh,
            'speed_leader_kmh':    spd_l_kmh,
            'accel_follower':      acc_f,
            'accel_leader':        acc_l,
            'Lx_m':                lx,
            'TTC_s':               ttc,
            'DRAC_mps2':           drac,
            'critical_TTC':        bool(row.get('critical_TTC',  False)),
            'critical_DRAC':       bool(row.get('critical_DRAC', False)),
        }
        annotated = draw_info_panel(frame, conflict_info)

        # --- Save PNG ---
        png_path = output_dir / f"conflict_f{fid}_l{lid}_t{ts:.1f}.png"
        cv2.imwrite(str(png_path), annotated)

        # --- Write to highlight video ---
        if highlight_writer:
            # Write CONTEXT_FRAMES×2+1 copies for ~0.5s pause on each conflict
            for _ in range(int(fps * 0.8)):
                highlight_writer.write(annotated)

        annotated_count += 1
        if annotated_count % 50 == 0:
            logger.info(f"  Annotated {annotated_count} frames so far …")

    cap.release()
    if highlight_writer:
        highlight_writer.release()

    logger.info(f"\nDone!")
    logger.info(f"  Annotated frames saved : {annotated_count}")
    logger.info(f"  Output directory       : {output_dir}")
    if highlight_video_path:
        logger.info(f"  Highlight video        : {highlight_video_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Annotate video frames where traffic conflicts occur"
    )
    ap.add_argument('--video',           required=True,
                    help='Original input video path')
    ap.add_argument('--trajectories',    default='results_v2/trajectories.csv')
    ap.add_argument('--smoothed',        default='results_v2/smoothed.csv')
    ap.add_argument('--conflicts',       default='results_v2/conflicts.csv')
    ap.add_argument('--ssm',             default='results_v2/ssm_results.csv')
    ap.add_argument('--output-dir',      default='results_v2/conflict_frames/')
    ap.add_argument('--highlight-video', default='results_v2/conflict_highlights.mp4')
    ap.add_argument('--video-fps',       type=float, default=25.0)
    ap.add_argument('--max-conflicts',   type=int,   default=None,
                    help='Limit number of conflicts to annotate (default: all)')
    args = ap.parse_args()

    critical_ssm, bbox_lookup, smooth_lookup = load_data(
        args.trajectories,
        args.smoothed,
        args.conflicts,
        args.ssm,
    )

    annotate_conflicts(
        video_path          = args.video,
        critical_ssm        = critical_ssm,
        bbox_lookup         = bbox_lookup,
        smooth_lookup       = smooth_lookup,
        output_dir          = args.output_dir,
        highlight_video_path= args.highlight_video,
        fps                 = args.video_fps,
        max_conflicts       = args.max_conflicts,
    )


if __name__ == '__main__':
    main()