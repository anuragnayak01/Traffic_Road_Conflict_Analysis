"""
SIGNAL — Traffic Conflict Analysis demo
----------------------------------------
Streamlit front-end for the Traffic_Road_Conflict_Analysis pipeline
(01_data_extractor.py -> 07_conflict_annotator.py).

WIRING NOTES (adjust the three spots marked TODO to match your actual
pipeline's CLI args / output schema — I don't have visibility into your
06_run_pipeline.py argument names or the exact results_v3 CSV columns,
so this uses reasonable placeholders you can swap in fast):
  1. run_pipeline()      -> subprocess command + args
  2. load_conflict_table() -> path + column names inside results_v3/
  3. load_annotated_video() -> output video filename pattern
"""

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config + design tokens
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SIGNAL — Traffic Conflict Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results_v3"

COLORS = {
    "bg": "#15171B",
    "surface": "#1D2025",
    "line": "#33383F",
    "amber": "#F2A93B",   # active / running signal
    "cyan": "#52C4D0",    # measured / complete
    "red": "#E5533D",     # severe conflict
    "text": "#ECEAE3",
    "muted": "#8B909A",
}

PIPELINE_STAGES = [
    ("01", "Extract", "01_data_extractor.py"),
    ("02", "Smooth", "02_trajectory_smoother.py"),
    ("03", "Leader/Follower", "03_leader_follower.py"),
    ("04", "SSM Calc", "04_ssm_calculator.py"),
    ("05", "Conflict Est.", "05_conflict_estimator.py"),
    ("05b", "Signal Detect", "05b_signal_detector.py"),
    ("07", "Annotate", "07_conflict_annotator.py"),
]

# --------------------------------------------------------------------------
# CSS — asphalt / signal-phase design system
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {COLORS['bg']};
        }}
        /* eyebrow / masthead */
        .signal-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: {COLORS['amber']};
            margin-bottom: 0.25rem;
        }}
        .signal-title {{
            font-size: 2.1rem;
            font-weight: 600;
            color: {COLORS['text']};
            margin: 0 0 0.15rem 0;
        }}
        .signal-sub {{
            color: {COLORS['muted']};
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }}
        /* instrument strip */
        .instrument-row {{
            display: flex;
            gap: 1px;
            background: {COLORS['line']};
            border: 1px solid {COLORS['line']};
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 1.8rem;
        }}
        .instrument {{
            flex: 1;
            background: {COLORS['surface']};
            padding: 1rem 1.2rem;
        }}
        .instrument-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {COLORS['muted']};
            margin-bottom: 0.4rem;
        }}
        .instrument-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.9rem;
            font-weight: 700;
            color: {COLORS['text']};
        }}
        .instrument-value.alert {{ color: {COLORS['red']}; }}
        .instrument-value.measure {{ color: {COLORS['cyan']}; }}
        /* pipeline stepper styled as a signal-phase timeline */
        .phase-strip {{
            display: flex;
            align-items: center;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 6px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 1.8rem;
            overflow-x: auto;
        }}
        .phase-step {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            white-space: nowrap;
        }}
        .phase-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: {COLORS['line']};
            flex-shrink: 0;
        }}
        .phase-dot.done {{ background: {COLORS['cyan']}; }}
        .phase-dot.active {{
            background: {COLORS['amber']};
            box-shadow: 0 0 0 3px rgba(242,169,59,0.18);
        }}
        .phase-num {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: {COLORS['muted']};
        }}
        .phase-label {{
            font-size: 0.78rem;
            color: {COLORS['text']};
        }}
        .phase-connector {{
            width: 28px;
            height: 1px;
            background: {COLORS['line']};
            margin: 0 0.7rem;
            flex-shrink: 0;
        }}
        /* sidebar as a control panel */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['surface']};
            border-right: 1px solid {COLORS['line']};
        }}
        section[data-testid="stSidebar"] .stButton button {{
            background-color: {COLORS['amber']};
            color: #1A1200;
            font-weight: 600;
            border: none;
            border-radius: 4px;
            width: 100%;
        }}
        section[data-testid="stSidebar"] .stButton button:hover {{
            background-color: #FFC266;
        }}
        /* empty state */
        .empty-state {{
            border: 1px dashed {COLORS['line']};
            border-radius: 6px;
            padding: 3rem 1.5rem;
            text-align: center;
            color: {COLORS['muted']};
            font-size: 0.9rem;
        }}
        .severity-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 0.15rem 0.5rem;
            border-radius: 3px;
            font-weight: 600;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="signal-eyebrow">Surrogate Safety Measures · Signalized Intersections</div>', unsafe_allow_html=True)
st.markdown('<div class="signal-title">SIGNAL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signal-sub">Upload intersection footage. Vehicles are tracked, '
    'leader/follower pairs resolved, and conflicts scored by time-to-collision (TTC) '
    'and post-encroachment time (PET).</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — control panel
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="signal-eyebrow">Input</div>', unsafe_allow_html=True)
    site_name = st.text_input("Intersection / site label", placeholder="e.g. MG Road x College Rd")
    uploaded_video = st.file_uploader("Video footage", type=["mp4", "avi", "mov"])
    st.markdown("---")
    st.markdown('<div class="signal-eyebrow">Detection</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("YOLOv8 confidence threshold", 0.1, 0.9, 0.35, 0.05)
    run_clicked = st.button("Run analysis", disabled=uploaded_video is None, use_container_width=True)

# --------------------------------------------------------------------------
# Pipeline runner
# --------------------------------------------------------------------------
def run_pipeline(video_path: Path, confidence: float) -> bool:
    """Kick off the existing pipeline against the uploaded video.

    TODO: replace the argument names below with whatever 06_run_pipeline.py
    actually expects (e.g. --input, --conf, --out-dir).
    """
    cmd = [
        "python", str(REPO_ROOT / "06_run_pipeline.py"),
        "--input", str(video_path),
        "--conf", str(confidence),
        "--output-dir", str(RESULTS_DIR),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        return True
    except Exception as e:
        st.error(f"Pipeline run failed: {e}")
        return False


def load_conflict_table() -> pd.DataFrame | None:
    """TODO: point this at the real results_v3 CSV and column names."""
    candidate = RESULTS_DIR / "conflicts.csv"
    if not candidate.exists():
        return None
    df = pd.read_csv(candidate)
    return df


def load_annotated_video() -> Path | None:
    """TODO: match this to whatever 07_conflict_annotator.py names its output."""
    candidate = RESULTS_DIR / "annotated_output.mp4"
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------
# Run + persist state
# --------------------------------------------------------------------------
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

if run_clicked and uploaded_video is not None:
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / uploaded_video.name
    video_path.write_bytes(uploaded_video.getbuffer())

    with st.status("Running pipeline stages 01 → 07...", expanded=True) as status:
        for code, label, script in PIPELINE_STAGES:
            st.write(f"`{code}` {label} — {script}")
        ok = run_pipeline(video_path, conf_thresh)
        status.update(label="Pipeline complete" if ok else "Pipeline failed", state="complete" if ok else "error")
    st.session_state.pipeline_ran = ok

# --------------------------------------------------------------------------
# Phase strip (signature element — mirrors the actual script sequence)
# --------------------------------------------------------------------------
active_idx = len(PIPELINE_STAGES) if st.session_state.pipeline_ran else (0 if run_clicked else -1)
phase_html = '<div class="phase-strip">'
for i, (code, label, _) in enumerate(PIPELINE_STAGES):
    dot_class = "done" if i < active_idx else ("active" if i == active_idx else "")
    phase_html += (
        f'<div class="phase-step">'
        f'<div class="phase-dot {dot_class}"></div>'
        f'<span class="phase-num">{code}</span>'
        f'<span class="phase-label">{label}</span>'
        f'</div>'
    )
    if i < len(PIPELINE_STAGES) - 1:
        phase_html += '<div class="phase-connector"></div>'
phase_html += "</div>"
st.markdown(phase_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
conflict_df = load_conflict_table() if st.session_state.pipeline_ran else None

if conflict_df is not None and not conflict_df.empty:
    n_conflicts = len(conflict_df)
    min_ttc = conflict_df["TTC"].min() if "TTC" in conflict_df else None
    avg_pet = conflict_df["PET"].mean() if "PET" in conflict_df else None
    n_vehicles = conflict_df["vehicle_id"].nunique() if "vehicle_id" in conflict_df else None

    st.markdown(
        f"""
        <div class="instrument-row">
            <div class="instrument">
                <div class="instrument-label">Conflicts detected</div>
                <div class="instrument-value alert">{n_conflicts}</div>
            </div>
            <div class="instrument">
                <div class="instrument-label">Min TTC (s)</div>
                <div class="instrument-value measure">{f'{min_ttc:.2f}' if min_ttc is not None else '—'}</div>
            </div>
            <div class="instrument">
                <div class="instrument-label">Avg PET (s)</div>
                <div class="instrument-value measure">{f'{avg_pet:.2f}' if avg_pet is not None else '—'}</div>
            </div>
            <div class="instrument">
                <div class="instrument-label">Vehicles tracked</div>
                <div class="instrument-value">{n_vehicles if n_vehicles is not None else '—'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_video, col_table = st.columns([3, 2])

    with col_video:
        st.markdown('<div class="signal-eyebrow">Annotated footage</div>', unsafe_allow_html=True)
        video_path = load_annotated_video()
        if video_path:
            st.video(str(video_path))
        else:
            st.markdown('<div class="empty-state">Annotated video not found in results_v3/</div>', unsafe_allow_html=True)

    with col_table:
        st.markdown('<div class="signal-eyebrow">Conflict log · ranked by severity</div>', unsafe_allow_html=True)
        sort_col = "TTC" if "TTC" in conflict_df.columns else conflict_df.columns[0]
        st.dataframe(
            conflict_df.sort_values(sort_col),
            use_container_width=True,
            height=420,
            hide_index=True,
        )

else:
    st.markdown(
        '<div class="empty-state">'
        'No results yet. Upload footage in the sidebar and run the analysis — '
        'the phase strip above will track progress through each pipeline stage.'
        '</div>',
        unsafe_allow_html=True,
    )
