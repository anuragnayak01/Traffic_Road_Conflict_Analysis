"""
SIGNAL — Traffic Conflict Analysis
------------------------------------
Streamlit front-end for the Traffic_Road_Conflict_Analysis pipeline
(01_data_extractor.py -> 07_conflict_annotator.py).

WIRING NOTES (three spots marked TODO — I don't have visibility into your
06_run_pipeline.py argument names or the exact results_v3 CSV columns):
  1. run_pipeline()        -> subprocess command + args
  2. load_conflict_table() -> path + column names inside results_v3/
  3. load_annotated_video()-> output video filename pattern
"""

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config + design tokens (light, explorable website)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SIGNAL — Traffic Conflict Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results_v3"

COLORS = {
    "bg": "#FFFFFF",
    "surface": "#F6F6F4",       # card fill
    "line": "#E3E3DF",          # hairlines
    "ink": "#14171A",           # primary text
    "muted": "#6B7076",         # secondary text
    "amber": "#F2A93B",         # fills (buttons, badges) — dark text on top
    "amber_ink": "#9C5E0B",     # amber used AS text/icon color (AA on white)
    "cyan": "#1D8A93",          # measured values
    "red": "#C43D2E",           # severe conflict
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
# CSS
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        html {{ scroll-behavior: smooth; }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background-color: {COLORS['bg']}; }}
        #MainMenu, header[data-testid="stHeader"] {{ background: transparent; }}

        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 3rem;
            max-width: 1080px;
        }}

        /* ---- top nav ---- */
        .site-nav {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 0;
            border-bottom: 1px solid {COLORS['line']};
            margin-bottom: 2.6rem;
        }}
        .site-nav .brand {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.05rem;
            color: {COLORS['ink']};
            letter-spacing: 0.02em;
        }}
        .site-nav .links a {{
            color: {COLORS['muted']};
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 500;
            margin-left: 1.8rem;
            transition: color 0.15s ease;
        }}
        .site-nav .links a:hover {{ color: {COLORS['ink']}; }}

        /* ---- hero ---- */
        .signal-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: {COLORS['amber_ink']};
            margin-bottom: 0.5rem;
        }}
        .signal-title {{
            font-size: 2.6rem;
            font-weight: 700;
            color: {COLORS['ink']};
            margin: 0 0 0.8rem 0;
            letter-spacing: -0.02em;
        }}
        .signal-sub {{
            color: {COLORS['muted']};
            font-size: 1.05rem;
            max-width: 640px;
            line-height: 1.55;
            margin-bottom: 2.4rem;
        }}
        .section-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {COLORS['amber_ink']};
            margin-bottom: 0.6rem;
        }}
        .section-heading {{
            font-size: 1.3rem;
            font-weight: 700;
            color: {COLORS['ink']};
            margin-bottom: 1.1rem;
        }}

        /* ---- control card (replaces sidebar) ---- */
        .control-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 10px;
            padding: 1.6rem 1.8rem 1.2rem 1.8rem;
            margin-bottom: 2.6rem;
        }}
        .control-card label, .control-card .stMarkdown p {{
            color: {COLORS['ink']} !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }}
        .stTextInput input {{
            background-color: {COLORS['bg']} !important;
            color: {COLORS['ink']} !important;
            border: 1px solid {COLORS['line']} !important;
            border-radius: 6px !important;
        }}
        .stTextInput input:focus {{
            border-color: {COLORS['amber_ink']} !important;
            box-shadow: 0 0 0 1px {COLORS['amber_ink']} !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {COLORS['bg']} !important;
            border: 1px dashed {COLORS['line']} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: transparent !important;
            color: {COLORS['cyan']} !important;
            border: 1px solid {COLORS['cyan']} !important;
            border-radius: 6px !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            background-color: rgba(29,138,147,0.08) !important;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
            background-color: {COLORS['amber']} !important;
            border-color: {COLORS['amber']} !important;
        }}
        [data-testid="stThumbValue"] {{
            color: {COLORS['amber_ink']} !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 700 !important;
        }}
        .stButton button {{
            background-color: {COLORS['amber']};
            color: #1A1200;
            font-weight: 700;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.4rem;
        }}
        .stButton button:hover {{ background-color: #E09A24; color: #1A1200; }}
        .stButton button:disabled {{
            background-color: rgba(242,169,59,0.15);
            color: rgba(156,94,11,0.5);
        }}

        /* ---- phase strip ---- */
        .phase-strip {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            row-gap: 0.9rem;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 10px;
            padding: 1.2rem 1.6rem;
            margin-bottom: 2.6rem;
        }}
        .phase-step {{ display: flex; align-items: center; gap: 0.5rem; white-space: nowrap; }}
        .phase-dot {{ width: 9px; height: 9px; border-radius: 50%; background: {COLORS['line']}; flex-shrink:0; }}
        .phase-dot.done {{ background: {COLORS['cyan']}; }}
        .phase-dot.active {{ background: {COLORS['amber']}; box-shadow: 0 0 0 3px rgba(242,169,59,0.25); }}
        .phase-num {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {COLORS['muted']}; }}
        .phase-label {{ font-size: 0.8rem; color: {COLORS['ink']}; font-weight: 500; }}
        .phase-connector {{ flex: 0 1 24px; min-width: 10px; height: 1px; background: {COLORS['line']}; margin: 0 0.7rem; }}

        /* ---- instruments ---- */
        .instrument-row {{
            display: flex; gap: 1px; background: {COLORS['line']};
            border: 1px solid {COLORS['line']}; border-radius: 10px;
            overflow: hidden; margin-bottom: 2.2rem;
        }}
        .instrument {{ flex: 1; background: {COLORS['bg']}; padding: 1.1rem 1.3rem; }}
        .instrument-label {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
            letter-spacing: 0.08em; text-transform: uppercase; color: {COLORS['muted']};
            margin-bottom: 0.4rem;
        }}
        .instrument-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700; color: {COLORS['ink']}; }}
        .instrument-value.alert {{ color: {COLORS['red']}; }}
        .instrument-value.measure {{ color: {COLORS['cyan']}; }}

        /* ---- empty state ---- */
        .empty-state {{
            border: 1px dashed {COLORS['line']}; border-radius: 10px; background: {COLORS['surface']};
            padding: 3.2rem 1.5rem; text-align: center; color: {COLORS['muted']}; font-size: 0.92rem;
            display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
        }}
        .empty-state-icon {{
            width: 10px; height: 32px; border-radius: 5px; background: {COLORS['bg']};
            border: 1px solid {COLORS['line']}; display: flex; flex-direction: column;
            align-items: center; justify-content: space-evenly; padding: 4px 0; margin-bottom: 0.4rem;
        }}
        .empty-state-icon span {{ width: 5px; height: 5px; border-radius: 50%; background: {COLORS['line']}; }}
        .empty-state-icon span:first-child {{ background: {COLORS['amber']}; }}

        footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Top nav
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="site-nav">
        <div class="brand">SIGNAL</div>
        <div class="links">
            <a href="#overview">Overview</a>
            <a href="#analyze">Analyze</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#results">Results</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Hero / overview
# --------------------------------------------------------------------------
st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
st.markdown('<div class="signal-eyebrow">Surrogate Safety Measures · Signalized Intersections</div>', unsafe_allow_html=True)
st.markdown('<div class="signal-title">Traffic conflict analysis, from raw footage to ranked risk.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signal-sub">Upload intersection footage. Vehicles are tracked, leader/follower pairs '
    'resolved, and conflicts scored by time-to-collision (TTC) and post-encroachment time (PET) — '
    'the same surrogate safety measures used in traffic engineering conflict studies.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Analyze — control card (replaces sidebar)
# --------------------------------------------------------------------------
st.markdown('<div id="analyze"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Analyze footage</div>', unsafe_allow_html=True)

st.markdown('<div class="control-card">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([2, 2.4, 1.6, 1.2])
with c1:
    site_name = st.text_input("Intersection / site label", placeholder="e.g. MG Road x College Rd")
with c2:
    uploaded_video = st.file_uploader("Video footage", type=["mp4", "avi", "mov"])
with c3:
    conf_thresh = st.slider("YOLOv8 confidence", 0.1, 0.9, 0.35, 0.05)
with c4:
    st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run analysis", disabled=uploaded_video is None, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pipeline runner
# --------------------------------------------------------------------------
def run_pipeline(video_path: Path, confidence: float) -> bool:
    """TODO: match args to 06_run_pipeline.py's real argparse flags."""
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
    """TODO: point at the real results_v3 CSV + column names."""
    candidate = RESULTS_DIR / "conflicts.csv"
    if not candidate.exists():
        return None
    return pd.read_csv(candidate)


def load_annotated_video() -> Path | None:
    """TODO: match 07_conflict_annotator.py's real output filename."""
    candidate = RESULTS_DIR / "annotated_output.mp4"
    return candidate if candidate.exists() else None


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
# Pipeline section
# --------------------------------------------------------------------------
st.markdown('<div id="pipeline"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Pipeline progress</div>', unsafe_allow_html=True)

active_idx = len(PIPELINE_STAGES) if st.session_state.pipeline_ran else (0 if run_clicked else -1)
phase_html = '<div class="phase-strip">'
for i, (code, label, _) in enumerate(PIPELINE_STAGES):
    dot_class = "done" if i < active_idx else ("active" if i == active_idx else "")
    phase_html += (
        f'<div class="phase-step"><div class="phase-dot {dot_class}"></div>'
        f'<span class="phase-num">{code}</span><span class="phase-label">{label}</span></div>'
    )
    if i < len(PIPELINE_STAGES) - 1:
        phase_html += '<div class="phase-connector"></div>'
phase_html += "</div>"
st.markdown(phase_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
st.markdown('<div id="results"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Step 3</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-label">Annotated footage</div>', unsafe_allow_html=True)
        video_path = load_annotated_video()
        if video_path:
            st.video(str(video_path))
        else:
            st.markdown('<div class="empty-state">Annotated video not found in results_v3/</div>', unsafe_allow_html=True)
    with col_table:
        st.markdown('<div class="section-label">Conflict log · ranked by severity</div>', unsafe_allow_html=True)
        sort_col = "TTC" if "TTC" in conflict_df.columns else conflict_df.columns[0]
        st.dataframe(conflict_df.sort_values(sort_col), use_container_width=True, height=420, hide_index=True)
else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-state-icon"><span></span><span></span><span></span></div>'
        f'<div style="color: {COLORS["ink"]}; font-weight: 700; font-size: 0.98rem;">Awaiting footage</div>'
        '<div>Upload a clip above and run the analysis — the pipeline section will '
        'track progress through each stage as it runs.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
