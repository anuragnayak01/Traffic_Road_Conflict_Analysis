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

CHANGE NOTE: this revision only reshapes two sections — "Analyze footage"
(Step 1) and "Pipeline progress" (Step 2). All backend wiring, session
state, and the Results section are unchanged.
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

# code, short label, script, one-line description shown under the tracker
# while that stage is the active one
PIPELINE_STAGES = [
    ("01", "Extract",          "01_data_extractor.py",      "Pulling raw vehicle detections and frame timestamps from the footage."),
    ("02", "Smooth",           "02_trajectory_smoother.py", "Filtering jitter out of raw tracks to get clean per-vehicle paths."),
    ("03", "Leader/Follower",  "03_leader_follower.py",     "Pairing vehicles that share a lane or crossing path."),
    ("04", "SSM Calc",         "04_ssm_calculator.py",      "Computing time-to-collision and post-encroachment time per pair."),
    ("05", "Conflict Est.",    "05_conflict_estimator.py",  "Scoring and ranking pairs by conflict severity."),
    ("05b", "Signal Detect",   "05b_signal_detector.py",    "Reading signal phase to contextualize each conflict."),
    ("07", "Annotate",         "07_conflict_annotator.py",  "Rendering the annotated output video and final conflict log."),
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
            padding: 1.7rem 1.9rem 1.3rem 1.9rem;
            margin-bottom: 0.6rem;
        }}
        .control-card label, .control-card .stMarkdown p {{
            color: {COLORS['ink']} !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }}
        .field-hint {{
            font-size: 0.74rem;
            color: {COLORS['muted']};
            margin-top: 0.35rem;
            line-height: 1.4;
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
            width: 100%;
        }}
        .stButton button:hover {{ background-color: #E09A24; color: #1A1200; }}
        .stButton button:disabled {{
            background-color: rgba(242,169,59,0.15);
            color: rgba(156,94,11,0.5);
        }}
        .run-status {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: {COLORS['muted']};
            margin-top: 0.5rem;
            text-align: right;
        }}
        .card-footnote {{
            font-size: 0.76rem;
            color: {COLORS['muted']};
            padding: 0.9rem 0.1rem 0.1rem 0.1rem;
            border-top: 1px solid {COLORS['line']};
            margin-top: 1.3rem;
        }}

        /* ---- pipeline tracker ---- */
        .pipeline-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 10px;
            padding: 1.6rem 1.9rem 1.5rem 1.9rem;
            margin-bottom: 2.6rem;
        }}
        .pipeline-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }}
        .pipeline-count {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            color: {COLORS['muted']};
        }}
        .pipeline-track {{
            display: flex;
            align-items: flex-start;
        }}
        .pipeline-node {{
            flex: 1;
            position: relative;
            text-align: center;
            padding-top: 15px;
        }}
        .pipeline-node::before {{
            content: '';
            position: absolute;
            top: 15px;
            left: -50%;
            width: 100%;
            height: 2px;
            background: {COLORS['line']};
            z-index: 0;
        }}
        .pipeline-node:first-child::before {{ content: none; }}
        .pipeline-node.line-done::before {{ background: {COLORS['cyan']}; }}
        .pipeline-node.line-active::before {{
            background: linear-gradient(to right, {COLORS['cyan']}, {COLORS['amber']});
        }}
        .node-circle {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin: -30px auto 0.6rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            z-index: 1;
            background: {COLORS['bg']};
            border: 2px solid {COLORS['line']};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.66rem;
            font-weight: 700;
            color: {COLORS['muted']};
        }}
        .node-circle.done {{
            background: {COLORS['cyan']};
            border-color: {COLORS['cyan']};
            color: #FFFFFF;
        }}
        .node-circle.active {{
            background: {COLORS['amber']};
            border-color: {COLORS['amber']};
            color: #1A1200;
            animation: nodePulse 1.6s ease-in-out infinite;
        }}
        @keyframes nodePulse {{
            0%, 100% {{ box-shadow: 0 0 0 4px rgba(242,169,59,0.22); }}
            50% {{ box-shadow: 0 0 0 7px rgba(242,169,59,0.06); }}
        }}
        .node-label {{
            font-size: 0.72rem;
            font-weight: 500;
            color: {COLORS['ink']};
            line-height: 1.3;
        }}
        .node-label.pending {{ color: {COLORS['muted']}; }}
        .pipeline-caption {{
            margin-top: 1.4rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS['line']};
            font-size: 0.85rem;
            color: {COLORS['muted']};
        }}
        .pipeline-caption b {{ color: {COLORS['ink']}; font-weight: 600; }}
        .pipeline-caption .tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: {COLORS['amber_ink']};
            letter-spacing: 0.06em;
            margin-right: 0.5rem;
        }}

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
st.markdown('<div class="signal-title">Traffic conflict analysis, <br> from raw footage to ranked risk.</div>', unsafe_allow_html=True)
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
    st.markdown(
        '<div class="field-hint">Used to name the output files in results_v3/.</div>',
        unsafe_allow_html=True,
    )
with c2:
    uploaded_video = st.file_uploader("Video footage", type=["mp4", "avi", "mov"])
    st.markdown(
        '<div class="field-hint">MP4, AVI or MOV · fixed camera angle, full intersection in frame.</div>',
        unsafe_allow_html=True,
    )
with c3:
    conf_thresh = st.slider("YOLOv8 confidence", 0.1, 0.9, 0.35, 0.05)
    st.markdown(
        '<div class="field-hint">Lower catches more vehicles; higher cuts false detections.</div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run analysis", disabled=uploaded_video is None, use_container_width=True)
    st.markdown(
        f'<div class="run-status">{"Upload footage to enable" if uploaded_video is None else "Ready to run"}</div>',
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="card-footnote">Runs stages 01 → 07 in sequence on the server. '
    'A ~60s clip typically takes a few minutes — progress is tracked below.</div>',
    unsafe_allow_html=True,
)
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
        for code, label, script, _ in PIPELINE_STAGES:
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

# active_idx: index of the currently running stage, or len(stages) once all
# are done, or -1 before anything has been kicked off
n_stages = len(PIPELINE_STAGES)
if st.session_state.pipeline_ran:
    active_idx = n_stages
elif run_clicked:
    active_idx = 0
else:
    active_idx = -1

completed = min(max(active_idx, 0), n_stages) if active_idx >= 0 else 0

track_html = '<div class="pipeline-track">'
for i, (code, label, _, _desc) in enumerate(PIPELINE_STAGES):
    if i < active_idx:
        node_state, line_state = "done", "line-done"
    elif i == active_idx:
        node_state, line_state = "active", "line-active"
    else:
        node_state, line_state = "", ""
    label_class = "node-label" if node_state else "node-label pending"
    track_html += (
        f'<div class="pipeline-node {line_state}">'
        f'<div class="node-circle {node_state}">{code}</div>'
        f'<div class="{label_class}">{label}</div>'
        f'</div>'
    )
track_html += "</div>"

if active_idx == -1:
    caption = (
        '<span class="tag">STANDING BY</span>'
        'Upload footage and run the analysis above to start the pipeline.'
    )
elif active_idx >= n_stages:
    caption = '<span class="tag">DONE</span>All 7 stages completed — see results below.'
else:
    code, label, script, desc = PIPELINE_STAGES[active_idx]
    caption = f'<span class="tag">STAGE {code}</span><b>{label}</b> — {desc}'

st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="pipeline-header">
        <div></div>
        <div class="pipeline-count">{completed} / {n_stages} stages complete</div>
    </div>
    {track_html}
    <div class="pipeline-caption">{caption}</div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

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
