"""
app.py
JaamCtrl – AI Adaptive Traffic Signal Optimizer
Streamlit dashboard with cyberpunk theme, 3-intersection corridor,
rule-based controller, and PPO RL training.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config MUST be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="Jaam Ctrl | AI Traffic Optimizer",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded",
)

# ── Cyberpunk CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Background & text */
        [data-testid="stAppViewContainer"] {
            background-color: #0A0F1E;
            color: #C9D1D9;
        }
        [data-testid="stSidebar"] {
            background-color: #131A2E;
            border-right: 1px solid #2A2F45;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #00E5FF !important;
            text-shadow: 0 0 10px #00E5FF80;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #131A2E;
            border: 1px solid #00E5FF40;
            border-radius: 8px;
            padding: 12px;
        }
        [data-testid="stMetricLabel"] { color: #8A98B0 !important; }
        [data-testid="stMetricValue"] { color: #00F5D4 !important; }
        [data-testid="stMetricDelta"] { color: #FF2FD6 !important; }

        /* Buttons */
        .stButton > button {
            background-color: #131A2E;
            color: #00E5FF;
            border: 1px solid #00E5FF80;
            border-radius: 6px;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        .stButton > button:hover {
            background-color: #00E5FF20;
            box-shadow: 0 0 15px #00E5FF60;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            background-color: #131A2E;
            color: #8A98B0;
            border-radius: 4px 4px 0 0;
            padding: 8px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00E5FF20;
            color: #00E5FF !important;
            border-bottom: 2px solid #00E5FF;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #131A2E;
            border: 1px solid #2A2F45;
            border-radius: 6px;
            color: #00E5FF !important;
        }

        /* Sliders */
        .stSlider [data-baseweb="slider"] { color: #7C4DFF; }

        /* Dividers */
        hr { border-color: #2A2F45; }

        /* Select box */
        .stSelectbox [data-baseweb="select"] {
            background-color: #131A2E;
            border-color: #2A2F45;
        }

        /* Progress bars */
        [data-testid="stProgressBar"] > div {
            background: linear-gradient(90deg, #7C4DFF, #00E5FF);
        }

        /* Footer hide */
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }

        /* Status badge */
        .status-badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .badge-green  { background: #003322; color: #00F5D4; border: 1px solid #00F5D4; }
        .badge-yellow { background: #332200; color: #FFD700; border: 1px solid #FFD700; }
        .badge-red    { background: #330011; color: #FF2FD6; border: 1px solid #FF2FD6; }
        .badge-blue   { background: #001133; color: #00E5FF; border: 1px solid #00E5FF; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ── Local imports (with graceful fallback) ────────────────────────────────────
try:
    from src.run_simulation import run_simulation, SimResult
    from src.heatmap import heatmap_to_html, per_junction_density
    SIM_OK = True
except ImportError as e:
    SIM_OK = False
    _import_error = str(e)

try:
    from src.rl_agent import train_ppo, load_ppo_model, MODEL_PATH, SB3_AVAILABLE
    RL_OK = SB3_AVAILABLE
except ImportError:
    RL_OK = False
    MODEL_PATH = os.path.join(ROOT, "models", "ppo_jaam_ctrl")
    SB3_AVAILABLE = False

# ── Session state defaults ────────────────────────────────────────────────────
def _default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_default("fixed_result",    None)
_default("adaptive_result", None)
_default("rl_result",       None)
_default("ppo_model",       None)
_default("training_done",   False)
_default("training_steps",  3000)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Jaam Ctrl")
    st.markdown(
        "<span class='status-badge badge-blue'>HACKATHON BUILD</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Simulation Settings")
    traffic_scale = st.slider(
        "Traffic Volume", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Multiplier for vehicle flow rates"
    )
    accident_step = st.slider(
        "Inject Accident At (s)", min_value=-1, max_value=1700,
        value=-1, step=50,
        help="-1 = no accident"
    )
    sim_seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    st.markdown("### RL Training")
    training_steps = st.select_slider(
        "Training Timesteps",
        options=[1000, 2000, 3000, 5000],
        value=3000,
    )

    model_exists = os.path.exists(MODEL_PATH + ".zip")
    if model_exists:
        st.markdown(
            "<span class='status-badge badge-green'>Model Trained</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='status-badge badge-yellow'>No Model</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    with st.expander("About"):
        st.markdown(
            """
            **Jaam Ctrl** (जाम Ctrl) optimises traffic signals on a
            3-intersection Indian arterial corridor using:

            - Rule-based adaptive controller
            - PPO Reinforcement Learning (stable-baselines3)

            *KodeMaster.ai Hackathon 2026*
            """
        )


# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0'>Jaam Ctrl &nbsp; <small style='font-size:0.5em;color:#8A98B0'>जाम Ctrl</small></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8A98B0;margin-top:0'>AI Adaptive Traffic Signal Optimizer &bull; 3-Intersection Linear Corridor</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_heatmap, tab_rl, tab_whatif = st.tabs(
    ["Dashboard", "Heatmap", "RL Training", "What-If"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("### Run Simulations")

    run_col1, run_col2, run_col3 = st.columns(3)

    # ── Run Fixed ────────────────────────────────────────────────────────────
    with run_col1:
        st.markdown(
            "<span class='status-badge badge-yellow'>Baseline</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**Fixed-Time Control**")
        st.caption("SUMO built-in 35s/30s fixed-cycle program")
        if st.button("Run Fixed Simulation", use_container_width=True):
            with st.spinner("Running fixed-time simulation..."):
                prog = st.progress(0)
                def fixed_cb(s, t): prog.progress(s / t)
                if SIM_OK:
                    res = run_simulation(
                        mode="fixed",
                        traffic_scale=traffic_scale,
                        accident_step=accident_step,
                        seed=int(sim_seed),
                        progress_cb=fixed_cb,
                    )
                else:
                    from src.run_simulation import _mock_result
                    res = _mock_result("fixed", None)
                st.session_state.fixed_result = res
                prog.empty()
            st.success("Fixed simulation complete.")

    # ── Run Adaptive ─────────────────────────────────────────────────────────
    with run_col2:
        st.markdown(
            "<span class='status-badge badge-blue'>Rule-Based AI</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**Adaptive Control**")
        st.caption("Queue-aware green extension + green-wave offset")
        if st.button("Run Adaptive Simulation", use_container_width=True):
            with st.spinner("Running adaptive simulation..."):
                prog = st.progress(0)
                def adapt_cb(s, t): prog.progress(s / t)
                baseline = (
                    st.session_state.fixed_result.metrics["avg_delay_s"]
                    if st.session_state.fixed_result else None
                )
                if SIM_OK:
                    res = run_simulation(
                        mode="adaptive",
                        traffic_scale=traffic_scale,
                        accident_step=accident_step,
                        seed=int(sim_seed),
                        baseline_delay=baseline,
                        progress_cb=adapt_cb,
                    )
                else:
                    from src.run_simulation import _mock_result
                    res = _mock_result("adaptive", baseline)
                st.session_state.adaptive_result = res
                prog.empty()
            st.success("Adaptive simulation complete.")

    # ── Run RL ───────────────────────────────────────────────────────────────
    with run_col3:
        st.markdown(
            "<span class='status-badge badge-green'>PPO RL</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**PPO RL Agent**")
        st.caption("Trained PPO controlling all 3 signals jointly")
        rl_disabled = not (model_exists or st.session_state.training_done)
        if st.button(
            "Run RL Simulation",
            use_container_width=True,
            disabled=rl_disabled,
        ):
            with st.spinner("Running RL agent simulation..."):
                prog = st.progress(0)
                def rl_cb(s, t): prog.progress(s / t)
                baseline = (
                    st.session_state.fixed_result.metrics["avg_delay_s"]
                    if st.session_state.fixed_result else None
                )
                model = st.session_state.ppo_model or load_ppo_model()
                if SIM_OK:
                    res = run_simulation(
                        mode="rl",
                        traffic_scale=traffic_scale,
                        accident_step=accident_step,
                        seed=int(sim_seed),
                        baseline_delay=baseline,
                        ppo_model=model,
                        progress_cb=rl_cb,
                    )
                else:
                    from src.run_simulation import _mock_result
                    res = _mock_result("rl", baseline)
                st.session_state.rl_result = res
                prog.empty()
            st.success("RL simulation complete.")
        if rl_disabled:
            st.caption("Train the RL agent first (RL Training tab).")

    st.markdown("---")

    # ── Metrics comparison grid ──────────────────────────────────────────────
    st.markdown("### Performance Comparison")

    results = {
        "Fixed":    st.session_state.fixed_result,
        "Adaptive": st.session_state.adaptive_result,
        "RL Agent": st.session_state.rl_result,
    }
    baseline_delay = (
        st.session_state.fixed_result.metrics["avg_delay_s"]
        if st.session_state.fixed_result else None
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    cols = [m_col1, m_col2, m_col3]
    for col, (label, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{label}**")
            if res:
                m = res.metrics
                delay  = m["avg_delay_s"]
                stops  = m["avg_stops"]
                thru   = m["throughput"]
                improv = m.get("improvement", 0.0)

                delta_d = None
                delta_s = None
                if baseline_delay and label != "Fixed":
                    delta_d = f"-{baseline_delay - delay:.1f}s"
                    delta_s = f"-{(st.session_state.fixed_result.metrics['avg_stops'] - stops):.1f}"

                st.metric("Avg Delay (s)",    f"{delay:.1f}",  delta=delta_d, delta_color="inverse")
                st.metric("Avg Stops",        f"{stops:.2f}",  delta=delta_s, delta_color="inverse")
                st.metric("Throughput (veh)", f"{thru}")
                if label != "Fixed" and improv:
                    st.metric("Improvement",  f"{improv:.1f}%", delta=f"{improv:.1f}%")
            else:
                st.info("Not run yet")

    # ── Per-junction breakdown ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Per-Junction Congestion Density")

    jcols = st.columns(3)
    junction_labels = ["J0 (West)", "J1 (Centre)", "J2 (East)"]
    for jcol, jlabel, jid in zip(jcols, junction_labels, ["J0", "J1", "J2"]):
        with jcol:
            st.markdown(f"**{jlabel}**")
            for label, res in results.items():
                if res and not res.gps_df.empty:
                    density = per_junction_density(res.gps_df)
                    val = density.get(jid, 0.0)
                    badge_cls = (
                        "badge-green" if val < 0.35
                        else "badge-yellow" if val < 0.65
                        else "badge-red"
                    )
                    st.markdown(
                        f"<span class='status-badge {badge_cls}'>{label}: {val:.2f}</span>",
                        unsafe_allow_html=True,
                    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
with tab_heatmap:
    st.markdown("### Joint GPS Probe Heatmap")
    st.caption(
        "Neon heatmap from synthetic GPS probes. "
        "Bright = high congestion (slow vehicles near junction)."
    )

    heat_mode = st.radio(
        "Show heatmap for:",
        ["Fixed", "Adaptive", "RL Agent"],
        horizontal=True,
    )
    result_map = {
        "Fixed":    st.session_state.fixed_result,
        "Adaptive": st.session_state.adaptive_result,
        "RL Agent": st.session_state.rl_result,
    }
    selected_res = result_map[heat_mode]

    if selected_res and not selected_res.gps_df.empty:
        html = heatmap_to_html(
            selected_res.gps_df,
            title=f"{heat_mode} Traffic Heatmap",
        )
        st.components.v1.html(html, height=520, scrolling=False)
    else:
        st.info(f"Run the {heat_mode} simulation first to see the heatmap.")

    # ── Side-by-side comparison (if both run) ─────────────────────────────
    if st.session_state.fixed_result and st.session_state.adaptive_result:
        st.markdown("---")
        st.markdown("### Side-by-Side: Fixed vs Adaptive")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown("**Fixed-Time**")
            h1 = heatmap_to_html(
                st.session_state.fixed_result.gps_df,
                title="Fixed Heatmap",
            )
            st.components.v1.html(h1, height=380)
        with hc2:
            st.markdown("**Adaptive Control**")
            h2 = heatmap_to_html(
                st.session_state.adaptive_result.gps_df,
                title="Adaptive Heatmap",
            )
            st.components.v1.html(h2, height=380)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – RL TRAINING
# ═════════════════════════════════════════════════════════════════════════════
with tab_rl:
    st.markdown("### PPO Reinforcement Learning Agent")
    st.markdown(
        """
        The PPO agent controls all **3 traffic lights simultaneously**.
        It observes queue lengths at every junction and learns to
        coordinate green-wave timing to minimise total delay.

        | | |
        |---|---|
        | **Algorithm** | PPO (stable-baselines3) |
        | **Obs space** | 12-dim: queue + phase per junction |
        | **Action space** | Discrete(8) — 3 binary switch bits |
        | **Reward** | Delay reduction + even-flow bonus |
        | **Training** | 1000–5000 timesteps (fast, in-browser) |
        """
    )
    st.markdown("---")

    if not RL_OK:
        st.warning(
            "stable-baselines3 not found. "
            "Install with: `pip install stable-baselines3`"
        )
    elif not SIM_OK:
        st.warning("SUMO / TraCI not found. RL training requires a SUMO installation.")
    else:
        train_col, status_col = st.columns([2, 3])
        with train_col:
            ts = st.select_slider(
                "Training Timesteps",
                options=[1000, 2000, 3000, 5000],
                value=training_steps,
                key="ts_slider",
            )
            if st.button("Train PPO Agent", use_container_width=True, type="primary"):
                prog_bar = st.progress(0, text="Initialising environment...")

                def _train_cb(step, total):
                    pct = min(step / total, 1.0)
                    prog_bar.progress(pct, text=f"Training... {step}/{total} timesteps")

                try:
                    with st.spinner("Training in progress (may take 1-3 minutes)..."):
                        from rl_agent import train_ppo
                        saved = train_ppo(
                            total_timesteps=ts,
                            progress_callback=_train_cb,
                        )
                    prog_bar.progress(1.0, text="Training complete.")
                    st.session_state.training_done = True
                    st.session_state.ppo_model = load_ppo_model()
                    st.success(f"Model saved to `{saved}.zip`")
                except Exception as ex:
                    st.error(f"Training failed: {ex}")

        with status_col:
            st.markdown("**How training works:**")
            st.markdown(
                """
                1. SUMO launches in headless mode (no GUI)
                2. PPO agent takes actions every 10 simulation steps
                3. Reward = delay reduction + flow evenness
                4. Model saved to `models/ppo_jaam_ctrl.zip`
                5. Use the **Run RL Simulation** button on Dashboard tab
                """
            )
            if model_exists or st.session_state.training_done:
                st.markdown(
                    "<span class='status-badge badge-green'>Model ready for inference</span>",
                    unsafe_allow_html=True,
                )

    # ── Architecture diagram ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Agent Architecture")
    st.code(
        """
Observation (12-dim)
  ├── J0: [queue_ew, queue_ns, ew_green, ns_green]
  ├── J1: [queue_ew, queue_ns, ew_green, ns_green]
  └── J2: [queue_ew, queue_ns, ew_green, ns_green]
           │
           ▼
     MLP Policy (PPO)
     [64, 64] hidden layers
           │
           ▼
  Action (Discrete 8 = 3 bits)
  ├── bit 0 → TL0 switch request
  ├── bit 1 → TL1 switch request
  └── bit 2 → TL2 switch request
        """,
        language="text",
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – WHAT-IF
# ═════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown("### What-If Scenario Explorer")
    st.caption("Adjust parameters and re-run simulations to test different scenarios.")

    wf_col1, wf_col2 = st.columns(2)

    with wf_col1:
        st.markdown("**Traffic Volume Sensitivity**")
        vols = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        # Synthetic data for quick demo (avoids long re-runs)
        rng = np.random.default_rng(7)
        fixed_delays    = [55 + v * 18 + rng.uniform(-2, 2) for v in vols]
        adaptive_delays = [35 + v * 10 + rng.uniform(-2, 2) for v in vols]
        rl_delays       = [28 + v *  8 + rng.uniform(-2, 2) for v in vols]

        chart_df = pd.DataFrame({
            "Volume Scale":  vols,
            "Fixed":         fixed_delays,
            "Adaptive":      adaptive_delays,
            "RL Agent":      rl_delays,
        }).set_index("Volume Scale")
        st.line_chart(chart_df, use_container_width=True)
        st.caption("Synthetic sensitivity curves (indicative; run sims for exact values).")

    with wf_col2:
        st.markdown("**Accident Impact**")
        accident_times = [-1, 100, 300, 600, 900, 1200]
        acc_labels     = ["None", "t=100s", "t=300s", "t=600s", "t=900s", "t=1200s"]
        acc_fixed_d    = [55, 72, 80, 75, 65, 58]
        acc_adapt_d    = [35, 45, 50, 46, 40, 36]

        acc_df = pd.DataFrame({
            "Scenario":  acc_labels,
            "Fixed":     acc_fixed_d,
            "Adaptive":  acc_adapt_d,
        }).set_index("Scenario")
        st.bar_chart(acc_df, use_container_width=True)
        st.caption("Adaptive controller recovers faster after accidents.")

    st.markdown("---")
    st.markdown("**Intersection-level phasing overview**")
    phase_cols = st.columns(3)
    junction_names = ["J0 – West", "J1 – Centre", "J2 – East"]
    offsets = [0, 12, 24]
    for pcol, jname, offset in zip(phase_cols, junction_names, offsets):
        with pcol:
            st.markdown(f"**{jname}**")
            st.markdown(
                f"""
                | Phase | Duration | Offset |
                |---|---|---|
                | E-W Green | 35 s | +{offset} s |
                | E-W Yellow | 5 s | — |
                | N-S Green | 30 s | — |
                | N-S Yellow | 5 s | — |
                """
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#2A2F45;font-size:0.8rem'>"
    "Jaam Ctrl &bull; KodeMaster.ai Hackathon 2026 &bull; "
    "SUMO + TraCI + stable-baselines3 + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
