import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from strategy_optimizer import F1StrategyOptimizer
import time

rcParams["font.family"] = "sans-serif"
rcParams["font.size"] = 10

COMPOUND_COLORS = {"SOFT": "#e80020", "MEDIUM": "#ffb800", "HARD": "#a0a0a0"}
F1_RED = "#e80020"
F1_BG = "#0f0f0f"
F1_BG2 = "#1a1a1a"
F1_TEXT = "#f0f0f0"

st.set_page_config(page_title="F1 Strategy Optimizer", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {F1_BG}; color: {F1_TEXT}; }}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stSelectbox label, .stNumberInput label, .stSlider label {{ color: {F1_TEXT} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; background-color: {F1_BG2}; }}
    .stTabs [data-baseweb="tab"] {{ color: {F1_TEXT}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ background-color: {F1_RED} !important; color: white !important; }}
    div[data-testid="stMetric"] {{ background-color: {F1_BG2}; border: 1px solid #333; border-radius: 8px; padding: 12px; }}
    div[data-testid="stMetric"] label {{ color: {F1_TEXT} !important; }}
    div[data-testid="stMetric"] div {{ color: {F1_TEXT} !important; }}
    .stButton button {{ background-color: {F1_RED}; color: white; font-weight: 600; border: none; border-radius: 4px; }}
    .stButton button:hover {{ background-color: #cc001a; }}
    .card {{ background: {F1_BG2}; border-left: 4px solid {F1_RED}; padding: 12px; margin: 6px 0; border-radius: 4px; }}
    [data-testid="stSidebar"] {{ background-color: {F1_BG2}; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_optimizer():
    return F1StrategyOptimizer()

@st.cache_data(ttl=600)
def run_optimization(track, laps, driver, mc_runs, sc_prob):
    return load_optimizer().optimize(track, laps, driver, mc_runs=mc_runs, sc_prob=sc_prob)

@st.cache_data(ttl=600)
def run_detailed(track, laps, driver, strategy_tuple):
    return load_optimizer().get_detailed_run(track, laps, driver, list(strategy_tuple))

opt = load_optimizer()
tracks = sorted(opt.race_baselines.keys())
drivers = sorted(opt.driver_offsets.keys())

st.markdown(f"<h1 style='color: {F1_RED};'>F1 RACE STRATEGY OPTIMIZER</h1>", unsafe_allow_html=True)

DRIVER_TEAMS = {
    "VER":"Red Bull","PER":"Red Bull","LEC":"Ferrari","SAI":"Ferrari",
    "HAM":"Mercedes","RUS":"Mercedes","NOR":"McLaren","PIA":"McLaren",
    "ALO":"Aston Martin","STR":"Aston Martin","OCO":"Alpine","DOO":"Alpine",
    "TSU":"RB","HAD":"RB","ANT":"Mercedes","BEA":"Haas","BOR":"Sauber",
    "COL":"Williams","ALB":"Williams","LAW":"Red Bull","GAS":"Alpine",
    "HUL":"Sauber","BOT":"Sauber","LIN":"RB"
}
TEAM_COLORS = {
    "Red Bull":"#1e41ff","Mercedes":"#27f4d2","Ferrari":"#e80020","McLaren":"#ff8700",
    "Aston Martin":"#00594f","Alpine":"#0090ff","RB":"#4e28d2","Haas":"#b6b6b6",
    "Williams":"#00a0c6","Sauber":"#52e252"
}

def team_color(driver):
    team = DRIVER_TEAMS.get(driver, "unknown")
    return TEAM_COLORS.get(team, "#666")

tab1, tab2, tab3 = st.tabs(["STRATEGY OPTIMIZER", "DRIVER COMPARISON", "STINT SIMULATOR"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        driver = st.selectbox("Driver", drivers, index=drivers.index("VER"))
    with col2:
        track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"))
    with col3:
        laps = st.number_input("Total Laps", 10, 80, 52, step=1)
    with col4:
        mc_runs = st.number_input("Monte Carlo runs", 10, 500, 100, step=10)
    sc_prob = st.slider("Safety Car Probability", 0.0, 0.5, 0.20, 0.05)

    if st.button("OPTIMIZE STRATEGY", type="primary", use_container_width=True):
        with st.spinner(f"Running {mc_runs} simulations per strategy..."):
            t0 = time.time()
            results = run_optimization(track, laps, driver, mc_runs, sc_prob)
            elapsed = time.time() - t0

        st.markdown(f"<p style='color:#4ade80;'>Evaluated in {elapsed:.1f}s — {len(results)} strategies ranked</p>", unsafe_allow_html=True)

        st.markdown(f"<h3>⏱ Top Strategies</h3>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, r in enumerate(results[:6]):
            strat_str = " → ".join([f"{c} ({l}l)" for c, l in r["strategy"]])
            mins, secs = divmod(int(r["mean_time"]), 60)
            color = TEAM_COLORS.get(DRIVER_TEAMS.get(driver, ""), F1_RED) if i == 0 else "#555"
            with cols[i % 2]:
                st.markdown(
                    f"<div class='card' style='border-left-color:{color};'>"
                    f"<b>#{i+1}</b> {strat_str}<br>"
                    f"<span style='font-size:1.2em;color:{F1_RED};'><b>{r['mean_time']:.1f}s</b></span> "
                    f"({mins}:{secs:02d}) ±{r['std_time']:.2f}s<br>"
                    f"<span style='color:#888;'>{r['min_time']:.1f} – {r['max_time']:.1f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<h3>📊 Race Time Comparison</h3>", unsafe_allow_html=True)
        labels = [" → ".join([f"{c[:3]}-{l}" for c, l in r["strategy"]]) for r in results[:8]]
        means = [r["mean_time"] / 60 for r in results[:8]]
        stds = [r["std_time"] / 60 for r in results[:8]]

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(F1_BG)
        ax.set_facecolor(F1_BG2)
        bar_colors = [TEAM_COLORS.get(DRIVER_TEAMS.get(driver, ""), F1_RED)] + ["#444"] * (len(labels) - 1)
        ax.barh(range(len(labels)), means, xerr=stds, color=bar_colors, capsize=3, height=0.6, edgecolor="none")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9, color=F1_TEXT)
        ax.set_xlabel("Total Race Time (minutes)", color=F1_TEXT)
        ax.tick_params(colors=F1_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.markdown("<h3>📈 Stint Degradation</h3>", unsafe_allow_html=True)
        best_strat = results[0]["strategy"]
        run = run_detailed(track, laps, driver, tuple((c, l) for c, l in best_strat))
        if run:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            fig2.patch.set_facecolor(F1_BG)
            ax2.set_facecolor(F1_BG2)
            lap_global = 1
            for s in run["stint_details"]:
                xs = list(range(lap_global, lap_global + s["laps"]))
                c = COMPOUND_COLORS.get(s["compound"], "#fff")
                ax2.plot(xs, s["lap_times"], color=c, linewidth=2, marker=".", markersize=4)
                z = np.polyfit(range(len(s["lap_times"])), s["lap_times"], 1)
                ax2.plot(xs, np.poly1d(z)(range(len(s["lap_times"]))), color=c, linestyle="--", alpha=0.5)
                lap_global += s["laps"]
            pit_x = []
            for idx, (_, sl) in enumerate(best_strat[:-1]):
                pit_lap = sum(sl for _, sl in best_strat[:idx+1])
                pit_x.append(pit_lap)
            for px in pit_x:
                ax2.axvline(x=px, color="#ff4444", linestyle=":", alpha=0.7)
            if pit_x:
                ax2.annotate("PIT", (pit_x[0], ax2.get_ylim()[1]*0.95), color="#ff4444", ha="center", fontsize=9)
            ax2.set_xlabel("Race Lap", color=F1_TEXT)
            ax2.set_ylabel("Lap Time (s)", color=F1_TEXT)
            ax2.tick_params(colors=F1_TEXT)
            for spine in ax2.spines.values():
                spine.set_color("#333")
            ax2.grid(True, alpha=0.15)
            patches = [mpatches.Patch(color=c, label=l) for l, c in COMPOUND_COLORS.items()]
            ax2.legend(handles=patches, facecolor=F1_BG2, labelcolor=F1_TEXT, framealpha=0.8)
            st.pyplot(fig2)

with tab2:
    st.markdown("<h3>⚔️ Head-to-Head</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.selectbox("Driver 1", drivers, index=drivers.index("VER"), key="d1")
    with col2:
        d2 = st.selectbox("Driver 2", drivers, index=drivers.index("HAM"), key="d2")

    col1, col2, col3 = st.columns(3)
    with col1:
        track_c = st.selectbox("Track", tracks, key="track_c")
    with col2:
        laps_c = st.number_input("Laps", 10, 80, 52, key="laps_c")
    with col3:
        mc_c = st.number_input("MC Runs", 10, 500, 100, step=10, key="mc_c")

    if st.button("COMPARE", type="primary", use_container_width=True, key="compare_btn"):
        with st.spinner("Running simulations..."):
            r1 = run_optimization(track_c, laps_c, d1, mc_c, 0.2)
            r2 = run_optimization(track_c, laps_c, d2, mc_c, 0.2)

        b1, b2 = r1[0], r2[0]
        diff = b1["mean_time"] - b2["mean_time"]
        winner, loser = (d1, d2) if diff < 0 else (d2, d1)

        col1, col2 = st.columns(2)
        for d, res, col, icon in [(d1, b1, col1, "🏎️"), (d2, b2, col2, "🏎️")]:
            strat = " → ".join([f"{c} ({l}l)" for c, l in res["strategy"]])
            mins, secs = divmod(int(res["mean_time"]), 60)
            tc = team_color(d)
            with col:
                st.markdown(
                    f"<div class='card' style='border-left-color:{tc};'>"
                    f"<span style='font-size:1.3em;'>{icon}</span> <b>{d}</b><br>"
                    f"{strat}<br>"
                    f"<span style='font-size:1.1em;'>{res['mean_time']:.1f}s</span> ({mins}:{secs:02d}) ±{res['std_time']:.2f}s"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(f"<h3 style='color: #4ade80;'>✅ {winner} beats {loser} by {abs(diff):.1f}s</h3>", unsafe_allow_html=True)

        t1 = tuple((c, l) for c, l in b1["strategy"])
        t2 = tuple((c, l) for c, l in b2["strategy"])
        run1 = run_detailed(track_c, laps_c, d1, t1)
        run2 = run_detailed(track_c, laps_c, d2, t2)

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        fig3.patch.set_facecolor(F1_BG)
        ax3.set_facecolor(F1_BG2)
        for d_name, run, style, marker in [(d1, run1, "-", "o"), (d2, run2, "--", "s")]:
            lg = 1
            for s in run["stint_details"]:
                xs = list(range(lg, lg + s["laps"]))
                c = COMPOUND_COLORS.get(s["compound"], "#fff")
                ax3.plot(xs, s["lap_times"], color=c, linestyle=style, linewidth=1.5,
                         marker=marker, markersize=2, label=f"{d_name} {s['compound']}")
                lg += s["laps"]
        ax3.set_xlabel("Race Lap", color=F1_TEXT)
        ax3.set_ylabel("Lap Time (s)", color=F1_TEXT)
        ax3.tick_params(colors=F1_TEXT)
        for spine in ax3.spines.values():
            spine.set_color("#333")
        ax3.legend(facecolor=F1_BG2, labelcolor=F1_TEXT, fontsize=8, framealpha=0.8)
        ax3.grid(True, alpha=0.15)
        st.pyplot(fig3)

with tab3:
    st.markdown("<h3>🔬 Single Stint Simulation</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sd = st.selectbox("Driver", drivers, index=drivers.index("VER"), key="sd")
    with col2:
        sc = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
    with col3:
        st_laps = st.number_input("Stint Length", 5, 50, 20)
    with col4:
        track_s = st.selectbox("Track", tracks, key="track_s")

    stint_run = run_detailed(track_s, st_laps, sd, ((sc, st_laps),))
    if stint_run:
        sd_data = stint_run["stint_details"][0]
        laps_arr = list(range(1, st_laps + 1))
        times = sd_data["lap_times"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Lap Time", f"{sd_data['avg_time']:.3f}s")
        col2.metric("Degradation", f"{times[-1] - times[0]:+.3f}s", delta_color="inverse")
        col3.metric("First Lap", f"{times[0]:.3f}s")

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        fig4.patch.set_facecolor(F1_BG)
        ax4.set_facecolor(F1_BG2)
        tc = COMPOUND_COLORS.get(sc, F1_RED)
        ax4.plot(laps_arr, times, color=tc, linewidth=2.5, marker="o", markersize=4)
        z = np.polyfit(laps_arr, times, 2)
        ax4.plot(laps_arr, np.poly1d(z)(laps_arr), linestyle="--", color="#888", alpha=0.6, label="Trend")
        ax4.set_xlabel("Lap in Stint", color=F1_TEXT)
        ax4.set_ylabel("Lap Time (s)", color=F1_TEXT)
        ax4.tick_params(colors=F1_TEXT)
        for spine in ax4.spines.values():
            spine.set_color("#333")
        ax4.legend(facecolor=F1_BG2, labelcolor=F1_TEXT, framealpha=0.8)
        ax4.grid(True, alpha=0.15)
        st.pyplot(fig4)

        st.markdown("<h3>📋 Lap-by-Lap Breakdown</h3>", unsafe_allow_html=True)
        tbl = pd.DataFrame({
            "Lap": laps_arr,
            "Lap Time": [f"{t:.3f}s" for t in times],
            "Delta": [f"{(t - times[0]):+.3f}s" for t in times]
        })
        st.dataframe(tbl, hide_index=True, width="stretch")

st.sidebar.markdown(f"<h2 style='color:{F1_RED};'>F1 OPTIMIZER</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**2026 Season**")
st.sidebar.markdown(f"Drivers: **{len(drivers)}**")
st.sidebar.markdown(f"Tracks: **{len(tracks)}**")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with fastf1 + XGBoost")
st.sidebar.markdown("Physics-based Monte Carlo optimizer")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<span style='color:#888;'>Select a driver in the tab above to begin</span>", unsafe_allow_html=True)
