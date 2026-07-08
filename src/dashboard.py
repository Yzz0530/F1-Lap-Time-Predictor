"""
F1 Strategy Optimizer Dashboard — Streamlit UI.

Four-tab interface for strategy optimization, driver comparison,
stint simulation, and track analysis powered by XGBoost lap time predictions.
"""
from __future__ import annotations

import os
import time
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from strategy_optimizer import F1StrategyOptimizer

F1_RED = "#e10600"
COMPOUND_COLORS = {"SOFT": "#e10600", "MEDIUM": "#ffb800", "HARD": "#a0a0a0"}
TEAMS: dict[str, tuple[str, str]] = {
    "VER": ("#1e41ff", "Red Bull"), "HAD": ("#1e41ff", "Red Bull"),
    "LEC": ("#dc0000", "Ferrari"), "HAM": ("#dc0000", "Ferrari"),
    "RUS": ("#00d2be", "Mercedes"), "ANT": ("#00d2be", "Mercedes"),
    "NOR": ("#ff8000", "McLaren"), "PIA": ("#ff8000", "McLaren"),
    "ALO": ("#00665e", "Aston Martin"), "STR": ("#00665e", "Aston Martin"),
    "GAS": ("#fd7cac", "Alpine"), "COL": ("#fd7cac", "Alpine"),
    "OCO": ("#b6b6b6", "Haas"), "BEA": ("#b6b6b6", "Haas"),
    "LAW": ("#4b2db8", "RB"), "LIN": ("#4b2db8", "RB"),
    "ALB": ("#005aff", "Williams"), "SAI": ("#005aff", "Williams"),
    "HUL": ("#00e701", "Audi"), "BOR": ("#00e701", "Audi"),
    "PER": ("#898989", "Cadillac"), "BOT": ("#898989", "Cadillac"),
}
DRIVERS_LIST = sorted(TEAMS.keys())

st.set_page_config(page_title="F1 Strategy Optimizer", layout="wide")
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', -apple-system, sans-serif; }}
    .stApp {{ background: #0f0f0f; }}
    .main > div {{ padding: 1rem 2rem; }}
    h1 {{ font-weight: 800; letter-spacing: -0.5px; font-size: 1.8rem; }}
    h2, h3 {{ font-weight: 600; color: #eee; font-size: 1.1rem; margin: 1.2rem 0 0.8rem; }}
    .stSelectbox label, .stNumberInput label, .stSlider label {{ color: #888 !important; font-size: 0.75rem; font-weight: 500; }}
    .stSelectbox div[data-baseweb="select"] {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px; }}
    .stNumberInput input {{ background: #1a1a1a; border: 1px solid #2a2a2a; color: #eee; border-radius: 4px; }}
    div.stButton button {{ background: #e10600; color: white; font-weight: 700; border: none; border-radius: 4px; padding: 0.4rem 1rem; font-size: 0.8rem; letter-spacing: 0.5px; }}
    div.stButton button:hover {{ background: #b80500; }}
    div.stButton button:disabled {{ opacity: 0.4; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{ color: #666; font-weight: 600; font-size: 0.8rem; letter-spacing: 0.3px; padding: 0.5rem 1.5rem; border-bottom: 2px solid transparent; }}
    .stTabs [aria-selected="true"] {{ color: #eee !important; border-bottom: 2px solid {F1_RED} !important; background: transparent !important; }}
    .stAlert {{ background: #1a1a1a; border: 1px solid #333; border-radius: 4px; color: #ccc; }}
    .sr {{ border-bottom: 1px solid #222; padding: 0.75rem 0; display: flex; align-items: center; }}
    .sr:last-child {{ border-bottom: none; }}
    .sr-rank {{ color: #555; font-weight: 700; font-size: 0.9rem; width: 2rem; }}
    .sr-strat {{ color: #ccc; font-size: 0.85rem; flex: 1; }}
    .sr-time {{ color: #eee; font-weight: 600; font-size: 0.9rem; text-align: right; }}
    .sr-std {{ color: #666; font-size: 0.8rem; text-align: right; margin-left: 0.5rem; }}
    .card {{ background: #1a1a1a; border: 1px solid #222; border-radius: 6px; padding: 1rem; margin: 0.5rem 0; }}
    .team-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
    [data-testid="stSidebar"] {{ background: #111; border-right: 1px solid #1a1a1a; }}
    [data-testid="stMetric"] {{ background: transparent; border: none; padding: 0.5rem 0; }}
    [data-testid="stMetric"] div {{ color: #eee !important; font-size: 1.5rem !important; font-weight: 700 !important; }}
    [data-testid="stMetric"] label {{ color: #666 !important; font-size: 0.75rem !important; font-weight: 500 !important; }}
    hr {{ border-color: #222; margin: 1rem 0; }}
    .winner-badge {{ background: #1a1a1a; border: 1px solid #222; border-radius: 6px; padding: 0.75rem 1rem; text-align: center; font-size: 0.95rem; }}
    .driver-header {{ display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; background: #1a1a1a; border: 1px solid #222; border-radius: 6px; margin: 0.5rem 0; }}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────

@st.cache_resource
def load_optimizer() -> F1StrategyOptimizer:
    return F1StrategyOptimizer()

@st.cache_data(ttl=600, show_spinner=False)
def run_optimization(track: str, laps: int, driver: str, mc_runs: int,
                     sc_prob: float, dnf_prob: float = 0.05) -> list[dict[str, Any]]:
    return load_optimizer().optimize(track, laps, driver, mc_runs=mc_runs, sc_prob=sc_prob, dnf_prob=dnf_prob)

@st.cache_data(ttl=600, show_spinner=False)
def run_detailed(track: str, laps: int, driver: str,
                 strategy_tuple: tuple[tuple[str, int], ...]) -> dict[str, Any] | None:
    return load_optimizer().get_detailed_run(track, laps, driver, list(strategy_tuple))

# ── Init ──────────────────────────────────────────────────────────────

try:
    opt = load_optimizer()
    tracks = sorted(opt.circuit_info.keys())
except Exception as e:
    st.error(f"Failed to load optimizer: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"<div style='padding:0.5rem 0;'>"
        f"<span style='color:{F1_RED}; font-weight:800; font-size:1.3rem;'>F1</span> "
        f"<span style='color:#eee; font-weight:600; font-size:0.9rem;'>Strategy Optimizer</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#666; font-size:0.75rem; text-transform:uppercase; "
        "letter-spacing:0.5px;'>2026 Season</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:#aaa; font-size:0.85rem; margin:0.3rem 0;'>"
        f"<b style='color:#eee;'>{len(DRIVERS_LIST)}</b> drivers &nbsp;&nbsp; "
        f"<b style='color:#eee;'>{len(tracks)}</b> tracks</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#555; font-size:0.8rem;'>"
        "XGBoost lap time prediction &middot; Monte Carlo simulation &middot; "
        "2026 race data</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#444; font-size:0.75rem;'>"
        "fastf1 &middot; XGBoost &middot; Streamlit</span>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────

st.markdown(
    f"<h1><span style='color:#eee;'>Strategy </span>"
    f"<span style='color:{F1_RED};'>Optimizer</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666; font-size:0.85rem; margin-top:-0.8rem;'>"
    "Compare pit strategies & tyre degradation across the 2026 grid</p>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏁 STRATEGY", "⚔️ DRIVER BATTLE", "🔄 STINT SIM", "📊 TRACK ANALYSIS"]
)


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Strategy Optimization
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        driver = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"))
    with col2:
        track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"))
    with col3:
        laps = st.number_input("Race Laps", 10, 80, 52, step=1)
    with col4:
        mc_runs = st.number_input("Simulations", 10, 500, 30, step=10)
    col5, col6 = st.columns(2)
    with col5:
        sc_prob = st.slider("Safety Car Probability", 0.0, 0.5, 0.20, 0.05)
    with col6:
        dnf_prob = st.slider("DNF Probability", 0.0, 0.3, 0.05, 0.01, format="%.2f")

    if st.button("RUN OPTIMIZATION", type="primary", use_container_width=True):
        with st.spinner(f"Running {mc_runs} simulations across all strategies..."):
            t0 = time.time()
            results = run_optimization(track, laps, driver, mc_runs, sc_prob, dnf_prob)
            elapsed = time.time() - t0

        if not results:
            st.warning("No valid strategies found. Try a different track or driver.")
            st.stop()

        dc, tn = TEAMS.get(driver, ("#666", ""))
        st.markdown(
            f"<div class='driver-header'>"
            f"<span class='team-dot' style='background:{dc};'></span>"
            f"<div><b style='color:#eee;'>{driver}</b> &nbsp;"
            f"<span style='color:#888; font-size:0.8rem;'>{tn} &middot; {track} &middot; {laps} laps</span></div>"
            f"<div style='margin-left:auto; text-align:right;'>"
            f"<span style='color:#888; font-size:0.8rem;'>{elapsed:.1f}s &nbsp; "
            f"{len(results)} strategies</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<h3>Top Strategies</h3>", unsafe_allow_html=True)
        best_time = results[0]["mean_time"]
        for i, r in enumerate(results[:6]):
            strat_str = " → ".join([f"{c} ({l}l)" for c, l in r["strategy"]])
            diff_to_best = r["mean_time"] - best_time
            st.markdown(
                f"<div class='sr'>"
                f"<span class='sr-rank'>#{i + 1}</span>"
                f"<span class='sr-strat'>{strat_str}</span>"
                f"<span class='sr-time'>{r['mean_time']:.1f}s</span>"
                f"<span class='sr-std'>+{diff_to_best:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<h3>Race Time Comparison</h3>", unsafe_allow_html=True)
        labels = [" → ".join([f"{c[:3]}-{l}" for c, l in r["strategy"]]) for r in results[:8]]
        means = [r["mean_time"] / 60 for r in results[:8]]
        stds = [r["std_time"] / 60 for r in results[:8]]

        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("#0f0f0f")
        dc, _ = TEAMS.get(driver, (F1_RED, ""))
        ax.barh(range(len(labels))[::-1], means, xerr=stds,
                color=dc, capsize=2, height=0.5, edgecolor="none")
        ax.set_yticks(range(len(labels))[::-1])
        ax.set_yticklabels(labels, fontsize=8, color="#888")
        ax.set_xlabel("minutes", color="#555", fontsize=8)
        ax.tick_params(colors="#444")
        ax.margins(y=0.12)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#222")
        st.pyplot(fig, clear_figure=True)

        st.markdown("<h3>Stint Degradation</h3>", unsafe_allow_html=True)
        best_strat = results[0]["strategy"]
        run = run_detailed(track, laps, driver, tuple((c, l) for c, l in best_strat))
        if run and run.get("stint_details"):
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            fig2.patch.set_facecolor("none")
            ax2.set_facecolor("#0f0f0f")
            lap_global = 1
            for s in run["stint_details"]:
                xs = list(range(lap_global, lap_global + s["laps"]))
                c = COMPOUND_COLORS.get(s["compound"], "#fff")
                ax2.plot(xs, s["lap_times"], color=c, linewidth=1.5, marker=".", markersize=3)
                if len(s["lap_times"]) > 1:
                    z = np.polyfit(range(len(s["lap_times"])), s["lap_times"], 1)
                    ax2.plot(xs, np.poly1d(z)(range(len(s["lap_times"]))),
                             color=c, linestyle="--", alpha=0.3)
                lap_global += s["laps"]
            pit_x = []
            for idx in range(len(best_strat) - 1):
                pit_lap = sum(sl for _, sl in best_strat[:idx + 1])
                pit_x.append(pit_lap)
            for px in pit_x:
                ax2.axvline(x=px, color="#e10600", linestyle=":", alpha=0.3)
            if pit_x:
                ax2.annotate("PIT", (pit_x[0], ax2.get_ylim()[1] * 0.95),
                             color="#e10600", ha="center", fontsize=7)
            ax2.set_xlabel("Lap", color="#555", fontsize=8)
            ax2.set_ylabel("Lap Time (s)", color="#555", fontsize=8)
            ax2.tick_params(colors="#444")
            for sp in ["top", "right", "left"]:
                ax2.spines[sp].set_visible(False)
            ax2.spines["bottom"].set_color("#222")
            patches = [mpatches.Patch(color=c, label=l) for l, c in COMPOUND_COLORS.items()]
            ax2.legend(handles=patches, facecolor="#1a1a1a", labelcolor="#aaa",
                       fontsize=7, framealpha=0.9, edgecolor="#333")
            st.pyplot(fig2, clear_figure=True)
        elif run is None:
            st.info("Detailed stint data unavailable for this strategy.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Compare Drivers
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.selectbox("Driver 1", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="d1")
    with col2:
        d2 = st.selectbox("Driver 2", DRIVERS_LIST, index=DRIVERS_LIST.index("HAM"), key="d2")

    if d1 == d2:
        st.info("Select two different drivers to compare.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        track_c = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="track_c")
    with col2:
        laps_c = st.number_input("Laps", 10, 80, 52, key="laps_c")
    with col3:
        mc_c = st.number_input("Simulations", 10, 500, 30, step=10, key="mc_c")

    col_dnf = st.columns(1)[0]
    dnf_c = st.slider("DNF Probability", 0.0, 0.3, 0.05, 0.01, format="%.2f", key="dnf_c")

    if st.button("COMPARE", type="primary", use_container_width=True):
        with st.spinner("Running comparison..."):
            r1 = run_optimization(track_c, laps_c, d1, mc_c, 0.2, dnf_c)
            r2 = run_optimization(track_c, laps_c, d2, mc_c, 0.2, dnf_c)

        if not r1 or not r2:
            st.warning("Optimization returned no results for one or both drivers.")
            st.stop()

        b1, b2 = r1[0], r2[0]
        diff = b1["mean_time"] - b2["mean_time"]
        if diff < 0:
            winner, loser = d1, d2
        else:
            winner, loser = d2, d1

        c1, c2 = st.columns(2)
        for d, res, col in [(d1, b1, c1), (d2, b2, c2)]:
            strat = " → ".join([f"{c} ({l}l)" for c, l in res["strategy"]])
            mins, secs = divmod(int(res["mean_time"]), 60)
            dc, tn = TEAMS.get(d, ("#666", ""))
            with col:
                st.markdown(
                    f"<div class='card'>"
                    f"<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>"
                    f"<span class='team-dot' style='background:{dc};'></span>"
                    f"<b style='color:#eee;'>{d}</b> "
                    f"<span style='color:#666; font-size:0.8rem;'>{tn}</span>"
                    f"</div>"
                    f"<div style='color:#aaa; font-size:0.85rem;'>{strat}</div>"
                    f"<div style='margin-top:0.5rem;'>"
                    f"<span style='font-size:1.2rem; font-weight:700; color:#eee;'>"
                    f"{res['mean_time']:.1f}s</span>"
                    f"<span style='color:#666; font-size:0.8rem; margin-left:0.5rem;'>"
                    f"({mins}:{secs:02d}) &plusmn;{res['std_time']:.2f}s</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"<div class='winner-badge'>"
            f"<b style='color:#eee;'>{winner}</b> beats "
            f"<b style='color:#eee;'>{loser}</b> by "
            f"<b style='color:{F1_RED};'>{abs(diff):.1f}s</b></div>",
            unsafe_allow_html=True,
        )

        t1 = tuple((c, l) for c, l in b1["strategy"])
        t2 = tuple((c, l) for c, l in b2["strategy"])
        run1 = run_detailed(track_c, laps_c, d1, t1)
        run2 = run_detailed(track_c, laps_c, d2, t2)

        if run1 and run2:
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            fig3.patch.set_facecolor("none")
            ax3.set_facecolor("#0f0f0f")
            for d_name, run, style, marker in [(d1, run1, "-", "o"), (d2, run2, "--", "s")]:
                lg = 1
                for s in run["stint_details"]:
                    xs = list(range(lg, lg + s["laps"]))
                    c = COMPOUND_COLORS.get(s["compound"], "#fff")
                    ax3.plot(xs, s["lap_times"], color=c, linestyle=style, linewidth=1.5,
                             marker=marker, markersize=2, label=f"{d_name} {s['compound']}")
                    lg += s["laps"]
            ax3.set_xlabel("Lap", color="#555", fontsize=8)
            ax3.set_ylabel("Lap Time (s)", color="#555", fontsize=8)
            ax3.tick_params(colors="#444")
            for sp in ["top", "right", "left"]:
                ax3.spines[sp].set_visible(False)
            ax3.spines["bottom"].set_color("#222")
            ax3.legend(facecolor="#1a1a1a", labelcolor="#aaa", fontsize=7,
                       framealpha=0.9, edgecolor="#333")
            st.pyplot(fig3, clear_figure=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Stint Simulation
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sd = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="sd")
    with col2:
        sc = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
    with col3:
        st_laps = st.number_input("Stint Length", 5, 50, 20)
    with col4:
        track_s = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="track_s")

    stint_run = run_detailed(track_s, st_laps, sd, ((sc, st_laps),))
    if stint_run and stint_run.get("stint_details"):
        sd_data = stint_run["stint_details"][0]
        times = sd_data["lap_times"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Lap Time", f"{sd_data['avg_time']:.3f}s")
        c2.metric("Degradation", f"{times[-1] - times[0]:+.3f}s")
        c3.metric("First Lap", f"{times[0]:.3f}s")

        fig4, ax4 = plt.subplots(figsize=(10, 3))
        fig4.patch.set_facecolor("none")
        ax4.set_facecolor("#0f0f0f")
        tc = COMPOUND_COLORS.get(sc, F1_RED)
        laps_arr = list(range(1, st_laps + 1))
        ax4.plot(laps_arr, times, color=tc, linewidth=2, marker="o", markersize=3)
        if len(times) > 2:
            z = np.polyfit(laps_arr, times, 2)
            ax4.plot(laps_arr, np.poly1d(z)(laps_arr), linestyle="--", color="#555",
                     alpha=0.4, label="Trend")
        ax4.set_xlabel("Lap in Stint", color="#555", fontsize=8)
        ax4.set_ylabel("Lap Time (s)", color="#555", fontsize=8)
        ax4.tick_params(colors="#444")
        for sp in ["top", "right", "left"]:
            ax4.spines[sp].set_visible(False)
        ax4.spines["bottom"].set_color("#222")
        ax4.legend(facecolor="#1a1a1a", labelcolor="#aaa", fontsize=7,
                   framealpha=0.9, edgecolor="#333")
        st.pyplot(fig4, clear_figure=True)

        st.markdown("<h3>Lap Times</h3>", unsafe_allow_html=True)
        tbl = pd.DataFrame({
            "Lap": laps_arr,
            "Time": [f"{t:.3f}s" for t in times],
            "Delta": [f"{(t - times[0]):+.3f}s" for t in times],
        })
        st.dataframe(tbl, hide_index=True, width="stretch")
    else:
        st.info("Run a stint simulation to see lap-by-lap data.")


# ═════════════════════════════════════════════════════════════════════════
# TAB 4 — Track Analysis
# ═════════════════════════════════════════════════════════════════════════

with tab4:
    col1, col2, col3 = st.columns(3)
    with col1:
        ta_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="ta_track")
    with col2:
        ta_year = st.selectbox("Year", ["All", "2026", "2025"], key="ta_year")
    with col3:
        ta_compound = st.selectbox("Compound", ["All", "SOFT", "MEDIUM", "HARD"], key="ta_compound")

    # ── Load data ──
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_raw = pd.read_csv(os.path.join(_BASE, "data", "all_races_master.csv"))
    df_track = df_raw[df_raw["Race"] == ta_track].copy()
    if ta_year != "All":
        df_track = df_track[df_track["Year"] == int(ta_year)]
    if ta_compound != "All":
        df_track = df_track[df_track["Compound"] == ta_compound]

    if df_track.empty:
        st.info(f"No lap data available for the selected filters.")
        st.stop()

    # ── Data Overview ──
    st.markdown("<div class='section-label'>Data Overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Laps", len(df_track))
    c2.metric("Drivers", df_track["Driver"].nunique())
    c3.metric("Avg Lap", f"{df_track['LapTime'].mean():.3f}s")
    c4.metric("Median Lap", f"{df_track['LapTime'].median():.3f}s")
    yrs = sorted(df_track["Year"].unique())
    c5.metric("Year(s)", "+".join(str(int(y)) for y in yrs) if len(yrs) <= 2 else f"{int(min(yrs))}-{int(max(yrs))}")

    # ── Lap Time Distribution ──
    st.markdown("<div class='section-label'>Lap Time Distribution by Compound</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#0f0f0f")
    for cpd in ["SOFT", "MEDIUM", "HARD"]:
        sub = df_raw[(df_raw["Race"] == ta_track) & (df_raw["Compound"] == cpd)]
        if not sub.empty:
            ax.hist(sub["LapTime"], bins=40, alpha=0.45,
                    color=COMPOUND_COLORS.get(cpd, "#888"),
                    label=cpd, density=True)
    ax.set_xlabel("Lap Time (s)", color="#555", fontsize=8)
    ax.set_ylabel("Density", color="#555", fontsize=8)
    ax.tick_params(colors="#444")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#222")
    ax.legend(facecolor="#1a1a1a", labelcolor="#aaa", fontsize=7, framealpha=0.9, edgecolor="#333")
    st.pyplot(fig, clear_figure=True)

    # ── Tyre Degradation ──
    st.markdown("<div class='section-label'>Tyre Degradation by Compound</div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 3.2))
    fig2.patch.set_facecolor("none")
    ax2.set_facecolor("#0f0f0f")
    df_trk = df_raw[df_raw["Race"] == ta_track]
    for cpd in ["SOFT", "MEDIUM", "HARD"]:
        sub = df_trk[df_trk["Compound"] == cpd]
        if not sub.empty and sub["TyreLife"].nunique() > 2:
            deg = sub.groupby("TyreLife")["LapTime"].mean().reset_index()
            ax2.plot(deg["TyreLife"], deg["LapTime"],
                     color=COMPOUND_COLORS.get(cpd, "#888"),
                     linewidth=2, label=cpd, marker="o", markersize=3)
    ax2.set_xlabel("Tyre Life (laps)", color="#555", fontsize=8)
    ax2.set_ylabel("Avg Lap Time (s)", color="#555", fontsize=8)
    ax2.tick_params(colors="#444")
    for sp in ["top", "right", "left"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#222")
    ax2.legend(facecolor="#1a1a1a", labelcolor="#aaa", fontsize=7, framealpha=0.9, edgecolor="#333")
    st.pyplot(fig2, clear_figure=True)

    # ── Driver Ranking ──
    st.markdown(f"<div class='section-label'>Driver Ranking</div>", unsafe_allow_html=True)
    driver_avg = df_trk.groupby("Driver")["LapTime"].mean().sort_values()
    n = len(driver_avg)
    fig3, ax3 = plt.subplots(figsize=(10, max(3.2, n * 0.32)))
    fig3.patch.set_facecolor("none")
    ax3.set_facecolor("#0f0f0f")
    bar_colors = [TEAMS.get(d, ("#666", ""))[0] for d in driver_avg.index]
    ax3.barh(range(n), driver_avg.values, color=bar_colors, height=0.7)
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(driver_avg.index, color="#aaa", fontsize=8)
    ax3.set_xlabel("Avg Lap Time (s)", color="#555", fontsize=8)
    ax3.tick_params(colors="#444", labelsize=7)
    for sp in ["top", "right", "bottom"]:
        ax3.spines[sp].set_visible(False)
    ax3.spines["left"].set_color("#222")
    for i, v in enumerate(driver_avg.values):
        ax3.text(v + 0.02, i, f"{v:.3f}s", color="#777", fontsize=6.5, va="center")
    st.pyplot(fig3, clear_figure=True)

    # ── Year Comparison ──
    if df_trk["Year"].nunique() > 1:
        st.markdown("<div class='section-label'>Year-over-Year Comparison</div>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(10, 3.2))
        fig4.patch.set_facecolor("none")
        ax4.set_facecolor("#0f0f0f")
        for yr in sorted(df_trk["Year"].unique()):
            sub = df_trk[df_trk["Year"] == yr]
            d_avg = sub.groupby("Driver")["LapTime"].mean().reindex(driver_avg.index)
            ax4.plot(d_avg.values, range(len(d_avg)),
                     marker="o", markersize=4, linewidth=1.5, label=str(int(yr)))
        ax4.set_xlabel("Avg Lap Time (s)", color="#555", fontsize=8)
        ax4.set_ylabel("Driver (ranked)", color="#555", fontsize=8)
        ax4.tick_params(colors="#444")
        ax4.set_yticks(range(len(driver_avg)))
        ax4.set_yticklabels(driver_avg.index, color="#aaa", fontsize=7)
        for sp in ["top", "right", "left"]:
            ax4.spines[sp].set_visible(False)
        ax4.spines["bottom"].set_color("#222")
        ax4.legend(facecolor="#1a1a1a", labelcolor="#aaa", fontsize=7,
                   framealpha=0.9, edgecolor="#333")
        st.pyplot(fig4, clear_figure=True)

