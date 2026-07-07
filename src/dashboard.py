import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from strategy_optimizer import F1StrategyOptimizer
import time

st.set_page_config(page_title="F1 Strategy Optimizer", layout="wide")
st.title("F1 Race Strategy Optimizer")

@st.cache_resource
def load_optimizer():
    return F1StrategyOptimizer()

opt = load_optimizer()

df = pd.read_csv("../data/all_races_clean.csv")
tracks = sorted(df["Race"].unique())
drivers = sorted(opt.driver_offsets.keys())

tab1, tab2, tab3 = st.tabs(["Strategy Optimizer", "Driver Comparison", "Stint Simulator"])

# ---- TAB 1: Strategy Optimizer ----
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        driver = st.selectbox("Driver", drivers, index=drivers.index("VER"))
    with col2:
        track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"))
    with col3:
        laps = st.number_input("Total Laps", min_value=10, max_value=80, value=52, step=1)
    with col4:
        mc_runs = st.number_input("Monte Carlo runs", min_value=10, max_value=500, value=100, step=10)

    sc_prob = st.slider("Safety Car Probability", 0.0, 0.5, 0.20, 0.05)

    if st.button("Optimize Strategy", type="primary"):
        with st.spinner(f"Running {mc_runs} simulations per strategy..."):
            t0 = time.time()
            results = opt.optimize(track, laps, driver, mc_runs=mc_runs, sc_prob=sc_prob)
            elapsed = time.time() - t0

        st.success(f"Evaluated in {elapsed:.1f}s — top {len(results)} strategies")

        st.subheader("Top Strategies")
        cols = st.columns(2)
        for i, r in enumerate(results[:6]):
            strat_str = " -> ".join([f"{c} ({l}l)" for c, l in r["strategy"]])
            mins, secs = divmod(int(r["mean_time"]), 60)
            color = "#1a9850" if i == 0 else "#333"
            with cols[i % 2]:
                st.markdown(
                    f"<div style='padding:10px; border-left:4px solid {color}; margin:5px 0; background:#f8f9fa; border-radius:4px;'>"
                    f"<b>#{i+1}</b> {strat_str}<br>"
                    f"<b>{r['mean_time']:.1f}s</b> ({mins}:{secs:02d}) "
                    f"±{r['std_time']:.2f}s  [{r['min_time']:.1f}–{r['max_time']:.1f}]"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Bar chart
        st.subheader("Race Time Comparison")
        labels = [" -> ".join([f"{c[:3]}-{l}" for c, l in r["strategy"]]) for r in results[:8]]
        means = [r["mean_time"] / 60 for r in results[:8]]
        stds = [r["std_time"] / 60 for r in results[:8]]

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#1a9850"] + ["#2166ac"] * (len(labels) - 1)
        ax.barh(range(len(labels)), means, xerr=stds, color=colors, capsize=3, height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Total Race Time (minutes)")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Degradation curves
        st.subheader("Stint Degradation Curves")
        best_strat = results[0]["strategy"]
        run = opt.get_detailed_run(track, laps, driver, best_strat)
        if run:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            colors_compound = {"SOFT": "#e41a1c", "MEDIUM": "#fdb462", "HARD": "#386cb0"}
            lap_global = 1
            for s in run["stint_details"]:
                xs = list(range(lap_global, lap_global + s["laps"]))
                c = colors_compound[s["compound"]]
                ax2.plot(xs, s["lap_times"], color=c, linewidth=2, marker=".", markersize=3)
                # Trend line
                z = np.polyfit(range(len(s["lap_times"])), s["lap_times"], 1)
                p = np.poly1d(z)
                ax2.plot(xs, p(range(len(s["lap_times"]))), color=c, linestyle="--", alpha=0.5)
                lap_global += s["laps"]
            pit_x = []
            for idx, (_, sl) in enumerate(best_strat[:-1]):
                pit_lap = sum(sl for _, sl in best_strat[:idx+1])
                pit_x.append(pit_lap)
            for px in pit_x:
                ax2.axvline(x=px, color="#666", linestyle=":", alpha=0.7, label="Pit" if px == pit_x[0] else "")
            if pit_x:
                ax2.text(pit_x[0], ax2.get_ylim()[1] * 0.95, "PIT", fontsize=9, color="#666", ha="center")

            ax2.set_xlabel("Race Lap")
            ax2.set_ylabel("Lap Time (s)")
            ax2.set_title(f"Best Strategy: {' -> '.join([f'{c} ({l})' for c,l in best_strat])}")
            patches = [mpatches.Patch(color=c, label=l) for l, c in colors_compound.items()]
            ax2.legend(handles=patches)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

# ---- TAB 2: Driver Comparison ----
with tab2:
    st.subheader("Head-to-Head Strategy Comparison")
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.selectbox("Driver 1", drivers, index=drivers.index("VER"), key="d1")
    with col2:
        d2 = st.selectbox("Driver 2", drivers, index=drivers.index("HAM"), key="d2")

    col1, col2, col3 = st.columns(3)
    with col1:
        track_c = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="track_c")
    with col2:
        laps_c = st.number_input("Laps", 10, 80, 52, key="laps_c")
    with col3:
        mc_c = st.number_input("MC Runs", 10, 500, 100, step=10, key="mc_c")

    if st.button("Compare", type="primary", key="compare_btn"):
        with st.spinner("Running simulations..."):
            r1 = opt.optimize(track_c, laps_c, d1, mc_runs=mc_c, sc_prob=0.2)
            r2 = opt.optimize(track_c, laps_c, d2, mc_runs=mc_c, sc_prob=0.2)

        best1, best2 = r1[0], r2[0]
        diff = best1["mean_time"] - best2["mean_time"]
        winner = d1 if diff < 0 else d2

        col1, col2 = st.columns(2)
        for d, res, col in [(d1, best1, col1), (d2, best2, col2)]:
            strat = " -> ".join([f"{c} ({l}l)" for c, l in res["strategy"]])
            mins, secs = divmod(int(res["mean_time"]), 60)
            with col:
                st.markdown(f"**{d}**")
                st.markdown(f"{strat}")
                st.markdown(f"**{res['mean_time']:.1f}s** ({mins}:{secs:02d}) ±{res['std_time']:.2f}s")

        st.subheader(f"{winner} wins by {abs(diff):.1f}s")

        # Side by side degradation
        run1 = opt.get_detailed_run(track_c, laps_c, d1, best1["strategy"])
        run2 = opt.get_detailed_run(track_c, laps_c, d2, best2["strategy"])

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        colors_compound = {"SOFT": "#e41a1c", "MEDIUM": "#fdb462", "HARD": "#386cb0"}
        for d_name, run, style, marker in [(d1, run1, "-", "o"), (d2, run2, "--", "s")]:
            lap_global = 1
            for s in run["stint_details"]:
                xs = list(range(lap_global, lap_global + s["laps"]))
                c = colors_compound[s["compound"]]
                ax3.plot(xs, s["lap_times"], color=c, linestyle=style, linewidth=1.5,
                         marker=marker, markersize=2, label=f"{d_name} {s['compound']}")
                lap_global += s["laps"]
        ax3.set_xlabel("Race Lap")
        ax3.set_ylabel("Lap Time (s)")
        ax3.set_title(f"{d1} (solid) vs {d2} (dashed)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

# ---- TAB 3: Stint Simulator ----
with tab3:
    st.subheader("Single Stint Simulation")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sd = st.selectbox("Driver", drivers, index=drivers.index("VER"), key="sd")
    with col2:
        sc = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
    with col3:
        st_laps = st.number_input("Stint Length", 5, 50, 20)
    with col4:
        track_s = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="track_s")

    run_detailed = opt.get_detailed_run(track_s, st_laps, sd, [(sc, st_laps)])
    if run_detailed:
        sd_data = run_detailed["stint_details"][0]
        laps_arr = list(range(1, st_laps + 1))
        times = sd_data["lap_times"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Lap Time", f"{sd_data['avg_time']:.3f}s")
        col2.metric("Degradation", f"{times[-1] - times[0]:+.3f}s")
        col3.metric("First Lap", f"{times[0]:.3f}s")

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(laps_arr, times, color="#e41a1c", linewidth=2.5, marker="o", markersize=4)
        z = np.polyfit(laps_arr, times, 2)
        p = np.poly1d(z)
        ax4.plot(laps_arr, p(laps_arr), linestyle="--", color="#666", alpha=0.6, label="Trend (quadratic)")
        ax4.set_xlabel("Lap in Stint")
        ax4.set_ylabel("Lap Time (s)")
        ax4.set_title(f"{sd} — {sc} at {track_s}")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)

        # Degradation table
        st.subheader("Lap-by-Lap Breakdown")
        tbl = pd.DataFrame({"Lap": laps_arr, "Lap Time": [f"{t:.3f}s" for t in times],
                            "Delta from Lap 1": [f"{(t - times[0]):+.3f}s" for t in times]})
        st.dataframe(tbl, hide_index=True, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("F1 Lap Time Predictor & Strategy Optimizer")
st.sidebar.markdown("Predicts lap times and finds optimal pit strategies")
st.sidebar.markdown(f"**Drivers loaded:** {len(drivers)}")
st.sidebar.markdown(f"**Tracks loaded:** {len(tracks)}")
