"""
F1 Race Results Tab — Session results for any track/session combination.

Supports: FP1, Qualifying, Sprint Qualifying, Sprint Race, Race
Data source: fastf1 (live API)
"""
from __future__ import annotations

import warnings
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TEAM_COLORS: dict[str, str] = {
    "Red Bull Racing": "#3671c6", "Red Bull": "#3671c6",
    "Ferrari": "#e8002d",
    "Mercedes": "#27f4d2",
    "McLaren": "#ff8000",
    "Aston Martin": "#229971",
    "Alpine": "#00a1e8",
    "Haas F1 Team": "#dee1e2", "Haas": "#dee1e2",
    "Racing Bulls": "#6692ff", "RB": "#6692ff",
    "Williams": "#1868db",
    "Audi": "#ff2d00",
    "Cadillac": "#aaaaad",
}

SESSION_TYPES = ["FP1", "Qualifying", "Sprint Qualifying", "Sprint Race", "Race"]

# ── F1 points system ──────────────────────────────────────────────────────
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def _team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#666666")


def _fmt_time(td) -> str:
    """Format a timedelta to a readable lap time string."""
    if pd.isna(td) or td is None:
        return "—"
    if isinstance(td, (int, float)):
        mins = int(td // 60)
        secs = td - mins * 60
        return f"{mins}:{secs:06.3f}"
    total = td.total_seconds() if hasattr(td, "total_seconds") else float(td)
    mins = int(total // 60)
    secs = total - mins * 60
    return f"{mins}:{secs:06.3f}"


def _fmt_gap(time_val, pole_time) -> str:
    """Format gap to pole/leader."""
    if pd.isna(time_val) or pd.isna(pole_time):
        return "—"
    if isinstance(time_val, (int, float)) and isinstance(pole_time, (int, float)):
        diff = time_val - pole_time
    else:
        diff = time_val.total_seconds() - pole_time.total_seconds()
    if diff == 0:
        return "Pole"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.3f}s"


def _load_session_results(year: int, race: str, session_type: str) -> pd.DataFrame | None:
    """Load session results from fastf1. Returns None if session not available."""
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load(laps=False, telemetry=False)
        return session.results.copy()
    except Exception:
        return None


def _build_results_table(results: pd.DataFrame, session_type: str) -> pd.DataFrame:
    """Build a clean results table from fastf1 results."""
    table = pd.DataFrame()

    table["Pos"] = results["Position"].astype("Int64")
    table["Driver"] = results["FullName"]
    table["No"] = results["DriverNumber"]
    table["Team"] = results["TeamName"]

    if session_type in ("Qualifying", "Sprint Qualifying"):
        table["Q1"] = results["Q1"].apply(_fmt_time)
        table["Q2"] = results["Q2"].apply(_fmt_time)
        table["Q3"] = results["Q3"].apply(_fmt_time)
        # Gap to pole
        pole_q3 = results["Q3"].iloc[0] if len(results) > 0 else None
        table["Gap"] = results["Q3"].apply(lambda x: _fmt_gap(x, pole_q3))

    elif session_type == "Race":
        table["Grid"] = results["GridPosition"].astype("Int64")
        table["Points"] = results["Points"]
        table["Status"] = results["Status"]
        # Gap to leader
        leader_time = results["Time"].iloc[0] if len(results) > 0 else None
        table["Gap"] = results["Time"].apply(lambda x: _fmt_gap(x, leader_time))

    elif session_type == "Sprint Race":
        table["Grid"] = results["GridPosition"].astype("Int64") if "GridPosition" in results.columns else "—"
        table["Points"] = results["Points"]
        table["Status"] = results["Status"]
        leader_time = results["Time"].iloc[0] if len(results) > 0 else None
        table["Gap"] = results["Time"].apply(lambda x: _fmt_gap(x, leader_time))

    elif session_type == "FP1":
        table["Best"] = results["BestLapTime"].apply(_fmt_time) if "BestLapTime" in results.columns else "—"
        table["Laps"] = results["Laps"] if "Laps" in results.columns else 0

    table = table.sort_values("Pos").reset_index(drop=True)
    return table


def _render_position_chart(results: pd.DataFrame, session_type: str, race: str, year: int):
    """Render a horizontal bar chart of finishing positions."""
    df = results.dropna(subset=["Position"]).copy()
    df = df.sort_values("Position")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    teams = df["TeamName"].values
    colors = [_team_color(t) for t in teams]
    y_pos = range(len(df))

    bars = ax.barh(y_pos, [1] * len(df), color=colors, height=0.7, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["FullName"], color="#eeeeee", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_xlim(0, 1.2)

    # Add position numbers
    for i, (_, row) in enumerate(df.iterrows()):
        pos = int(row["Position"])
        ax.text(0.05, i, f"P{pos}", va="center", ha="left", color="white",
                fontsize=9, fontweight="bold")

    ax.tick_params(axis="x", colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")

    plt.title(f"{race} {year} — {session_type}", color="#eeeeee", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_results_tab():
    """Render the race results tab."""
    # ── Get available sessions for selected year+race ─────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_available_sessions(year: int, race: str) -> list[str]:
        available = []
        for stype in SESSION_TYPES:
            try:
                session = fastf1.get_session(year, race, stype)
                session.load(laps=False, telemetry=False)
                if session.results is not None and len(session.results) > 0:
                    available.append(stype)
            except Exception:
                pass
        return available

    @st.cache_data(ttl=300, show_spinner=False)
    def _get_results(year: int, race: str, session_type: str) -> pd.DataFrame | None:
        return _load_session_results(year, race, session_type)

    @st.cache_data(ttl=300, show_spinner=False)
    def _get_schedule(year: int) -> list[str]:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule["EventFormat"] != "testing"]
        return sorted(schedule["EventName"].tolist())

    # ── Layout ────────────────────────────────────────────────────────────
    tracks = _get_schedule(2026)

    c1, c2 = st.columns(2)
    with c1:
        r_track = st.selectbox("Track", tracks, index=tracks.index("Belgian Grand Prix") if "Belgian Grand Prix" in tracks else 0, key="res_track")
    with c2:
        r_year = st.selectbox("Year", [2026, 2025], key="res_year")

    # Update tracks for selected year
    tracks_for_year = _get_schedule(r_year)
    if r_track not in tracks_for_year:
        r_track = tracks_for_year[0]
    with c1:
        r_track = st.selectbox("Track", tracks_for_year, index=tracks_for_year.index(r_track) if r_track in tracks_for_year else 0, key="res_track_2")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load available sessions ───────────────────────────────────────────
    with st.spinner(f"Loading sessions for {r_track}..."):
        available = _get_available_sessions(r_year, r_track)

    if not available:
        st.info("No session data available for this race.")
        return

    # ── Session selector ──────────────────────────────────────────────────
    r_session = st.radio("Session", available, horizontal=True, key="res_session")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load and display results ──────────────────────────────────────────
    with st.spinner(f"Loading {r_session} results..."):
        results = _get_results(r_year, r_track, r_session)

    if results is None or results.empty:
        st.info(f"No results available for {r_session}.")
        return

    table = _build_results_table(results, r_session)

    # ── Summary metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drivers", len(results))

    if r_session == "Race" and "Status" in results.columns:
        finishers = len(results[results["Status"] == "Finished"])
        m2.metric("Finishers", finishers)
        m3.metric("DNFs", len(results) - finishers)
        if "GridPosition" in results.columns and "Position" in results.columns:
            best_mover = results.copy()
            best_mover["Gain"] = best_mover["GridPosition"] - best_mover["Position"]
            if len(best_mover) > 0:
                mover = best_mover.loc[best_mover["Gain"].idxmax()]
                m4.metric("Best Mover", f"{mover['FullName']}", f"+{int(mover['Gain'])} places")
    elif r_session == "Sprint Race" and "Status" in results.columns:
        finishers = len(results[results["Status"] == "Finished"])
        m2.metric("Finishers", finishers)
        m3.metric("DNFs", len(results) - finishers)
        m4.metric("—", "—")
    else:
        m2.metric("—", "—")
        m3.metric("—", "—")
        m4.metric("—", "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Position chart ────────────────────────────────────────────────────
    _render_position_chart(results, r_session, r_track, r_year)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{r_session} Results</div>", unsafe_allow_html=True)

    # Color the team column
    def _color_team(row):
        team = row.get("Team", "")
        c = _team_color(team)
        return [f"background-color: {c}22"] * len(row)

    styled = table.style.apply(_color_team, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=min(40 * len(table) + 40, 500))
