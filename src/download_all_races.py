"""
Download all completed race lap data for a given F1 season from fastf1.

Usage:
    python src/download_all_races.py --year 2025
    python src/download_all_races.py --year 2026
"""
from __future__ import annotations

import argparse
import os
import sys

import fastf1
import pandas as pd

BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR: str = os.path.join(BASE, "cache")
DATA_DIR: str = os.path.join(BASE, "data")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)


def download_year(year: int) -> None:
    schedule: pd.DataFrame = fastf1.get_event_schedule(year)
    today: pd.Timestamp = pd.Timestamp.now()
    schedule = schedule[schedule["EventFormat"] != "testing"]
    schedule = schedule[schedule["EventDate"] <= today]
    races: list[str] = schedule["EventName"].tolist()

    print(f"Found {len(races)} completed race events in {year}:")
    for r in races:
        print(f"  - {r}")

    all_laps: list[pd.DataFrame] = []
    for race in races:
        try:
            print(f"\nLoading {race} ({year})...")
            session = fastf1.get_session(year, race, "R")
            session.load()
            laps = session.laps
            laps["Race"] = race
            all_laps.append(laps)
            print(f"{race} loaded: {len(laps)} laps")
        except Exception as e:
            print(f"Failed {race}: {e}")

    if not all_laps:
        print(f"\nNo race data available for {year}. Exiting.")
        return

    df: pd.DataFrame = pd.concat(all_laps, ignore_index=True)
    file_path: str = os.path.join(DATA_DIR, f"all_races_{year}.csv")
    df.to_csv(file_path, index=False)
    print(f"\nDONE — {year}")
    print(f"Total laps: {len(df)}")
    print(f"Saved to: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download F1 lap data for a season.")
    parser.add_argument("--year", type=int, default=2026, help="Season year (default: 2026)")
    args = parser.parse_args()
    download_year(args.year)


if __name__ == "__main__":
    main()
