import fastf1
import pandas as pd
import os

BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR: str = os.path.join(BASE, "cache")
DATA_DIR: str = os.path.join(BASE, "data")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

YEAR: int = 2026

schedule: pd.DataFrame = fastf1.get_event_schedule(YEAR)
today: pd.Timestamp = pd.Timestamp.now()
schedule = schedule[~schedule["EventFormat"].isin(["testing", None])]
schedule = schedule[schedule["EventDate"] <= today]
races: list[str] = schedule["EventName"].tolist()
print(f"Found {len(races)} completed race events in {YEAR}:")
for r in races:
    print(f"  - {r}")

all_laps: list[pd.DataFrame] = []
for race in races:
    try:
        print(f"\nLoading {race}...")
        session = fastf1.get_session(YEAR, race, "R")
        session.load()
        laps = session.laps
        laps["Race"] = race
        all_laps.append(laps)
        print(f"{race} loaded: {len(laps)} laps")
    except Exception as e:
        print(f"Failed {race}: {e}")

df: pd.DataFrame = pd.concat(all_laps, ignore_index=True)
file_path: str = os.path.join(DATA_DIR, f"all_races_{YEAR}.csv")
df.to_csv(file_path, index=False)
print("\nDONE")
print("Total laps:", len(df))
print("Saved to:", file_path)
