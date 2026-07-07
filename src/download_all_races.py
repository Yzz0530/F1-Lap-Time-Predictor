import fastf1
import pandas as pd
import os

os.makedirs("../cache", exist_ok=True)
os.makedirs("../data", exist_ok=True)

fastf1.Cache.enable_cache("../cache")

YEAR = 2026

schedule = fastf1.get_event_schedule(YEAR)
today = pd.Timestamp.now()
schedule = schedule[~schedule["EventFormat"].isin(["testing", None])]
schedule = schedule[schedule["EventDate"] <= today]
races = schedule["EventName"].tolist()
print(f"Found {len(races)} completed race events in {YEAR}:")
for r in races:
    print(f"  - {r}")

all_laps = []
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

df = pd.concat(all_laps, ignore_index=True)
file_path = f"../data/all_races_{YEAR}.csv"
df.to_csv(file_path, index=False)
print("\nDONE")
print("Total laps:", len(df))
print("Saved to:", file_path)
