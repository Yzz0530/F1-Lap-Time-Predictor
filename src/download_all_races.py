import fastf1
import pandas as pd
import os

import os

# Use project-local paths
os.makedirs("../cache", exist_ok=True)
os.makedirs("../data", exist_ok=True)

fastf1.Cache.enable_cache("../cache")

# 2025 race list (we'll expand later if needed)
races = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Japanese Grand Prix",
    "Chinese Grand Prix",
    "Miami Grand Prix",
    "Emilia Romagna Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Canadian Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix"
]

all_laps = []

for race in races:
    try:
        print(f"\nLoading {race}...")

        session = fastf1.get_session(2025, race, "R")
        session.load()

        laps = session.laps

        laps["Race"] = race

        all_laps.append(laps)

        print(f"{race} loaded: {len(laps)} laps")

    except Exception as e:
        print(f"Failed {race}: {e}")

# Combine all races
df = pd.concat(all_laps, ignore_index=True)

file_path = "../data/all_races_2025.csv"
df.to_csv(file_path, index=False)

print("\nDONE")
print("Total laps:", len(df))
print("Saved to:", file_path)