import pandas as pd
import numpy as np
import fastf1
import os
import warnings
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache("C:/F1_CACHE")

RAW_PATH = "../data/all_races_2026.csv"
OUT_PATH = "../data/all_races_master.csv"

# Build race name -> location map dynamically from the fastf1 schedule
schedule = fastf1.get_event_schedule(2026)
RACE_MAP = {}
for _, r in schedule.iterrows():
    if r["EventFormat"] != "testing":
        RACE_MAP[r["EventName"]] = r["Location"]

print("Loading raw data...")
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

# Parse timedelta columns
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

# Convert Time to numeric seconds for merge
df["Time_sec"] = pd.to_timedelta(df["Time"]).dt.total_seconds()

# --- Feature engineering from raw data ---
print("Engineering features...")

# Sector speeds (km/h from time deltas)
for s, name in [(1, "S1"), (2, "S2"), (3, "S3")]:
    col = f"Sector{s}Time"
    df[f"{name}_speed"] = 1000 / df[col].clip(lower=0.001) * 3.6  # rough sector speed

# Speed trap features (already in km/h)
df["AvgSpeed"] = df[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].mean(axis=1)

# Position as a feature (normalized by number of cars)
df["Position_normalized"] = df["Position"] / df.groupby("Race")["Position"].transform("max")

# Is personal best
df["IsPersonalBest_int"] = df["IsPersonalBest"].fillna(False).astype(int)

# Fresh tire
if "FreshTyre" in df.columns:
    df["FreshTire_int"] = df["FreshTyre"].fillna(False).astype(int)
else:
    df["FreshTire_int"] = 0

# Qualifying vs race stint flag (based on TyreLife = 0 or 1)
df["IsStartLap"] = (df["TyreLife"] <= 1).astype(int)

# --- Weather data - load per race ---
print("Loading weather data (this will take ~1-2 min)...")
weather_dfs = []

all_races = df["Race"].unique()
for i, race in enumerate(all_races):
    try:
        short_name = RACE_MAP.get(race, race.split()[0])
        session = fastf1.get_session(2026, short_name, "R")
        session.load(laps=False, telemetry=False, weather=True)
        wd = session.weather_data.copy()
        wd["Race"] = race
        weather_dfs.append(wd)
        print(f"  [{i+1}/{len(all_races)}] {race}: {len(wd)} weather rows")
    except Exception as e:
        print(f"  [{i+1}/{len(all_races)}] {race}: FAILED - {e}")

all_weather = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()
print(f"Weather data combined: {len(all_weather)} rows")

# Merge weather with lap data (asof merge on elapsed time)
if not all_weather.empty:
    all_weather["Time_sec"] = pd.to_timedelta(all_weather["Time"]).dt.total_seconds()
    
    merged_dfs = []
    for race in all_races:
        race_laps = df[df["Race"] == race].sort_values("Time_sec").copy()
        race_wx = all_weather[all_weather["Race"] == race].sort_values("Time_sec").copy()
        if len(race_wx) == 0:
            merged_dfs.append(race_laps)
            continue
        merged = pd.merge_asof(race_laps, race_wx[["Time_sec", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]],
                               on="Time_sec", direction="nearest")
        merged_dfs.append(merged)
    df = pd.concat(merged_dfs, ignore_index=True)
else:
    pass

df = df.drop(columns=["Time_sec"], errors="ignore")

# --- Final cleaning ---
print("Final cleaning...")
df = df.dropna(subset=["LapTime"])
df = df[df["LapTime"] > 60]

# Per-race filtering (107% rule)
df["MinRaceLap"] = df.groupby("Race")["LapTime"].transform("min")
df = df[df["LapTime"] <= df["MinRaceLap"] * 1.07]
df = df.drop(columns=["MinRaceLap"])

# Lap number within race (not just stint)
df["LapInRace"] = df.groupby(["Race", "Driver"]).cumcount() + 1

# Fuel weight effect
df["FuelWeightEffect"] = df["LapInRace"] * -0.03

# Feature engineering for ML
df["TyreLife_sq"] = df["TyreLife"] ** 2
df["LapInRace_sq"] = df["LapInRace"] ** 2

# Driver form (rolling avg of last 5 laps)
df["DriverForm"] = df.groupby("Driver")["LapTime"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)
df["DriverForm"] = df["DriverForm"].fillna(df["LapTime"])

# --- Clean up columns ---
keep_cols = [
    "Driver", "LapTime", "Compound", "TyreLife", "TyreLife_sq",
    "Stint", "TrackStatus", "Race", "LapInRace", "LapInRace_sq",
    "FuelWeightEffect", "DriverForm", "Position_normalized",
    "S1_speed", "S2_speed", "S3_speed", "AvgSpeed",
    "IsPersonalBest_int", "FreshTire_int", "IsStartLap",
    "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"
]
keep = [c for c in keep_cols if c in df.columns]
df = df[keep]

# Drop rows with missing values
df = df.dropna()

print(f"Final shape: {df.shape}")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")
print(f"Features: {[c for c in df.columns if c not in ['Driver','LapTime','Race']]}")
