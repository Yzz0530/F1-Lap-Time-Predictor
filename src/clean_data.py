import pandas as pd
import numpy as np

# Load full dataset
df = pd.read_csv("../data/all_races_2025.csv")
print("Raw shape:", df.shape)

# Keep only useful columns
cols = [
    "Driver",
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "Race"
]
df = df[cols]

# Drop missing lap times
df = df.dropna(subset=["LapTime"])

# Convert LapTime to seconds
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

# 🔥 DYNAMIC FILTER: Remove pit stops, crashes, and safety cars per race
# Calculates the fastest lap for each specific Grand Prix
df["MinRaceLap"] = df.groupby("Race")["LapTime"].transform("min")

# Keep only laps within 107% of the fastest lap for that track
df = df[(df["LapTime"] >= df["MinRaceLap"]) & (df["LapTime"] <= df["MinRaceLap"] * 1.07)]
df = df.drop(columns=["MinRaceLap"])

# Add tyre degradation tracking feature
df["LapNumber"] = df.groupby(["Race", "Driver"]).cumcount() + 1

# Add Fuel Weight feature (Assuming a max 60-lap race, fuel burning off)
# Cars get roughly 0.03s faster per lap as fuel burns off
df["FuelWeightEffect"] = df["LapNumber"] * -0.03

print("Clean shape:", df.shape)

# Save cleaned dataset
out_path = "../data/all_races_clean.csv"
df.to_csv(out_path, index=False)
print("Saved cleaned dataset to:", out_path)