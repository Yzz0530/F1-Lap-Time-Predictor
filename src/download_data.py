import fastf1
import os

os.makedirs("../cache", exist_ok=True)
os.makedirs("../data", exist_ok=True)

fastf1.Cache.enable_cache("../cache")

print("Loading F1 session...")

# Load Silverstone race session
session = fastf1.get_session(2025, "Silverstone", "R")
session.load()

print("Session loaded!")

# Get lap data
laps = session.laps

print(f"Total laps: {len(laps)}")

# Make sure data folder exists
os.makedirs("../data", exist_ok=True)

# Save file
file_path = "../data/silverstone_2025.csv"
laps.to_csv(file_path, index=False)

print(f"Data saved to {file_path}")
