import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("../data/all_races_clean.csv")
df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")

le_driver = joblib.load("../models/le_driver.pkl")
le_compound = joblib.load("../models/le_compound.pkl")
model = joblib.load("../models/xgb_f1_model.pkl")

# 3. Enhanced Simulation Function
def simulate_stint(driver, compound, race_name, stint_length=20):
    try:
        driver_enc = le_driver.transform([driver])[0]
        compound_enc = le_compound.transform([str(compound)])[0]
    except ValueError as e:
        print(f"Error: Driver '{driver}' or Compound '{compound}' wasn't found in training data.")
        return [], []

    # Target the exact track selected
    race_data = df[df["Race"] == race_name]
    if race_data.empty:
        print(f"Error: Race '{race_name}' not found in data.")
        return [], []
        
    race_baseline = race_data["RaceBaseline"].iloc[0]

    laps = []
    predictions = []

    for lap in range(1, stint_length + 1):
        tyre_life = lap
        tyre_sq = tyre_life ** 2
        lap_sq = lap ** 2
        fuel_effect = lap * -0.03  # Car gets lighter every lap

        # Must match our trained feature array exactly
        input_data = pd.DataFrame([[
            driver_enc,
            compound_enc,
            tyre_life,
            tyre_sq,
            1,       # Stint number
            1,       # Track status (Green Flag)
            lap,     # Current Race Lap
            lap_sq,
            fuel_effect
        ]], columns=[
            "Driver_enc", "Compound_enc", "TyreLife", "TyreLife_sq", 
            "Stint", "TrackStatus", "LapNumber", "LapNumber_sq", "FuelWeightEffect"
        ])

        # Predict the delta and add it to the specific track's baseline
        delta = model.predict(input_data)[0]
        lap_time = race_baseline + delta

        laps.append(lap)
        predictions.append(lap_time)

    return laps, predictions

if __name__ == "__main__":
    target_race = "British Grand Prix" 
    target_driver = df["Driver"].iloc[0]
    target_compound = df["Compound"].iloc[0]

    laps, times = simulate_stint(
        driver=target_driver, 
        compound=target_compound, 
        race_name=target_race, 
        stint_length=25
    )

    if laps:
        plt.figure(figsize=(10, 5))
        plt.plot(laps, times, marker='o', color='red', linewidth=2)
        plt.xlabel("Lap of Stint")
        plt.ylabel("Predicted Lap Time (seconds)")
        plt.title(f"F1 Stint Simulation: {target_driver} at {target_race} ({target_compound} Tyres)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        print("Simulation complete.")