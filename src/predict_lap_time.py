import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

os.makedirs("../models", exist_ok=True)

df = pd.read_csv("../data/all_races_clean.csv")

le_driver = LabelEncoder()
le_compound = LabelEncoder()

df["Driver_enc"] = le_driver.fit_transform(df["Driver"])
df["Compound_enc"] = le_compound.fit_transform(df["Compound"].astype(str))

joblib.dump(le_driver, "../models/le_driver.pkl")
joblib.dump(le_compound, "../models/le_compound.pkl")

df["TyreLife_sq"] = df["TyreLife"] ** 2
df["LapNumber_sq"] = df["LapNumber"] ** 2

df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")
df["Target"] = df["LapTime"] - df["RaceBaseline"]

features = [
    "Driver_enc", "Compound_enc",
    "TyreLife", "TyreLife_sq",
    "Stint", "TrackStatus",
    "LapNumber", "LapNumber_sq",
    "FuelWeightEffect"
]

X = df[features]
y = df["Target"]

# Split by race to avoid data leakage
races = df["Race"].unique()
np.random.seed(42)
np.random.shuffle(races)
split_idx = int(len(races) * 0.8)
train_races = races[:split_idx]
test_races = races[split_idx:]

X_train = X[df["Race"].isin(train_races)]
y_train = y[df["Race"].isin(train_races)]
X_test = X[df["Race"].isin(test_races)]
y_test = y[df["Race"].isin(test_races)]

print(f"Training on {len(train_races)} races, testing on {len(test_races)} races")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"Test MAE (delta seconds): {mae:.3f}")

# Also show in real lap times
test_baseline = df[df["Race"].isin(test_races)]["RaceBaseline"].values
real_pred = pred + test_baseline[:len(pred)]
real_actual = y_test.values + test_baseline[:len(pred)]
real_mae = mean_absolute_error(real_actual, real_pred)
print(f"Test MAE (actual lap time seconds): {real_mae:.3f}")

joblib.dump(model, "../models/xgb_f1_model.pkl")
print("Model and encoders saved to ../models/")