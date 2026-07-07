import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import numpy as np

# Load data
df = pd.read_csv("../data/all_races_clean.csv")

print("Dataset shape:", df.shape)

# Encode categorical variables
le_driver = LabelEncoder()
le_compound = LabelEncoder()

df["Driver"] = le_driver.fit_transform(df["Driver"])
df["Compound"] = le_compound.fit_transform(df["Compound"].astype(str))

# 🧠 STEP 1: create race baseline (VERY IMPORTANT FIX)
df["RaceAvgLap"] = df.groupby("Race")["LapTime"].transform("mean")

# Target becomes deviation from race pace
df["LapDelta"] = df["LapTime"] - df["RaceAvgLap"]

# Features (NO raw race effect now)
features = [
    "Driver",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "LapNumber"
]

X = df[features]
y = df["LapDelta"]

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

print(f"Train races: {len(train_races)}, Test races: {len(test_races)}")

# Model
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("MAE (delta seconds):", mae)