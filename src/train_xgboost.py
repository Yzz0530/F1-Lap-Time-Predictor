import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Load cleaned data
df = pd.read_csv("../data/all_races_clean.csv")

print("Dataset shape:", df.shape)

# Encode categorical data
le_driver = LabelEncoder()
le_compound = LabelEncoder()
le_race = LabelEncoder()

df["Driver"] = le_driver.fit_transform(df["Driver"])
df["Compound"] = le_compound.fit_transform(df["Compound"].astype(str))
df["Race"] = le_race.fit_transform(df["Race"])

# Features
X = df[[
    "Driver",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "LapNumber",
    "Race"
]]

y = df["LapTime"]

# Split by race (no data leakage)
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

# XGBoost model (strong for structured data)
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

print("MAE (seconds):", mae)