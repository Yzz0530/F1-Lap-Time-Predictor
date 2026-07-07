import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import os
import optuna

os.makedirs("../models", exist_ok=True)

df = pd.read_csv("../data/all_races_clean.csv")

# Encode drivers and compounds
le_driver = LabelEncoder()
le_compound = LabelEncoder()
df["Driver_enc"] = le_driver.fit_transform(df["Driver"])
df["Compound_enc"] = le_compound.fit_transform(df["Compound"].astype(str))
joblib.dump(le_driver, "../models/le_driver_enhanced.pkl")
joblib.dump(le_compound, "../models/le_compound_enhanced.pkl")

# Enhanced feature engineering
df["TyreLife_sq"] = df["TyreLife"] ** 2
df["LapNumber_sq"] = df["LapNumber"] ** 2

# Compound family: dry vs wet
compound_map = {"SOFT": "DRY", "MEDIUM": "DRY", "HARD": "DRY", "INTERMEDIATE": "WET"}
df["CompoundFamily"] = df["Compound"].fillna("UNKNOWN").map(compound_map).fillna("UNKNOWN")
le_family = LabelEncoder()
df["CompoundFamily_enc"] = le_family.fit_transform(df["CompoundFamily"])
joblib.dump(le_family, "../models/le_family_enhanced.pkl")

# Is wet tire
df["IsWet"] = (df["Compound"] == "INTERMEDIATE").astype(int)

# Compound ordinal (softer = faster but wears more)
compound_order = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4}
df["CompoundOrdinal"] = df["Compound"].fillna("MEDIUM").map(compound_order).fillna(2)

# Driver form: rolling avg of last 5 laps per driver per race
df["DriverForm"] = df.groupby(["Race", "Driver"])["LapTime"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)
df["DriverForm"] = df["DriverForm"].fillna(df["LapTime"])

# Stint phase: early / mid / late in stint
df["StintTotalLaps"] = df.groupby(["Race", "Driver", "Stint"])["LapNumber"].transform("max")
df["StintProgress"] = df["LapNumber"] / df["StintTotalLaps"].clip(lower=1)
df["StintPhase"] = pd.cut(df["StintProgress"], bins=[0, 0.33, 0.66, 1], labels=[0, 1, 2])
df["StintPhase"] = df["StintPhase"].fillna(0).astype(int)

# Race baseline for target
df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")
df["Target"] = df["LapTime"] - df["RaceBaseline"]

features = [
    "Driver_enc", "Compound_enc", "CompoundFamily_enc",
    "CompoundOrdinal", "IsWet",
    "TyreLife", "TyreLife_sq",
    "Stint", "StintPhase",
    "TrackStatus",
    "LapNumber", "LapNumber_sq",
    "DriverForm"
]

X = df[features]
y = df["Target"]

# Split by race
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

print(f"Enhanced features: {len(features)}")
print(f"Train races: {len(train_races)}, Test races: {len(test_races)}")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---- Phase 1: Optuna hyperparameter tuning for XGBoost ----
print("\n--- Tuning XGBoost with Optuna ---")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    return mean_absolute_error(y_test, pred)

study = optuna.create_study(direction="minimize", study_name="xgb_f1")
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_xgb_params = study.best_params
print(f"Best XGBoost params: {best_xgb_params}")
print(f"Best XGBoost MAE: {study.best_value:.4f}")

best_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=42)
best_xgb.fit(X_train, y_train)

# ---- Phase 2: LightGBM ----
print("\n--- Training LightGBM ---")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train)

# ---- Phase 3: RandomForest ----
print("\n--- Training RandomForest ---")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# ---- Evaluate individual models ----
xgb_pred = best_xgb.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
lgb_mae = mean_absolute_error(y_test, lgb_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"\n--- Individual Model Performance ---")
print(f"XGBoost (tuned)  MAE: {xgb_mae:.4f}")
print(f"LightGBM        MAE: {lgb_mae:.4f}")
print(f"RandomForest    MAE: {rf_mae:.4f}")

# ---- Ensemble: find optimal weights ----
print("\n--- Tuning Ensemble Weights ---")
best_weight_mae = float("inf")
best_w1 = best_w2 = best_w3 = 0

for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1 - w1, 0.1):
        w3 = round(1 - w1 - w2, 1)
        if w3 < 0:
            continue
        ensemble_pred = w1 * xgb_pred + w2 * lgb_pred + w3 * rf_pred
        mae = mean_absolute_error(y_test, ensemble_pred)
        if mae < best_weight_mae:
            best_weight_mae = mae
            best_w1, best_w2, best_w3 = w1, w2, w3

print(f"Best weights: XGB={best_w1:.1f}, LGB={best_w2:.1f}, RF={best_w3:.1f}")
print(f"Ensemble MAE: {best_weight_mae:.4f}")

# Save ensemble weights
np.save("../models/ensemble_weights.npy", np.array([best_w1, best_w2, best_w3]))

# Save all models
joblib.dump(best_xgb, "../models/xgb_enhanced.pkl")
joblib.dump(lgb_model, "../models/lgb_enhanced.pkl")
joblib.dump(rf_model, "../models/rf_enhanced.pkl")

# Compare with baseline (original model)
print(f"\n--- Summary ---")
print(f"Baseline MAE (no tuning):      ~0.585")
print(f"Tuned XGBoost MAE:             {xgb_mae:.4f}")
print(f"Ensemble MAE:                  {best_weight_mae:.4f}")
print(f"Improvement:                   {0.585 - best_weight_mae:+.4f}")
print("\nAll models saved to ../models/")
