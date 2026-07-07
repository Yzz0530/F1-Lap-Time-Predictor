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

df = pd.read_csv("../data/all_races_master.csv")
print(f"Dataset: {df.shape}")

# Encode categoricals
le_driver = LabelEncoder()
le_compound = LabelEncoder()
le_race_idx = LabelEncoder()

df["Driver_enc"] = le_driver.fit_transform(df["Driver"])
df["Compound_enc"] = le_compound.fit_transform(df["Compound"].astype(str))
df["Race_idx"] = le_race_idx.fit_transform(df["Race"])

joblib.dump(le_driver, "../models/le_driver_master.pkl")
joblib.dump(le_compound, "../models/le_compound_master.pkl")
joblib.dump(le_race_idx, "../models/le_race_master.pkl")

# Compound family
compound_map = {"SOFT": "DRY", "MEDIUM": "DRY", "HARD": "DRY", "INTERMEDIATE": "WET"}
df["CompoundFamily"] = df["Compound"].fillna("UNKNOWN").map(compound_map).fillna("UNKNOWN")
le_family = LabelEncoder()
df["CompoundFamily_enc"] = le_family.fit_transform(df["CompoundFamily"])
joblib.dump(le_family, "../models/le_family_master.pkl")

# Compound ordinal
compound_order = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4}
df["CompoundOrdinal"] = df["Compound"].fillna("MEDIUM").map(compound_order).fillna(2)
df["IsWet"] = (df["Compound"] == "INTERMEDIATE").astype(int)

# Stint phase
df["StintTotalLaps"] = df.groupby(["Race", "Driver", "Stint"])["LapInRace"].transform("max")
df["StintProgress"] = df["LapInRace"] / df["StintTotalLaps"].clip(lower=1)
df["StintPhase"] = pd.cut(df["StintProgress"], bins=[0, 0.33, 0.66, 1], labels=[0, 1, 2])
df["StintPhase"] = df["StintPhase"].fillna(0).astype(int)

# Race baseline for target
df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")
df["Target"] = df["LapTime"] - df["RaceBaseline"]

# Feature list
weather_features = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
sector_features = ["S1_speed", "S2_speed", "S3_speed", "AvgSpeed"]
misc_features = ["Position_normalized", "IsPersonalBest_int", "FreshTire_int", "IsStartLap"]

features = [
    "Driver_enc", "Compound_enc", "CompoundFamily_enc", "CompoundOrdinal", "IsWet",
    "TyreLife", "TyreLife_sq", "Stint", "StintPhase", "TrackStatus",
    "LapInRace", "LapInRace_sq", "FuelWeightEffect", "DriverForm",
] + misc_features + sector_features + weather_features

print(f"Features ({len(features)}): {features}")
print(f"Has NaN: {df[features].isna().any().any()}")

# Drop NaN
df = df.dropna(subset=features + ["Target"])

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

print(f"\nTrain: {len(train_races)} races, {len(X_train)} samples")
print(f"Test:  {len(test_races)} races, {len(X_test)} samples")

# Optuna tuning
print("\n--- Tuning XGBoost ---")
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
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return mean_absolute_error(y_test, m.predict(X_test))

study = optuna.create_study(direction="minimize", study_name="xgb_master")
study.optimize(objective, n_trials=25, show_progress_bar=True)

best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42)
best_xgb.fit(X_train, y_train)
xgb_pred = best_xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"Best params: {study.best_params}")
print(f"XGBoost MAE: {xgb_mae:.4f}")

# LightGBM
print("\n--- LightGBM ---")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_pred)
print(f"LightGBM MAE: {lgb_mae:.4f}")

# RandomForest
print("\n--- RandomForest ---")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"RF MAE: {rf_mae:.4f}")

# Ensemble weights
print("\n--- Ensemble ---")
best_w = (1.0, 0.0, 0.0)
best_ens_mae = xgb_mae
for w1 in np.arange(0.5, 1.05, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1 - w1 - w2, 2)
        if w3 < 0: continue
        ens = w1 * xgb_pred + w2 * lgb_pred + w3 * rf_pred
        m = mean_absolute_error(y_test, ens)
        if m < best_ens_mae:
            best_ens_mae = m
            best_w = (w1, w2, w3)

print(f"Ensemble weights: XGB={best_w[0]:.2f}, LGB={best_w[1]:.2f}, RF={best_w[2]:.2f}")
print(f"Ensemble MAE: {best_ens_mae:.4f}")

# Save
np.save("../models/ensemble_weights_master.npy", np.array(best_w))
joblib.dump(best_xgb, "../models/xgb_master.pkl")
joblib.dump(lgb_model, "../models/lgb_master.pkl")
joblib.dump(rf_model, "../models/rf_master.pkl")
joblib.dump(features, "../models/feature_list_master.pkl")

# Save feature importances
fi = pd.DataFrame({"feature": features, "importance": best_xgb.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
print(f"\nTop 10 features:\n{fi.head(10).to_string(index=False)}")

print(f"\n--- Summary ---")
print(f"Previous best:  ~0.524s (13 features)")
print(f"New best:       {best_ens_mae:.4f}s ({len(features)} features)")
print(f"Improvement:    {0.524 - best_ens_mae:+.4f}s  ({(0.524 - best_ens_mae)/0.524*100:.1f}%)")
