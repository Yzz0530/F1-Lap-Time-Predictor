"""
Legacy training pipeline — kept for reference.
Use train.py instead for production training.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import os
import optuna
from typing import Any

BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE, "data")
MODEL_DIR: str = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "all_races_master.csv"))
print(f"Dataset: {df.shape}")

le_driver: LabelEncoder = LabelEncoder()
le_compound: LabelEncoder = LabelEncoder()

df["Driver_enc"] = le_driver.fit_transform(df["Driver"])
df["Compound_enc"] = le_compound.fit_transform(df["Compound"].astype(str))

compound_map: dict[str, str] = {"SOFT": "DRY", "MEDIUM": "DRY", "HARD": "DRY", "INTERMEDIATE": "WET"}
df["CompoundFamily"] = df["Compound"].fillna("UNKNOWN").map(compound_map).fillna("UNKNOWN")
le_family: LabelEncoder = LabelEncoder()
df["CompoundFamily_enc"] = le_family.fit_transform(df["CompoundFamily"])

compound_order: dict[str, int] = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4}
df["CompoundOrdinal"] = df["Compound"].fillna("MEDIUM").map(compound_order).fillna(2)
df["IsWet"] = (df["Compound"] == "INTERMEDIATE").astype(int)

stint_total = df.groupby(["Race", "Driver", "Stint"])["LapInRace"].transform("max")
progress = df["LapInRace"] / stint_total.clip(lower=1)
df["StintPhase"] = pd.cut(progress, bins=[0, 0.33, 0.66, 1], labels=[0, 1, 2]).fillna(0).astype(int)

df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")
df["Target"] = df["LapTime"] - df["RaceBaseline"]

misc: list[str] = ["Position_normalized", "IsPersonalBest_int", "FreshTire_int", "IsStartLap"]
sectors: list[str] = ["S1_speed", "S2_speed", "S3_speed", "AvgSpeed"]
weather: list[str] = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]

features: list[str] = [
    "Driver_enc", "Compound_enc", "CompoundFamily_enc", "CompoundOrdinal", "IsWet",
    "TyreLife", "TyreLife_sq", "Stint", "StintPhase", "TrackStatus",
    "LapInRace", "LapInRace_sq", "FuelWeightEffect", "DriverForm",
] + misc + sectors + weather

df = df.dropna(subset=features + ["Target"])
X: pd.DataFrame = df[features]
y: pd.Series = df["Target"]

races: np.ndarray = df["Race"].unique()
rng: np.random.RandomState = np.random.RandomState(42)
rng.shuffle(races)
split_idx: int = int(len(races) * 0.8)
train_races, test_races = races[:split_idx], races[split_idx:]

X_train = X[df["Race"].isin(train_races)]
y_train = y[df["Race"].isin(train_races)]
X_test = X[df["Race"].isin(test_races)]
y_test = y[df["Race"].isin(test_races)]

print(f"\nTrain: {len(train_races)} races, {len(X_train)} samples")
print(f"Test:  {len(test_races)} races, {len(X_test)} samples")

print("\n--- Tuning XGBoost ---")
def objective(trial: optuna.Trial) -> float:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
    }
    m: xgb.XGBRegressor = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return float(mean_absolute_error(y_test, m.predict(X_test)))

study: optuna.study.Study = optuna.create_study(direction="minimize", study_name="xgb_master")
study.optimize(objective, n_trials=25, show_progress_bar=True)

model: xgb.XGBRegressor = xgb.XGBRegressor(**study.best_params, random_state=42)
model.fit(X_train, y_train)
val_mae: float = float(mean_absolute_error(y_test, model.predict(X_test)))
print(f"Best params: {study.best_params}")
print(f"Validation MAE: {val_mae:.4f}")

joblib.dump(model, os.path.join(MODEL_DIR, "xgb_master.pkl"))
joblib.dump(le_driver, os.path.join(MODEL_DIR, "le_driver_master.pkl"))
joblib.dump(le_compound, os.path.join(MODEL_DIR, "le_compound_master.pkl"))
joblib.dump(le_family, os.path.join(MODEL_DIR, "le_family_master.pkl"))
joblib.dump(features, os.path.join(MODEL_DIR, "feature_list_master.pkl"))

fi: pd.DataFrame = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
print(f"\nTop 10 features:\n{fi.head(10).to_string(index=False)}")

print(f"\nValidation MAE: {val_mae:.4f}s")
