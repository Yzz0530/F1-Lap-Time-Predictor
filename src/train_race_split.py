import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Load cleaned dataset
df = pd.read_csv("../data/all_races_clean.csv")

print("Dataset shape:", df.shape)

# Encode categorical features
le_driver = LabelEncoder()
le_compound = LabelEncoder()
le_race = LabelEncoder()

df["Driver"] = le_driver.fit_transform(df["Driver"])
df["Compound"] = le_compound.fit_transform(df["Compound"].astype(str))
df["Race"] = le_race.fit_transform(df["Race"])

# Features
features = [
    "Driver",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "LapNumber",
    "Race"
]

X = df[features]
y = df["LapTime"]

# 🚨 IMPORTANT: race-based split (NO leakage)
train_races = df["Race"].unique()[:18]   # train on most races
test_races = df["Race"].unique()[18:]    # test on unseen races

train_df = df[df["Race"].isin(train_races)]
test_df = df[df["Race"].isin(test_races)]

X_train = train_df[features]
y_train = train_df["LapTime"]

X_test = test_df[features]
y_test = test_df["LapTime"]

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

print("Race-split MAE (seconds):", mae)