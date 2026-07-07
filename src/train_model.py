import pandas as pd

# Load data
df = pd.read_csv("../data/silverstone_2025.csv")

print("Raw data shape:", df.shape)

# Keep useful columns only (FastF1 sometimes has extra junk columns)
cols = [
    "Driver",
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus"
]

df = df[cols]

# Drop missing lap times (important)
df = df.dropna(subset=["LapTime"])

# Convert LapTime → seconds
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

#Keep only realistic racing laps
df = df[(df["LapTime"] > 80) & (df["LapTime"] < 120)]

df["LapNumber"] = df.groupby("Driver").cumcount() + 1

print(df.head())
print("Clean data shape:", df.shape)

from sklearn.preprocessing import LabelEncoder

le_driver = LabelEncoder()
le_compound = LabelEncoder()

df["Driver"] = le_driver.fit_transform(df["Driver"])
df["Compound"] = le_compound.fit_transform(df["Compound"].astype(str))

X = df[["Driver", "Compound", "TyreLife", "Stint", "TrackStatus", "LapNumber"]]
y = df["LapTime"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("MAE (seconds):", mae)
