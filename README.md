# F1 Lap Time Predictor

Predicts F1 lap times and optimizes race strategy using machine learning.

## Pipeline

```
download_data.py    → Downloads F1 session data via fastf1
clean_data.py       → Cleans and engineers features
predict_lap_time.py → Trains XGBoost model (baseline, ~0.58s MAE)
train_enhanced.py   → Optuna-tuned XGBoost + ensemble (~0.52s MAE)
f1_simulator.py     → Simulates a single stint with tire degradation
strategy_optimizer.py → Finds optimal pit strategy for a full race
```

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
cd src
python download_all_races.py   # downloads 2025 race data
python clean_data.py           # cleans + feature engineering
python train_enhanced.py       # trains tuned ensemble model
python strategy_optimizer.py   # finds optimal race strategy
```

## Results

- **Baseline XGBoost**: ~0.585s MAE on unseen tracks
- **Tuned XGBoost + features**: ~0.524s MAE (10% improvement)
- **Strategy optimizer**: evaluates 1-stop and 2-stop strategies to minimize total race time
