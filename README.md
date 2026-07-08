# F1 Lap Time Predictor

Predicts F1 lap times and optimizes race strategy using XGBoost + Monte Carlo simulation.

## Pipeline

```
download_all_races.py   → Downloads F1 session data via fastf1
prepare_enhanced_data.py → Cleans and engineers features (circuit, weather, sector speeds)
train.py                → Optuna-tuned XGBoost (absolute lap time, ~1.01s MAE)
strategy_optimizer.py   → Monte Carlo strategy simulation (physics + ML blend)
dashboard.py            → Streamlit UI for strategy optimization & driver comparison
```

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
python src/download_all_races.py
python src/prepare_enhanced_data.py
python src/train.py
streamlit run src/dashboard.py
```

## Results

- **XGBoost (absolute lap time)**: ~1.01s MAE on unseen tracks (31 features)
- **70/30 ML-physics blend**: lap-time prediction for both known and new circuits
- **Strategy optimizer**: evaluates 1-stop and 2-stop strategies with safety car and DNF simulation
- **Data**: 25 races (2025 + 2026), 24 drivers, 25,697 laps
