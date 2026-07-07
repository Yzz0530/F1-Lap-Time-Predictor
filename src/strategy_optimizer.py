import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

class F1StrategyOptimizer:
    COMPOUND_SPEED = {"SOFT": -0.35, "MEDIUM": 0.0, "HARD": 0.20}
    TIRE_DEG_RATE = {"SOFT": 0.08, "MEDIUM": 0.045, "HARD": 0.025}
    FUEL_BURN = -0.020
    OUT_LAP_PENALTY = 0.5
    PIT_LOSS = 22.0

    def __init__(self):
        base = "../models"
        self.le_driver = joblib.load(f"{base}/le_driver_master.pkl")
        self.le_compound = joblib.load(f"{base}/le_compound_master.pkl")
        self.le_family = joblib.load(f"{base}/le_family_master.pkl")
        self.xgb_model = joblib.load(f"{base}/xgb_master.pkl")

        df = pd.read_csv(f"{base}/../data/all_races_master.csv")
        df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")
        self.race_baselines = df.groupby("Race")["RaceBaseline"].first().to_dict()

        overall_avg = df["LapTime"].mean()
        self.driver_offsets = {}
        for d in df["Driver"].unique():
            avg = df[df["Driver"] == d]["LapTime"].mean()
            self.driver_offsets[d] = avg - overall_avg

    def lap_time(self, driver, compound, lap_in_stint, lap_number, total_stint_laps, 
                 noise=0.0, sc_active=False):
        base_speed = self.COMPOUND_SPEED[compound]
        tire_deg = self.TIRE_DEG_RATE[compound] * (lap_in_stint - 1)
        fuel = self.FUEL_BURN * (lap_number - 1)
        out_lap = self.OUT_LAP_PENALTY if lap_in_stint == 1 else 0
        driver_off = self.driver_offsets.get(driver, 0)
        sc_delta = 3.0 if sc_active else 0.0
        return base_speed + tire_deg + fuel + out_lap + driver_off + sc_delta + noise

    def simulate_strategy(self, race_name, total_laps, driver, strategy, 
                          pit_loss=None, rng=None, sc_prob=0.0):
        if race_name not in self.race_baselines:
            return None
        baseline = self.race_baselines[race_name]
        if driver not in self.driver_offsets:
            return None

        if rng is None:
            rng = np.random.RandomState()
        if pit_loss is None:
            pit_loss = self.PIT_LOSS

        total_time = 0
        stint_details = []
        lap_number = 1
        sc_next_lap = None
        if rng.random() < sc_prob:
            sc_next_lap = rng.randint(5, max(15, total_laps // 3))

        for stint_idx, (compound, stint_laps) in enumerate(strategy):
            times = []
            for lis in range(1, stint_laps + 1):
                sc_active = (sc_next_lap == lap_number)
                # Reset SC timer if it happened
                if sc_active:
                    sc_next_lap = None

                noise = rng.normal(0, 0.08)
                delta = self.lap_time(driver, compound, lis, lap_number, stint_laps,
                                      noise=noise, sc_active=sc_active)
                lt = baseline + delta
                times.append(lt)
                total_time += lt
                lap_number += 1

            stint_details.append({
                "compound": compound, "laps": stint_laps,
                "avg_time": np.mean(times), "lap_times": times
            })
            if stint_idx < len(strategy) - 1:
                total_time += pit_loss

        return {
            "race": race_name, "driver": driver,
            "total_laps": total_laps, "stints": len(strategy),
            "strategy": [(c, l) for c, l in strategy],
            "total_time": total_time,
            "pit_loss_total": pit_loss * (len(strategy) - 1),
            "stint_details": stint_details,
            "sc_deployed": sc_next_lap is None and self._had_sc(strategy, total_laps, rng)
        }

    def _had_sc(self, strategy, total_laps, rng):
        return rng.random() < 0.2

    def optimize(self, race_name, total_laps, driver, mc_runs=1, sc_prob=0.0):
        compounds = ["SOFT", "MEDIUM", "HARD"]
        step = max(3, total_laps // 15)
        strategies = []

        for c1 in compounds:
            for c2 in compounds:
                for s1 in range(step, total_laps - step, step):
                    strategies.append([(c1, s1), (c2, total_laps - s1)])

        for c1 in compounds[:2]:
            for c2 in compounds[:2]:
                for c3 in compounds:
                    for s1 in range(step, total_laps - 2 * step, 2 * step):
                        for s2 in range(s1 + step, total_laps - step, 2 * step):
                            s3 = total_laps - s1 - s2
                            if s3 >= step:
                                strategies.append([(c1, s1), (c2, s2), (c3, s3)])

        results = []
        for strat in strategies:
            times = []
            for _ in range(mc_runs):
                rng = np.random.RandomState()
                r = self.simulate_strategy(race_name, total_laps, driver, strat,
                                           rng=rng, sc_prob=sc_prob)
                if r:
                    times.append(r["total_time"])
            if times:
                results.append({
                    "strategy": strat,
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "stints": len(strat)
                })

        results.sort(key=lambda r: r["mean_time"])
        return results[:10]

    def strategy_summary(self, race_name, total_laps, driver, strategy):
        r = self.simulate_strategy(race_name, total_laps, driver, strategy)
        if not r:
            return None
        total_sec = r["total_time"]
        mins, secs = divmod(int(total_sec), 60)
        lines = [f"Strategy: {' -> '.join([f'{c} ({l}l)' for c,l in strategy])}",
                 f"Total: {total_sec:.1f}s ({mins}:{secs:02d}) | Stops: {r['stints']}"]
        for s in r["stint_details"]:
            first, last = s["lap_times"][0], s["lap_times"][-1]
            deg = last - first
            lines.append(f"  {s['compound']:8s} {s['laps']:2d}l avg {s['avg_time']:.3f}s  {first:.3f}->{last:.3f} deg:{deg:+.3f}")
        return "\n".join(lines)

    def get_detailed_run(self, race_name, total_laps, driver, strategy):
        return self.simulate_strategy(race_name, total_laps, driver, strategy)

if __name__ == "__main__":
    import time
    t0 = time.time()

    opt = F1StrategyOptimizer()
    print("Available drivers:", sorted(opt.driver_offsets.keys()))
    print()

    race, laps, driver = "British Grand Prix", 52, "VER"
    print(f"Optimizing (MC=100, SC=20%) for {driver} @ {race} ({laps} laps)...")
    top = opt.optimize(race, laps, driver, mc_runs=100, sc_prob=0.20)

    print(f"\nTOP 5 (with uncertainty)")
    print(f"{'='*70}")
    for i, r in enumerate(top[:5], 1):
        strat = " -> ".join([f"{c} ({l}l)" for c, l in r["strategy"]])
        mins, secs = divmod(int(r["mean_time"]), 60)
        print(f"#{i} {strat}")
        print(f"   Time: {r['mean_time']:.1f}s ({mins}:{secs:02d}) +/- {r['std_time']:.2f}s  "
              f"[{r['min_time']:.1f} - {r['max_time']:.1f}]")
    print(f"\nEvaluated in {time.time() - t0:.1f}s")
