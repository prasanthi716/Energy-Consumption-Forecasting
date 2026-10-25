
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "daily_energy_features.csv"
METRICS_FILE = BASE_DIR / "outputs" / "model_evaluation.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "baseline_comparison.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

target = "daily_consumption_kwh"

split_index = int(len(df) * 0.8)
test = df.iloc[split_index:].copy()

actual = test[target]

baselines = {
    "Previous-day baseline": test["lag_1_day"],
    "Previous-week baseline": test["lag_7_days"],
}

results = []

for model_name, predictions in baselines.items():
    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5

    results.append({
        "model": model_name,
        "MAE_kWh": mae,
        "RMSE_kWh": rmse,
    })

    print(f"\n{model_name}")
    print(f"MAE:  {mae:.4f} kWh")
    print(f"RMSE: {rmse:.4f} kWh")

if METRICS_FILE.exists():
    model_metrics = pd.read_csv(METRICS_FILE)

    results.extend(model_metrics.to_dict("records"))
else:
    print("\nModel evaluation file not found.")
    print("Run src/04_train_models.py first.")

comparison = pd.DataFrame(results)
comparison = comparison.sort_values("MAE_kWh")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 55)
print("BASELINE AND MODEL COMPARISON")
print("=" * 55)
print(comparison.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE}")
print("\nBASELINE COMPARISON COMPLETED")
