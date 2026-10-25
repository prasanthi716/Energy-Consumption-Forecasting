
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "daily_energy_features.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ENERGY CONSUMPTION FORECASTING")
print("STEP 4: MODEL TRAINING AND EVALUATION")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_FILE}\n"
        "Run src/03_feature_engineering.py first."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

target = "daily_consumption_kwh"

features = [
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "is_weekend",
    "lag_1_day",
    "lag_7_days",
    "rolling_mean_7_days",
]

X = df[features]
y = df[target]

# Split chronologically: earlier dates for training,
# later dates for testing.
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["date"].iloc[split_index:]

print(f"\nTotal records: {len(df):,}")
print(f"Training records: {len(X_train):,}")
print(f"Testing records: {len(X_test):,}")
print(f"Training ends: {df['date'].iloc[split_index - 1].date()}")
print(f"Testing starts: {test_dates.iloc[0].date()}")

models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    results.append({
        "model": name,
        "MAE_kWh": mae,
        "RMSE_kWh": rmse,
    })

    trained_models[name] = model

    print(f"MAE:  {mae:.4f} kWh")
    print(f"RMSE: {rmse:.4f} kWh")

# Select the model with the lowest test MAE.
results_df = pd.DataFrame(results).sort_values("MAE_kWh")
results_df.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)

best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

# Save actual and predicted consumption for later analysis.
best_predictions = best_model.predict(X_test)

predictions_df = pd.DataFrame({
    "date": test_dates.to_numpy(),
    "actual_consumption_kwh": y_test.to_numpy(),
    "predicted_consumption_kwh": best_predictions,
})

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

print(f"\nBest model by test MAE: {best_name}")
print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
print(f"Saved metrics: {OUTPUT_DIR / 'model_evaluation.csv'}")
print(f"Saved predictions: {OUTPUT_DIR / 'test_predictions.csv'}")
print("\nMODEL TRAINING COMPLETED")
