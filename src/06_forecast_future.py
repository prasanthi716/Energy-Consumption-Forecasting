from pathlib import Path
import math

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "daily_energy_features.csv"
MODEL_FILE = BASE_DIR / "models" / "best_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "future_forecast.csv"

FORECAST_DAYS = 7

print("=" * 60)
print("ENERGY CONSUMPTION FORECASTING")
print("FUTURE ELECTRICITY CONSUMPTION PREDICTION")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_FILE}\n"
        "Run src/03_feature_engineering.py first."
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_FILE}\n"
        "Run src/04_train_models.py first."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

if df.empty:
    raise ValueError("The feature dataset contains no records.")

if df["date"].isna().any():
    raise ValueError("The feature dataset contains invalid dates.")

if df["date"].duplicated().any():
    raise ValueError("The feature dataset contains duplicate dates.")

if "daily_consumption_kwh" not in df.columns:
    raise ValueError(
        "Target column daily_consumption_kwh is missing."
    )

model = joblib.load(MODEL_FILE)

if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "The model does not expose its training feature names. "
        "Check src/04_train_models.py before forecasting."
    )

feature_columns = list(model.feature_names_in_)

required_columns = {
    "date",
    "daily_consumption_kwh",
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "is_weekend",
    "lag_1_day",
    "lag_7_days",
    "rolling_mean_7_days",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

missing_model_features = set(feature_columns).difference(df.columns)

if missing_model_features:
    raise ValueError(
        "The dataset is missing model training features: "
        f"{sorted(missing_model_features)}"
    )

history = df["daily_consumption_kwh"].astype(float).tolist()
last_date = df["date"].iloc[-1]

if len(history) < 7:
    raise ValueError(
        "At least seven days of historical consumption are required."
    )

if not all(math.isfinite(value) and value >= 0 for value in history):
    raise ValueError(
        "Historical consumption contains invalid or negative values."
    )

forecast_rows = []

print(f"\nLast historical date: {last_date.date()}")
print(f"Forecast horizon: {FORECAST_DAYS} days")
print("\nGenerating predictions...")

for step in range(1, FORECAST_DAYS + 1):
    forecast_date = last_date + pd.Timedelta(days=step)

    features = {
        "day_of_week": forecast_date.dayofweek,
        "day_of_month": forecast_date.day,
        "month": forecast_date.month,
        "day_of_year": forecast_date.dayofyear,
        "is_weekend": int(forecast_date.dayofweek >= 5),
        "lag_1_day": history[-1],
        "lag_7_days": history[-7],
        "rolling_mean_7_days": sum(history[-7:]) / 7,
    }

    missing_features = set(feature_columns).difference(features)

    if missing_features:
        raise ValueError(
            "The trained model expects features that this script "
            f"does not generate: {sorted(missing_features)}. "
            "Check src/04_train_models.py."
        )

    input_data = pd.DataFrame(
        [{column: features[column] for column in feature_columns}],
        columns=feature_columns,
    )

    prediction = float(model.predict(input_data)[0])

    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError(
            f"Invalid predicted consumption for "
            f"{forecast_date.date()}: {prediction}"
        )

    forecast_rows.append({
        "date": forecast_date,
        "predicted_consumption_kwh": prediction,
    })

    history.append(prediction)

forecast_df = pd.DataFrame(forecast_rows)

if len(forecast_df) != FORECAST_DAYS:
    raise ValueError(
        f"Expected {FORECAST_DAYS} forecast rows, "
        f"but generated {len(forecast_df)}."
    )

if forecast_df["predicted_consumption_kwh"].isna().any():
    raise ValueError("Forecast contains missing predictions.")

if not forecast_df["date"].is_monotonic_increasing:
    raise ValueError("Forecast dates are not in chronological order.")

if forecast_df["date"].iloc[0] <= last_date:
    raise ValueError(
        "The forecast must begin after the last historical date."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
forecast_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("SEVEN-DAY FORECAST")
print("=" * 60)

display_df = forecast_df.copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df["predicted_consumption_kwh"] = (
    display_df["predicted_consumption_kwh"].round(2)
)

print(display_df.to_string(index=False))

print(
    "\nTotal predicted consumption: "
    f"{forecast_df['predicted_consumption_kwh'].sum():.2f} kWh"
)

print(
    "Average predicted daily consumption: "
    f"{forecast_df['predicted_consumption_kwh'].mean():.2f} kWh"
)

print(
    "Minimum predicted daily consumption: "
    f"{forecast_df['predicted_consumption_kwh'].min():.2f} kWh"
)

print(
    "Maximum predicted daily consumption: "
    f"{forecast_df['predicted_consumption_kwh'].max():.2f} kWh"
)

print(f"\nForecast saved to: {OUTPUT_FILE}")
print("\nFUTURE FORECAST COMPLETED")
