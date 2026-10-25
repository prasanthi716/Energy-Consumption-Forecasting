
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "household_power_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "daily_energy_features.csv"

print("=" * 60)
print("ENERGY CONSUMPTION FORECASTING")
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# Check input file
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {INPUT_FILE}\n"
        "Run src/02_data_cleaning.py first."
    )

# Load cleaned dataset
print("\nLoading cleaned dataset...")
df = pd.read_csv(INPUT_FILE)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# Ensure active power is numeric
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df = df.dropna(subset=["Global_active_power"])

# Active power is measured in kilowatts (kW).
# Each reading represents approximately one minute.
# Therefore, daily energy (kWh) = sum of minute readings / 60.
print("\nCalculating daily energy consumption...")

daily = (
    df.set_index("timestamp")["Global_active_power"]
    .resample("D")
    .sum()
    .div(60)
    .to_frame(name="daily_consumption_kwh")
)

# Exclude days without any readings
daily = daily.dropna()

# Create calendar features
daily["day_of_week"] = daily.index.dayofweek
daily["day_of_month"] = daily.index.day
daily["month"] = daily.index.month
daily["day_of_year"] = daily.index.dayofyear
daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)

# Create historical lag features.
# These use only previous days, not future consumption.
daily["lag_1_day"] = daily["daily_consumption_kwh"].shift(1)
daily["lag_7_days"] = daily["daily_consumption_kwh"].shift(7)

# Rolling mean is shifted first to avoid using the current day's target.
daily["rolling_mean_7_days"] = (
    daily["daily_consumption_kwh"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

# Save a copy for later model training.
# Initial rows without enough history will be removed.
daily = daily.dropna().reset_index()
daily = daily.rename(columns={"timestamp": "date"})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUTPUT_FILE, index=False)

print("\nFEATURE ENGINEERING RESULTS")
print(f"Daily records: {len(daily):,}")
print(f"Total columns: {len(daily.columns)}")

print("\nColumns created:")
for column in daily.columns:
    print("-", column)

print("\nFirst five rows:")
print(daily.head().to_string(index=False))

print(f"\nSaved daily features to:\n{OUTPUT_FILE}")
print("\nFEATURE ENGINEERING COMPLETED")
