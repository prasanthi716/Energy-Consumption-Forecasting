
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "household_power_cleaned.csv"

print("=" * 60)
print("ENERGY CONSUMPTION FORECASTING - DATA CLEANING")
print("=" * 60)

# Check that the original dataset exists
if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE}\n"
        "Check that the original file is inside data/raw."
    )

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Read the original dataset; '?' represents missing readings
print("\nReading original dataset...")
df = pd.read_csv(
    RAW_FILE,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print(f"Original rows: {len(df):,}")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove completely duplicated records
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print(f"Duplicate rows removed: {duplicate_count:,}")

# Convert date and time columns
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d/%m/%Y",
    errors="coerce"
)

df["Time"] = pd.to_timedelta(
    df["Time"],
    errors="coerce"
)

# Remove records with invalid dates or times
invalid_mask = df["Date"].isna() | df["Time"].isna()
invalid_count = int(invalid_mask.sum())
df = df.loc[~invalid_mask].copy()

print(f"Rows removed for invalid date/time: {invalid_count:,}")

# Combine date and time into one timestamp
df["timestamp"] = df["Date"] + df["Time"]

# Convert measurement columns to numeric values
measurement_columns = [
    column for column in df.columns
    if column not in ["Date", "Time", "timestamp"]
]

for column in measurement_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Report missing values before filling them
print("\nMissing values before handling:")
print(df[measurement_columns].isna().sum().to_string())

# Fill missing measurements using time interpolation.
# Limit interpolation to gaps of at most 5 consecutive records.
df = df.sort_values("timestamp").set_index("timestamp")

df[measurement_columns] = df[measurement_columns].interpolate(
    method="time",
    limit=5,
    limit_area="inside"
)

# Remaining missing readings are filled with each column's median.
# This is a simple fallback; we will assess data quality later.
for column in measurement_columns:
    if df[column].isna().any():
        median_value = df[column].median()
        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)

# Restore timestamp as a regular column
df = df.reset_index()

# Save cleaned data without modifying the raw file
df.to_csv(OUTPUT_FILE, index=False)

print("\nMissing values after handling:")
print(df[measurement_columns].isna().sum().to_string())

print(f"\nFinal rows: {len(df):,}")
print(f"Final columns: {len(df.columns)}")
print(f"\nCleaned dataset saved to:\n{OUTPUT_FILE}")

print("\nDATA CLEANING COMPLETED")
