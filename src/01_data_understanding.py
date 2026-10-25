
from pathlib import Path
import pandas as pd

# Locate the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the original UCI dataset
DATA_FILE = BASE_DIR / "data" / "raw" / "household_power_consumption.txt"

print("=" * 60)
print("ENERGY CONSUMPTION FORECASTING")
print("STEP 1: DATASET UNDERSTANDING")
print("=" * 60)

# Check whether the dataset exists
if not DATA_FILE.exists():
    print("\nERROR: Dataset file was not found.")
    print("Expected location:")
    print(DATA_FILE)
    print("\nPlease check the filename and data/raw folder.")
    raise SystemExit(1)

print(f"\nDataset found: {DATA_FILE.name}")

# Read the UCI dataset
print("\nReading dataset... Please wait.")

df = pd.read_csv(
    DATA_FILE,
    sep=";",
    na_values=["?"],
    low_memory=False
)

# Display basic information
print("\n1. DATASET SHAPE")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\n2. COLUMN NAMES")
for column in df.columns:
    print("-", column)

print("\n3. FIRST FIVE ROWS")
print(df.head().to_string(index=False))

print("\n4. DATA TYPES")
print(df.dtypes.to_string())

print("\n5. MISSING VALUES")
print(df.isnull().sum().to_string())

print("\n6. DUPLICATE ROWS")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\n7. DATE RANGE")
if "Date" in df.columns:
    dates = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )
    print(f"First date: {dates.min()}")
    print(f"Last date:  {dates.max()}")
    print(f"Invalid dates: {dates.isna().sum():,}")

print("\n8. NUMERIC SUMMARY")
print(df.describe().to_string())

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)
