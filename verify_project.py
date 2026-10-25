
from pathlib import Path
import pandas as pd
import joblib

required_files = [
    "models/best_model.joblib",
    "outputs/model_evaluation.csv",
    "outputs/test_predictions.csv",
    "outputs/future_forecast.csv",
    "data/processed/daily_energy_features.csv",
]

print("\nENERGY FORECASTING - FILE VERIFICATION\n")

for file in required_files:
    path = Path(file)
    print(f"{'PASS' if path.exists() else 'FAIL'}: {file}")

if all(Path(f).exists() for f in required_files):
    model = joblib.load("models/best_model.joblib")
    evaluation = pd.read_csv("outputs/model_evaluation.csv")
    predictions = pd.read_csv("outputs/test_predictions.csv")
    forecast = pd.read_csv("outputs/future_forecast.csv")
    features = pd.read_csv("data/processed/daily_energy_features.csv")

    print("\nMODEL EVALUATION")
    print(evaluation.to_string(index=False))

    print("\nDATA VALIDATION")
    print("Feature records:", len(features))
    print("Test prediction records:", len(predictions))
    print("Future forecast records:", len(forecast))
    print("Missing forecast values:", forecast.isna().sum().sum())

    print("\nFORECAST PREVIEW")
    print(forecast.head(7).to_string(index=False))

    print("\nMODEL LOADED SUCCESSFULLY:", type(model).__name__)
else:
    print("\nSome required files are missing. Check the FAIL entries.")
