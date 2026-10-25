# Energy Consumption Forecasting System

## Project Overview

The Energy Consumption Forecasting System is a machine learning project designed to analyze historical household electricity consumption and forecast future energy usage. It provides an interactive Streamlit dashboard for exploring consumption patterns, evaluating machine learning models, viewing forecasts, and estimating energy budgets.

## Objectives

- Analyze historical electricity consumption data.
- Clean and prepare data for machine learning.
- Create features for time-based energy forecasting.
- Train and compare forecasting models.
- Evaluate model performance using MAE and RMSE.
- Forecast energy consumption for the next seven days.
- Estimate energy budgets and provide tariff-based recommendations.
- Present insights through an interactive dashboard.

## Features

1. **Dashboard** — View key energy consumption metrics and summaries.
2. **Historical Analysis** — Explore past electricity consumption patterns.
3. **Future Forecast** — View predicted consumption for the next seven days.
4. **Model Evaluation** — Compare machine learning model performance.
5. **Energy Budget** — Estimate electricity usage costs and budgets.
6. **Tariff & Recommendations** — Explore tariff-based cost estimates and energy-saving suggestions.

## Machine Learning Models

The project compares:

- Random Forest Regressor
- Linear Regression

The current evaluation results are:

| Model | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| Random Forest Regressor | 4.1394 | 5.5644 |
| Linear Regression | 4.2218 | 5.7870 |

Based on these evaluation results, the Random Forest model performed better on the evaluated test data.

MAE measures the average absolute prediction error. RMSE gives greater weight to larger errors.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib / visualization libraries used in the application
- Joblib
- Git and GitHub

## Project Structure

```text
Energy_Consumption_Forecasting/
├── app.py
├── requirements.txt
├── verify_project.py
├── .gitignore
├── data/
│   └── processed/
│       └── daily_energy_features.csv
├── models/
│   └── best_model.joblib
├── outputs/
│   ├── baseline_comparison.csv
│   ├── future_forecast.csv
│   ├── model_evaluation.csv
│   └── test_predictions.csv
└── src/
    ├── 01_data_understanding.py
    ├── 02_data_cleaning.py
    ├── 03_feature_engineering.py
    ├── 04_train_models.py
    ├── 05_baseline_comparison.py
    ├── 05_forecast.py
    └── 06_forecast_future.py
```

The raw and large intermediate datasets are excluded from the GitHub repository to keep the upload within GitHub's standard file-size limits. The complete datasets remain in the local project folder.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prasanthi716/Energy-Consumption-Forecasting.git
cd Energy-Consumption-Forecasting
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

**Note:** The application and training scripts may require the original dataset or intermediate data files that are excluded from this repository. Obtain the required dataset and run the relevant preprocessing and training scripts if those files are missing.

## Model Verification

Run the project verification script:

```powershell
python verify_project.py
```

This script can be used to check the availability of required project files and the saved model.

## Forecasting Scope and Limitations

The application produces a seven-day forecast based on the historical data and trained model available in the project. Forecast accuracy depends on the quality, coverage, and characteristics of the input data. Predictions are estimates and should not be treated as guaranteed future consumption.

## Future Enhancements

- Add live smart-meter data integration.
- Improve forecasting with additional time-series models.
- Add interactive comparisons across longer forecast periods.
- Support downloadable reports.
- Deploy the application to a cloud hosting platform.

## Author

**Prasanthi**

## Project Type

Academic Machine Learning and Data Analytics Project