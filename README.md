# EV Charging Demand Forecasting

This project forecasts daily charging energy (`energy_kwh`) and session volume from the `sessions` sheet of `ev_charging_dataset.xlsx`. It requires Python 3.10 or later.

## Run the pipeline

```powershell
python -m pip install -r requirements.txt
python train_forecasts.py
```

The pipeline cleans invalid and duplicate sessions, fills inactive time buckets with zero demand, and writes these artifacts to `outputs/`:

- `cleaned_hourly_demand.csv` and `cleaned_daily_demand.csv`
- `model_metrics.csv` with chronological holdout MAPE and RMSE
- `forecasts.csv` with model forecasts and available prediction intervals
- `data_quality.csv` with included and excluded source-row counts

The final 90 days are used for validation (or a bounded smaller holdout for shorter histories). MAPE excludes holdout days where actual demand is zero. Forecasts are clipped at zero because negative charging demand is not meaningful.

## Models

- **SARIMA**: `(1,1,1) × (1,1,1,7)` with a weekly seasonal cycle.
- **Prophet**: weekly and yearly seasonalities.
- **XGBoost**: recursive daily forecasts using lag, rolling-average, calendar, and trend features.

## Dashboard

After generating the outputs, start the interactive dashboard:
To view the dashboard, click on link below
https://bmhpcj4ygeekhwui7igkps.streamlit.app/
To view locally:
```powershell
python -m streamlit run dashboard.py
```

Select energy or session volume, compare models, and view 30-, 90-, or 365-day forecasts, backtest accuracy, and weekday/month-of-year patterns. The source workbook is kept unchanged.

Forecast dates begin the day after the last date in the workbook. The supplied data ends on 2024-12-31, so refresh it with newer sessions before using the forecasts as current operational guidance.
