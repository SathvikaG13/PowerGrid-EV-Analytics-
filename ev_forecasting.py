"""Data preparation, model backtesting, and EV charging demand forecasts."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ("energy_kwh", "sessions")
MODEL_NAMES = ("SARIMA", "Prophet", "XGBoost")
FEATURE_LAGS = (1, 7, 14, 28)


def load_and_aggregate(source: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load sessions from Excel and return complete hourly/daily demand series."""
    required = {"session_id", "start_timestamp", "energy_kwh"}
    sessions = pd.read_excel(
        source,
        sheet_name="sessions",
        usecols=lambda column: column in required,
    )
    missing = required.difference(sessions.columns)
    if missing:
        raise ValueError(f"Sessions sheet is missing required columns: {sorted(missing)}")

    original_rows = len(sessions)
    sessions["start_timestamp"] = pd.to_datetime(
        sessions["start_timestamp"], errors="coerce"
    )
    sessions["energy_kwh"] = pd.to_numeric(sessions["energy_kwh"], errors="coerce")
    sessions = sessions.dropna(subset=["session_id", "start_timestamp", "energy_kwh"])
    sessions = sessions.loc[sessions["energy_kwh"] >= 0]
    sessions = sessions.drop_duplicates(subset="session_id", keep="first")
    if sessions.empty:
        raise ValueError("No valid charging sessions remain after cleaning.")

    sessions = sessions.set_index("start_timestamp").sort_index()
    hourly = sessions.resample("h").agg(
        sessions=("session_id", "count"),
        energy_kwh=("energy_kwh", "sum"),
    )
    hourly["sessions"] = hourly["sessions"].astype("int64")
    hourly["energy_kwh"] = hourly["energy_kwh"].astype("float64")
    daily = hourly.resample("D").sum()
    daily["sessions"] = daily["sessions"].astype("int64")

    quality = {
        "input_rows": original_rows,
        "clean_rows": len(sessions),
        "excluded_rows": original_rows - len(sessions),
    }
    return hourly, daily, quality


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    day_of_year = index.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "day_of_week": index.dayofweek,
            "month": index.month,
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "trend": (index - index[0]).days,
        },
        index=index,
    )


def _xgboost_training_features(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    features = _calendar_features(series.index)
    for lag in FEATURE_LAGS:
        features[f"lag_{lag}"] = series.shift(lag)
    features["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    features["rolling_mean_28"] = series.shift(1).rolling(28).mean()
    usable = features.notna().all(axis=1)
    return features.loc[usable], series.loc[usable]


def _next_xgboost_features(
    history: list[float], date: pd.Timestamp, origin: pd.Timestamp
) -> pd.DataFrame:
    values: dict[str, float] = {}
    for lag in FEATURE_LAGS:
        values[f"lag_{lag}"] = history[-lag]
    values["rolling_mean_7"] = float(np.mean(history[-7:]))
    values["rolling_mean_28"] = float(np.mean(history[-28:]))
    day_of_year = date.dayofyear
    values.update(
        {
            "day_of_week": float(date.dayofweek),
            "month": float(date.month),
            "day_of_year_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
            "day_of_year_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
            "trend": float((date - origin).days),
        }
    )
    return pd.DataFrame([values])


def _make_xgboost_model():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def _fit_predict(
    model_name: str, history: pd.Series, future_index: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if model_name == "SARIMA":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        fitted = SARIMAX(
            history,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)
        prediction = fitted.get_forecast(steps=len(future_index))
        forecast = np.asarray(prediction.predicted_mean, dtype=float)
        interval = np.asarray(prediction.conf_int(), dtype=float)
        return forecast, interval[:, 0], interval[:, 1]

    if model_name == "Prophet":
        from prophet import Prophet

        training = pd.DataFrame(
            {"ds": history.index, "y": history.to_numpy(dtype=float)}
        )
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=200,
        )
        model.fit(training)
        prediction = model.predict(
            pd.DataFrame({"ds": future_index})
        )
        return (
            prediction["yhat"].to_numpy(dtype=float),
            prediction["yhat_lower"].to_numpy(dtype=float),
            prediction["yhat_upper"].to_numpy(dtype=float),
        )

    if model_name == "XGBoost":
        features, target = _xgboost_training_features(history)
        if features.empty:
            raise ValueError("At least 29 daily observations are needed for XGBoost.")
        model = _make_xgboost_model()
        model.fit(features, target)
        history_values = history.to_list()
        origin = history.index[0]
        forecasts = []
        for date in future_index:
            row = _next_xgboost_features(history_values, date, origin)
            value = max(0.0, float(model.predict(row[features.columns])[0]))
            forecasts.append(value)
            history_values.append(value)
        return np.asarray(forecasts), None, None

    raise ValueError(f"Unsupported model: {model_name}")


def _accuracy_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(predicted, 0.0)
    errors = prediction - actual.to_numpy(dtype=float)
    nonzero = actual.to_numpy(dtype=float) != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual.to_numpy(dtype=float)[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAPE": mape,
        "RMSE": float(np.sqrt(np.mean(errors**2))),
    }


def train_and_forecast(
    daily: pd.DataFrame, horizon: int = 365
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest each model on a chronological holdout, then forecast from all history."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day.")
    if len(daily) < 120:
        raise ValueError("At least 120 daily observations are required for reliable training.")

    holdout_size = min(90, max(14, len(daily) // 5))
    split = len(daily) - holdout_size
    metrics: list[dict[str, float | str | int]] = []
    forecasts: list[pd.DataFrame] = []

    for target_name in TARGETS:
        series = daily[target_name].astype("float64").asfreq("D", fill_value=0)
        train = series.iloc[:split]
        actual = series.iloc[split:]

        for model_name in MODEL_NAMES:
            backtest, _, _ = _fit_predict(model_name, train, actual.index)
            scores = _accuracy_metrics(actual, backtest)
            metrics.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "training_days": len(train),
                    "test_days": len(actual),
                    **scores,
                }
            )

            future_index = pd.date_range(
                series.index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq="D",
            )
            predicted, lower, upper = _fit_predict(model_name, series, future_index)
            predicted = np.maximum(predicted, 0.0)
            result = pd.DataFrame(
                {
                    "date": future_index,
                    "target": target_name,
                    "model": model_name,
                    "forecast": predicted,
                    "lower": np.maximum(lower, 0.0) if lower is not None else np.nan,
                    "upper": np.maximum(upper, 0.0) if upper is not None else np.nan,
                }
            )
            forecasts.append(result)

    return pd.DataFrame(metrics), pd.concat(forecasts, ignore_index=True)


def run_pipeline(
    source: str | Path, output_dir: str | Path, horizon: int = 365
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write cleaned time series, backtest metrics, and forecast CSVs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hourly, daily, quality = load_and_aggregate(source)
    hourly.to_csv(output / "cleaned_hourly_demand.csv", index_label="timestamp")
    daily.to_csv(output / "cleaned_daily_demand.csv", index_label="date")

    metrics, forecasts = train_and_forecast(daily, horizon)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    forecasts.to_csv(output / "forecasts.csv", index=False)
    pd.DataFrame([quality]).to_csv(output / "data_quality.csv", index=False)
    return metrics, forecasts


def seasonal_insights(daily: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    """Summarize weekday and month-of-year demand relative to the overall mean."""
    series = daily[target].astype(float)
    weekday = series.groupby(series.index.day_name()).mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    month = series.groupby(series.index.month).mean().reindex(range(1, 13))
    overall = float(series.mean())
    return {
        "weekday_index": weekday / overall * 100 if overall else weekday * 0,
        "month_index": month / overall * 100 if overall else month * 0,
    }
