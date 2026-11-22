"""Streamlit dashboard for EV charging demand forecasts."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ev_forecasting import seasonal_insights

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

st.set_page_config(page_title="EV Charging Demand Forecast", layout="wide")
st.title("EV Charging Demand Forecast")
st.caption("Daily charging demand forecasts for capacity, staffing, and energy planning.")

required_files = [
    OUTPUT / "cleaned_daily_demand.csv",
    OUTPUT / "model_metrics.csv",
    OUTPUT / "forecasts.csv",
]
missing = [path.name for path in required_files if not path.is_file()]
if missing:
    st.info(
        "Forecast artifacts are missing. Install the requirements and run "
        "`python train_forecasts.py` before opening the dashboard."
    )
    st.code("pip install -r requirements.txt\npython train_forecasts.py\nstreamlit run dashboard.py")
    st.stop()

daily = pd.read_csv(required_files[0], parse_dates=["date"], index_col="date")
metrics = pd.read_csv(required_files[1])
forecasts = pd.read_csv(required_files[2], parse_dates=["date"])
st.warning(
    f"Source data ends on {daily.index.max():%Y-%m-%d}; forecasts start the following day. "
    "Refresh the source workbook before using these dates for current operational planning."
)

target_options = {
    "Energy (kWh)": "energy_kwh",
    "Charging sessions": "sessions",
}
selection = st.sidebar.selectbox("Demand measure", list(target_options))
target = target_options[selection]
horizon = st.sidebar.select_slider("Forecast horizon", options=[30, 90, 365], value=90)

target_forecasts = forecasts.loc[forecasts["target"] == target]
available_models = target_forecasts["model"].drop_duplicates().tolist()
models = st.sidebar.multiselect(
    "Models",
    available_models,
    default=available_models,
)
forecast_start = target_forecasts["date"].min()
visible = target_forecasts.loc[
    target_forecasts["model"].isin(models)
    & (target_forecasts["date"] < forecast_start + pd.Timedelta(days=horizon))
]

if visible.empty:
    st.warning("Select at least one model to display forecasts.")
    st.stop()

latest_actual_date = daily.index.max()
actual = daily.loc[daily.index >= latest_actual_date - pd.Timedelta(days=89), target]
history_plot = pd.DataFrame(
    {"date": actual.index, "value": actual.to_numpy(), "series": "Observed"}
)
forecast_plot = visible[["date", "forecast", "model"]].rename(
    columns={"forecast": "value", "model": "series"}
)
chart_data = pd.concat([history_plot, forecast_plot], ignore_index=True)
figure = px.line(
    chart_data,
    x="date",
    y="value",
    color="series",
    labels={"date": "Date", "value": selection, "series": "Series"},
    title=f"{selection}: observed demand and {horizon}-day forecast",
)
st.plotly_chart(figure, width="stretch")

metric_view = metrics.loc[
    (metrics["target"] == target) & metrics["model"].isin(models)
].sort_values("RMSE")
st.subheader("Chronological holdout accuracy")
st.caption("Metrics use the final 90 (or available holdout) days; MAPE excludes zero-actual days.")
st.dataframe(
    metric_view[["model", "training_days", "test_days", "MAPE", "RMSE"]].style.format(
        {"MAPE": "{:.2f}%", "RMSE": "{:.2f}"}
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Weekly and seasonal patterns")
insights = seasonal_insights(daily, target)
weekday, month = st.columns(2)
with weekday:
    weekday_data = insights["weekday_index"].rename("Demand index").rename_axis("Weekday").reset_index()
    st.plotly_chart(
        px.bar(weekday_data, x="Weekday", y="Demand index", title="Weekday demand (average = 100)"),
        width="stretch",
    )
with month:
    month_data = insights["month_index"].rename("Demand index").rename_axis("Month").reset_index()
    st.plotly_chart(
        px.line(
            month_data,
            x="Month",
            y="Demand index",
            markers=True,
            title="Month-of-year demand (average = 100)",
        ),
        width="stretch",
    )

weekday_values = insights["weekday_index"].dropna()
month_values = insights["month_index"].dropna()
if not weekday_values.empty and not month_values.empty:
    st.write(
        f"Peak weekday: **{weekday_values.idxmax()}** "
        f"({weekday_values.max():.1f} index). "
        f"Peak month: **{month_values.idxmax()}** ({month_values.max():.1f} index)."
    )

with st.expander("Forecast values"):
    st.dataframe(
        visible.pivot(index="date", columns="model", values="forecast").reset_index(),
        hide_index=True,
        width="stretch",
    )
