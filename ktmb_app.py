import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import altair as alt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# Data & forecasting functions
# ----------------------------
def get_ktmb_data():
    URL_DATA = "https://storage.data.gov.my/transportation/ktmb/ridership_ktmb_daily.parquet"
    df = pd.read_parquet(URL_DATA)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def data_by_service(df):
    df = df.copy()

    service = df.loc[0, "service"]
    ridership = df["ridership"].tolist()

    df["predict_ridership"] = df["ridership"]

    # save last 5 data
    last5_data = ridership[-5:] if len(ridership) >= 5 else ridership

    last_date = datetime.strptime(df["date"].iloc[-1], "%Y-%m-%d")

    for i in range(1, 6):
        pdate = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")

        avg_ridership = round(sum(last5_data[-5:]) / len(last5_data), 0)
        last5_data.append(avg_ridership)

        df.loc[len(df)] = {
            "date": pdate,
            "service": service,
            "ridership": float("nan"),
            "predict_ridership": avg_ridership,
        }

    return df


def sarima_forecast(df, steps=7):
    df = df.copy()

    df_actual_data = df[df["ridership"].notna()].copy()
    df_actual_data["date"] = pd.to_datetime(df_actual_data["date"])
    df_actual_data.set_index("date", inplace=True)

    # daily frequency
    df_actual_data = df_actual_data.asfreq("D")

    model = SARIMAX(
        df_actual_data["ridership"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
    )

    results = model.fit(disp=False)
    forecast = results.forecast(steps=steps)

    future_dates = pd.date_range(
        start=df_actual_data.index[-1] + timedelta(days=1),
        periods=steps,
    )

    df_forecast = pd.DataFrame(
        {
            "date": future_dates,
            "sarima_forecast": forecast.values,
        }
    )

    df_forecast["sarima_forecast"] = df_forecast["sarima_forecast"].round(2)

    return df_forecast


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("KTMB Ridership Forecast")

# Load data
df_ridership = get_ktmb_data()

# Service selection
service_list = df_ridership["service"].unique().tolist()
selected_service = st.selectbox("Select Service", service_list)

# Filter by service
df_service = df_ridership[df_ridership["service"] == selected_service].copy()
df_service.reset_index(drop=True, inplace=True)

# Forecast extension (moving average)
df_result = data_by_service(df_service)

# Type conversions
df_result["date"] = pd.to_datetime(df_result["date"])
df_result["ridership"] = pd.to_numeric(df_result["ridership"], errors="coerce")
df_result["predict_ridership"] = pd.to_numeric(df_result["predict_ridership"], errors="coerce")

# Date range filter
min_date = df_result["date"].min()
max_date = df_result["date"].max()

date_range = st.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_result[
        (df_result["date"] >= pd.to_datetime(start_date))
        & (df_result["date"] <= pd.to_datetime(end_date))
    ]
else:
    df_filtered = df_result

# ----------------------------
# Moving average chart
# ----------------------------
st.subheader("Ridership Moving Average")

df_a = df_filtered[df_filtered["ridership"].notna()].copy()
df_a["type"] = "Actual"
df_a = df_a[["date", "ridership", "type"]]

df_b = df_filtered[df_filtered["ridership"].isna()].copy()
df_b["type"] = "Moving average"
df_b["ridership"] = df_b["predict_ridership"].copy()
df_b = df_b[["date", "ridership", "type"]]

df_moving_average = pd.concat([df_a, df_b])

ma_chart = (
    alt.Chart(df_moving_average)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("ridership:Q", title="Ridership"),
        color=alt.Color("type:N", title="Legend"),
        strokeDash=alt.condition(
            "datum.type == 'Moving average'",
            alt.value([5, 5]),
            alt.value([0]),
        ),
        tooltip=["date:T", "ridership:Q", "type:N"],
    )
    .interactive()
)

st.altair_chart(ma_chart, use_container_width=True)

# ----------------------------
# SARIMA forecast chart
# ----------------------------
st.subheader("Ridership SARIMA Forecast")

df_sarima = sarima_forecast(df_filtered)

df_sActual = df_filtered[df_filtered["ridership"].notna()][["date", "ridership"]]
df_sActual["type"] = "Actual"

df_sarima_plot = df_sarima.copy()
df_sarima_plot.rename(columns={"sarima_forecast": "ridership"}, inplace=True)
df_sarima_plot["type"] = "SARIMA Forecast"

df_combined = pd.concat([df_sActual, df_sarima_plot])
df_combined.reset_index(drop=True, inplace=True)

sarima_chart = (
    alt.Chart(df_combined)
    .mark_line(interpolate="basis-closed")
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("ridership:Q", title="Ridership"),
        color=alt.Color("type:N", title="Legend"),
        strokeDash=alt.condition(
            "datum.type == 'SARIMA Forecast'",
            alt.value([5, 5]),
            alt.value([0]),
        ),
        tooltip=["date:T", "ridership:Q", "type:N"],
    )
    .interactive()
)

st.altair_chart(sarima_chart, use_container_width=True)

# ----------------------------
# Regression trend
# ----------------------------
st.subheader("Regression Trend (Actual Data)")

df_actual = df_filtered[df_filtered["ridership"].notna()].copy()

regression_chart = (
    alt.Chart(df_actual)
    .mark_point(color="green")
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("ridership:Q", title="Ridership"),
        tooltip=["date:T", "ridership:Q"],
    )
    + alt.Chart(df_actual)
    .transform_regression("date", "ridership")
    .mark_line(color="red")
    .encode(
        x="date:T",
        y="ridership:Q",
    )
    .interactive()
)

st.altair_chart(regression_chart, use_container_width=True)

# ----------------------------
# Data table
# ----------------------------
st.subheader("Data Table")

df_filtered_ = df_filtered[["date", "service", "ridership"]].copy()
df_filtered_["date"] = pd.to_datetime(df_filtered_["date"]).dt.strftime("%Y-%m-%d")
df_filtered_ = df_filtered_[df_filtered_["ridership"].notna()]

st.dataframe(df_filtered_, hide_index=True)
