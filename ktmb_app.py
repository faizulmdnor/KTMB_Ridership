import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import altair as alt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# Forecast function
# ----------------------------
# SARIMA forecasting function
def sarima_forecast(df, steps=7):
    df = df.copy()

    df_actual_data = df[df['ridership'].notna()].copy()

    df_actual_data['date'] = pd.to_datetime(df_actual_data['date'])
    df_actual_data.set_index('date', inplace=True)

    df_actual_data = df_actual_data.asfreq('D')

    model = SARIMAX(
        df_actual_data['ridership'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    results = model.fit(disp=False)
    forecast = results.forecast(steps=steps)

    future_dates = pd.date_range(
        start=df_actual_data.index[-1] + timedelta(days=1),
        periods=steps
    )

    df_forecast = pd.DataFrame({
        'date': future_dates,
        'sarima_forecast': forecast.values
    })

    df_forecast['sarima_forecast'] = round(df_forecast['sarima_forecast'], 2)

    return df_forecast

def data_by_service(df):
    df = df.copy()

    service = df.loc[0, 'service']
    ridership = df['ridership'].tolist()

    df['predict_ridership'] = df['ridership']

    # safe last 5 data
    last5_data = ridership[-5:] if len(ridership) >= 5 else ridership

    last_date = datetime.strptime(df['date'].iloc[-1], '%Y-%m-%d')

    count = 0

    while count <= 4:
        pdate = (last_date + timedelta(days=count+1)).strftime('%Y-%m-%d')

        avg_ridership = round(sum(last5_data[-5:]) / len(last5_data), 0)
        last5_data.append(avg_ridership)

        df.loc[len(df)] = {
            'date': pdate,
            'service': service,
            'ridership': float('nan'),
            'predict_ridership': avg_ridership
        }

        count += 1

    return df


def get_ktmb_data():
    URL_DATA = 'https://storage.data.gov.my/transportation/ktmb/ridership_ktmb_daily.parquet'
    df = pd.read_parquet(URL_DATA)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("KTMB Ridership Forecast")

df_ridership = get_ktmb_data()

# Dropdown service
service_list = df_ridership['service'].unique().tolist()
selected_service = st.selectbox("Select Service", service_list)

# Filter data
df_service = df_ridership[df_ridership['service'] == selected_service].copy()
df_service.reset_index(drop=True, inplace=True)

# Forecast
df_result = data_by_service(df_service)

# Convert types
df_result['date'] = pd.to_datetime(df_result['date'])
df_result['ridership'] = pd.to_numeric(df_result['ridership'], errors='coerce')
df_result['predict_ridership'] = pd.to_numeric(df_result['predict_ridership'], errors='coerce')

# Convert date (ensure datetime)
df_result['date'] = pd.to_datetime(df_result['date'])

# Get latest month
latest_date = df_result['date'].max()
latest_month = latest_date.month
latest_year = latest_date.year

# Filter latest month
df_latest_month = df_result[
    (df_result['date'].dt.month == latest_month) &
    (df_result['date'].dt.year == latest_year)
]

# Date range filter
min_date = df_result['date'].min()
max_date = df_result['date'].max()

date_range = st.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_result[
        (df_result['date'] >= pd.to_datetime(start_date)) &
        (df_result['date'] <= pd.to_datetime(end_date))
    ]
else:
    df_filtered = df_result

# Base chart
base = alt.Chart(df_filtered)

df_a = df_filtered[df_filtered['ridership'].notna()].copy()
df_a['type'] = 'Actual'
df_a = df_a[['date', 'ridership', 'type']]
df_b = df_filtered[df_filtered['ridership'].isna()].copy()
df_b['type'] = 'Moving average'
df_b['ridership'] = df_b['predict_ridership'].copy()
df_b = df_b[['date', 'ridership', 'type']]
df_moving_average = pd.concat([df_a, df_b])

# Ridership Moving Average
st.subheader('Ridership moving average.')

# plot MA Chart
ma_chart = alt.Chart(df_moving_average).mark_line(point=True).encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('ridership:Q', title='Ridership'),
    color=alt.Color('type:N', title='Legend'),
    strokeDash=alt.condition(
        "datum.type == 'Moving average'",
        alt.value([5, 5]),
        alt.value([0])
    ),
    tooltip=['date:T', 'ridership:Q', 'type:N']
).interactive()

st.altair_chart(ma_chart, use_container_width=True)


# # Bar chart for actual
# bar = base.mark_bar(color='blue').encode(
#     x=alt.X('date:T', title='Date'),
#     y=alt.Y('ridership:Q', title='Ridership'),
#     tooltip=['date:T', 'ridership:Q']
# )
#
# # Line chart for prediction
# line = base.mark_line(
#     color='orange',
#     strokeDash=[5, 5],
#     point=True
# ).encode(
#     x='date:T',
#     y='predict_ridership:Q',
#     tooltip=['date:T', 'predict_ridership:Q']
# ).interactive()
#
# # Combine both
# chart = bar + line
#
# st.altair_chart(chart, use_container_width=True)

# SARIMA Forecast.
df_sarima = sarima_forecast(df_filtered)

st.subheader("Ridership SARIMA Forecast.")

df_sActual = df_filtered[df_filtered['ridership'].notna()][['date', 'ridership']]
df_sActual['type'] = 'Actual'

df_sarima_plot = df_sarima.copy()
df_sarima_plot.rename(columns={'sarima_forecast': 'ridership'}, inplace=True)
df_sarima_plot['type'] = 'SARIMA Forecast'

df_combined = pd.concat([df_sActual, df_sarima_plot])
df_combined.reset_index(drop=True, inplace=True)

# SARIMA chart
sarima_chart = alt.Chart(df_combined).mark_line(point='stretch').encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('ridership:Q', title='Ridership'),
    color=alt.Color('type:N', title='Legend'),
    strokeDash=alt.condition(
        "datum.type == 'SARIMA Forecast'",
        alt.value([5, 5]),
        alt.value([0])
    ),
    tooltip=['date:T', 'ridership:Q', 'type:N']
).interactive()

st.altair_chart(sarima_chart, use_container_width='stretch')

# regression trend
st.subheader("Regression Trend (Actual Data)")

df_actual = df_filtered[df_filtered['ridership'].notna()]

# regression chart
regression_chart = alt.Chart(df_actual).mark_point(color='green').encode(
    x=alt.X('date:T', title='date'),
    y=alt.Y('ridership:Q', title='ridership'),
    tooltip=['date:T', 'ridership:Q']
) + alt.Chart(df_actual).transform_regression(
    'date',
    'ridership'
).mark_line(color='red').encode(
    x='date:T',
    y='ridership:Q'
).interactive()

# show regression table
st.altair_chart(regression_chart, use_container_width='stretch')

# Show table
st.subheader("Data Table")
df_filtered_ = df_filtered[['date', 'service', 'ridership']]
df_filtered_['date'] = pd.to_datetime(df_filtered_['date']).dt.strftime('%Y-%m-%d')
df_filtered_ = df_filtered_[df_filtered_['ridership'].notna()]
st.dataframe(df_filtered_, hide_index=True)
