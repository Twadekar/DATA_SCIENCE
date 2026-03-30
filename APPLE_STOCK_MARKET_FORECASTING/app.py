import pandas as pd                                       #for data loading and manipulation
import numpy as np                                        #for numerical operations
import matplotlib.pyplot as plt                           #for static visualizations
from statsmodels.tsa.statespace.sarimax import SARIMAX    #for time series forecasting
import streamlit as st                                    #for creating interactive web app
import plotly.graph_objects as go                         #for interactive charts
import warnings                                           #Ignore warnings for cleaner output
warnings.filterwarnings("ignore")


#PAGE SETTINGS
st.set_page_config(page_title="Apple Stock Forecast App", layout="wide")
st.title("Apple Stock Forecasting")


# LOAD DATA
df = pd.read_csv("Stock Market.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# DATE RANGE SELECTOR
st.subheader("Select Date Range for Visualization")
start_date = st.date_input("Start Date", df["Date"].min().date())
end_date = st.date_input("End Date", df["Date"].max().date())

if start_date > end_date:
    st.error("Start date cannot be greater than end date.")
    st.stop()

df_range = df[(df["Date"] >= pd.to_datetime(start_date)) &
              (df["Date"] <= pd.to_datetime(end_date))]

st.write("Data for Selected Range")
st.dataframe(df_range)



# GRAPH
st.subheader("Chart")

fig = go.Figure(data=[go.Candlestick(
    x=df_range["Date"],
    open=df_range["Open"],
    high=df_range["High"],
    low=df_range["Low"],
    close=df_range["Close"]
)])
fig.update_layout(height=500, xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)



#MOVING AVERAGES

st.subheader("📉 Moving Averages (MA20 / MA50 / MA200)")

df_ma = df.copy()
df_ma["MA20"] = df_ma["Close"].rolling(20).mean()
df_ma["MA50"] = df_ma["Close"].rolling(50).mean()
df_ma["MA200"] = df_ma["Close"].rolling(200).mean()

fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_ma["Date"], df_ma["Close"], label="Close")
ax.plot(df_ma["Date"], df_ma["MA20"], label="MA20")
ax.plot(df_ma["Date"], df_ma["MA50"], label="MA50")
ax.plot(df_ma["Date"], df_ma["MA200"], label="MA200")
ax.legend()
ax.grid()
st.pyplot(fig2)



if st.button("Generate SARIMA Forecast"):

    st.success("Forecasting last 30 days from end of dataset")

    # Prepare time series
    ts = df.set_index("Date")["Close"]

    # Build SARIMA model
    model = SARIMAX(ts, order=(5,1,2), seasonal_order=(0,0,0,0))
    model_fit = model.fit()

    # Correct way to forecast

    forecast_obj = model_fit.get_forecast(steps=30)
    preds = forecast_obj.predicted_mean

    # Create future date index
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    # Create forecast df
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": preds.values})

    # Display clean table
    forecast_display = forecast_df.copy()
    forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(forecast_display)

    # CSV Download
    csv = forecast_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "sarima_forecast.csv", "text/csv")

    # Plot forecast + history
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df["Date"], df["Close"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], marker="o", color="orange", label="SARIMA Forecast")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
