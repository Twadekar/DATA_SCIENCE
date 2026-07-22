import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import os

st.title("NIFTY 50 Bank Stock Price Prediction")
st.write("forecast for the next 30 days using prophet")

#Get all the saved models
models_file = [f for f in os.listdir("saved_models") if f.endswith(".pkl")]

#extract bank names
bank_names = [file.replace(".pkl", "") for file in models_file]

#Dropdown menu
selected_bank = st.selectbox("select a bank", sorted(bank_names))
st.write("selected bank:", selected_bank)

model_path = f"saved_models/{selected_bank}.pkl"
model = joblib.load(model_path)

#create future dataframe for next 30 days
future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

st.subheader("forecast chart")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.subheader("Next 30 days forecast")
forecast_30 = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
forecast_30.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
st.dataframe(forecast_30, use_container_width=True)

csv = forecast_30.to_csv(index = False)
st.download_button("download forecast csv", csv, file_name=f"{selected_bank}_forecast.csv", mime="text/csv")

st.subheader("prediction summary")
st.write(f"selected bank: {selected_bank}")
st.write(f"forecast period: Next 30 Days")
st.write(f"Latest predicted preice: $ {forecast_30["Predicted Price"].iloc[-1]:.2f}")

forecast_30["Date"] = forecast_30["Date"].dt.strftime("%d-%m-%Y")
forecast_30 = forecast_30.round(2)