NIFTY 50 Bank Stock Price Prediction

Project Overview
This project forecasts the next 30 days of NIFTY Bank stock prices using Facebook Prophet. A Streamlit web application allows users to select a bank model and visualize forecasts.

Features
- Interactive Streamlit dashboard
- Select any trained bank model
- 30-day forecast
- Forecast chart and components
- Download forecast as CSV

Technologies
- Python
- Pandas
- Prophet
- Streamlit
- Plotly
- Joblib

Project Structure
- bn.ipynb : Model training and forecasting
- app.py : Streamlit application
- banknifty.csv : Dataset
- saved_models/ : Trained Prophet models

How to Run
1. Install requirements.
2. Place trained models inside saved_models.
3. Run:
   streamlit run app.py

Future Improvements
- Add multiple forecasting models
- Performance comparison
- Live stock data integration