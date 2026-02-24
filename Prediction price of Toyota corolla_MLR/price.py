#import Libraries
import numpy as np
import pickle
import streamlit as st

#Load Saved Model
with open("model.pkl", "rb") as p:
 model = pickle.load(p)

#Set app title n layout
st.set_page_config(page_title="car Price Prediction", layout="centered")
st.title("Car Prediction System")
st.write("Please Enter Required Values")

#User Streamlit input widgets for user input
Age_08_04 = st.number_input("Age_08_04", min_value=0.0, format="%.2f")
KM = st.number_input("KM", min_value=0.0, format="%.2f")
Fuel_Type= st.number_input("Fuel_Type(Petrol = 0, Diesel = 1, CNG=2)", min_value=0.0, format="%.2f")
HP = st.number_input("HP", min_value=0.0, format="%.2f")
Automatic = st.number_input("Automatic", min_value=0.0, format="%.2f")
cc = st.number_input("cc", min_value=0.0, format="%.2f")
Doors = st.number_input("Doors", min_value=0.0, format="%.2f")
Cylinders = st.number_input("Cylinders", min_value=0.0, format="%.2f")
Gears = st.number_input("Gears", min_value=0.0, format="%.2f")
Weight = st.number_input("Weight", min_value=0.0, format="%.2f")

#For Price Calculation
if st.button("Predict SalesPrices"):
    features= np.array([[ Age_08_04, KM, Fuel_Type, HP, Automatic, cc, Doors, Cylinders, Gears, Weight]])
    predicted_price =model.predict(features).item()
    st.success(f"predicted sales: {predicted_price:.2f}")

