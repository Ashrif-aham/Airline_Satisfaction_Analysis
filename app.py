import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("airline_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Airline Satisfaction Prediction")

# Inputs
Gender = st.selectbox("Gender", [0, 1])
Customer_Type = st.selectbox("Customer Type", [0, 1])
Age = st.number_input("Age", 1, 100)
Flight_Distance = st.number_input("Flight Distance", 1, 10000)

# Temporary dummy values for remaining features
dummy_features = [0] * 19

if st.button("Predict"):

    features = [[
        Gender,
        Customer_Type,
        Age,
        Flight_Distance,
        *dummy_features
    ]]

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Satisfied Passenger")
    else:
        st.error("Not Satisfied Passenger")