import streamlit as st
import numpy as np
import joblib

st.title("🍷 Wine Quality Prediction App")

model = joblib.load('wine_model.pkl')

st.header("Enter Wine Properties")
alcohol = st.number_input("Alcohol", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)

# Add more if needed. You must match total number of input features.
if st.button("Predict Quality"):
    # Placeholder input (you must expand to full feature list!)
    features = np.array([[7.4, volatile_acidity, citric_acid, 0.036, 0.9988, 3.20, 0.68, sulphates, alcohol, 0.7, 0.2]])
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0].capitalize()}")
