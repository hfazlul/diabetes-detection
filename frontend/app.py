import os
import requests
import streamlit as st

# Set API URL (change here if needed)
API_URL = "https://diabetes-detection-1.onrender.com"

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'Diabetic' if result['prediction'] == 1 else 'Not Diabetic'} (Confidence: {result['confidence']:.2f})")
        else:
            st.error("API returned an error.")
    except requests.exceptions.RequestException:
        st.error("Could not connect to the prediction API.")
