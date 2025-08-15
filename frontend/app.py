import os
import requests
import streamlit as st
import pandas as pd

# Set API URL (change via Render environment variable if needed)
API_URL = os.getenv("API_URL", "https://diabetes-detection-1.onrender.com")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Metrics"])
st.sidebar.markdown(f"[📄 View API Docs]({API_URL}/docs)")

# ---------------- Prediction Page ----------------
if page == "Prediction":
    st.title("🩺 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose", min_value=0, max_value=200)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
    insulin = st.number_input("Insulin", min_value=0, max_value=900)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, format="%.1f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, format="%.2f")
    age = st.number_input("Age", min_value=1, max_value=120)

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
            response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                prediction_label = "Diabetic" if result["prediction"] == 1 else "Not Diabetic"
                st.success(f"Prediction: {prediction_label} "
                           f"(Confidence: {result['confidence'] * 100:.1f}%)")
            else:
                st.error(f"API returned an error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the prediction API: {e}")

# ---------------- Model Metrics Page ----------------
elif page == "Model Metrics":
    st.title("📊 Model Performance Metrics")

    try:
        response = requests.get(f"{API_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics_data = response.json()
            df = pd.DataFrame(metrics_data).T  # Convert dict to DataFrame
            df.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
            st.dataframe(df.style.format("{:.2%}"))
        else:
            st.error(f"API returned an error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the metrics API: {e}")
