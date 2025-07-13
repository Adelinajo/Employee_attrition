import streamlit as st
import pandas as pd
import joblib

# Load model & utilities
model = joblib.load("lmodel_col.pkl")
scaler = joblib.load("scaler_col.pkl")
feature_columns = joblib.load("features_col.pkl")  # Should be 6 features

st.title("🎯 Employee Attrition Predictor")

with st.form("attrition_form"):
    age = st.number_input("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Monthly Income", 1000, 100000, 5000)
    satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    years = st.slider("Years at Company", 0, 40, 5)
    submit = st.form_submit_button("Predict")

if submit:
    # Encode categorical variables
    gender = 1 if gender == "Male" else 0
    overtime = 1 if overtime == "Yes" else 0

    # Create input DataFrame with correct column names
    data = pd.DataFrame([[age, gender, income, satisfaction, overtime, years]],
                        columns=feature_columns)

    # Scale and predict
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]
    threshold = 0.5  
    pred = 1 if proba > threshold else 0

    # Display results
    st.subheader(f"Attrition Probability: {proba:.2%}")
    if pred == 1:
        st.error("⚠️ High risk of attrition.")
    else:
        st.success("✅ Low risk of attrition.")
