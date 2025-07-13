import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

# Load model & utilities
model = joblib.load("lmodel_col.pkl")
scaler = joblib.load("scaler_col.pkl")
feature_columns = joblib.load("features_col.pkl")


st.title("🎯Employee Performance Rating Predictor")

# --- Input Features Section ---
# st.header("📋 Input Employee Features")
education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
job_involvement = st.slider("Job Involvement", 1, 4)
job_level = st.slider("Job Level", 1, 5)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
years_at_company = st.slider("Years at Company", 0, 40)
years_in_current_role = st.slider("Years in Current Role", 0, 20)


# --- Display Result ---
if st.button("Predict Performance Rating"):
    # Create input DataFrame with correct column names
    input_data = pd.DataFrame([[education, job_involvement, job_level, monthly_income, years_at_company, years_in_current_role]],
                              columns=feature_columns)
    
    # Scale and predict
    data_scaled = scaler.transform(input_data)
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(data_scaled)[0][1]
    threshold = 0.5  
    pred = 1 if proba > threshold else 0

    
    if pred == 0:
        st.markdown("Performance  Rating is 3")
    else :
        st.markdown("Performance Rating is 4")

    result = f"Predicted Performance Rating: {pred}"

  

    st.success(result)