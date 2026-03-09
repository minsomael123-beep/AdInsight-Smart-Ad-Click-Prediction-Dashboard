import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعداد الصفحة
st.set_page_config(page_title="Ad Click Predictor", layout="wide")
st.title("📊 Ad Click Prediction Dashboard")
st.markdown("Predict whether a user will click an ad based on their behavior and profile.")

# تحميل الموديل والسكالر
model = joblib.load("log.h5")
scaler = joblib.load("scaler.h5")

# Sidebar لإدخال بيانات فردية
st.sidebar.header("Enter User Data")
daily_time = st.sidebar.number_input("Daily Time Spent on Site", value=68.0)
age = st.sidebar.number_input("Age", value=35)
area_income = st.sidebar.number_input("Area Income", value=60000)
daily_internet = st.sidebar.number_input("Daily Internet Usage", value=180)
male = st.sidebar.selectbox("Gender", ["Female", "Male"])
male = 1 if male == "Male" else 0

if st.sidebar.button("Predict"):
    features = np.array([[daily_time, age, area_income, daily_internet, male]])
    try:
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Click Probability", f"{probability:.2%}")
        with col2:
            if probability >= 0.5:
                st.success("User WILL Click the Ad 🎯")
            else:
                st.error("User will NOT Click the Ad ❌")
    except ValueError as e:
        st.error(f"Error in input features: {e}")

st.markdown("---")
st.subheader("📂 Bulk Prediction (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["daily_time", "age", "area_income", "daily_internet", "male"]
    # تحقق من الأعمدة
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
    else:
        try:
            df_scaled = scaler.transform(df[required_cols])
            preds = model.predict_proba(df_scaled)[:, 1]
            df["Click_Probability"] = preds
            df["Prediction"] = (preds > 0.5).astype(int)

            st.write("Preview of Predictions:")
            st.dataframe(df.head())

            st.subheader("📊 Prediction Distribution")
            st.bar_chart(df["Prediction"].value_counts())
        except ValueError as e:
            st.error(f"Error transforming CSV data: {e}")




