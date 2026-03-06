import streamlit as st
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Ad Click Predictor", layout="wide")

# تحميل الموديل
model = load_model("log.h5")

# تحميل السكالر
scaler = joblib.load("scaler.h5")

st.title("📊 Ad Click Prediction Dashboard")

st.sidebar.header("Enter User Data")

daily_time = st.sidebar.number_input("Daily Time Spent on Site")
age = st.sidebar.number_input("Age")
area_income = st.sidebar.number_input("Area Income")
daily_internet = st.sidebar.number_input("Daily Internet Usage")

male = st.sidebar.selectbox("Gender", ["Female", "Male"])
male = 1 if male == "Male" else 0

if st.sidebar.button("Predict"):

    features = np.array([[daily_time, age, area_income, daily_internet, male]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = prediction[0][0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability", f"{probability:.2%}")

    with col2:
        if probability >= 0.5:
            st.success("User WILL Click the Ad 🎯")
        else:
            st.error("User will NOT Click the Ad ❌")

uploaded_file = st.file_uploader("Upload CSV for Bulk Prediction")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    df["Prediction"] = (preds > 0.5).astype(int)

    st.write(df.head())
    st.bar_chart(df["Prediction"].value_counts())