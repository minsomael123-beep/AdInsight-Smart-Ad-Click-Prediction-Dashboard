import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(page_title="Ad Click Predictor", layout="wide")

# --- Load Model & Scaler ---
model = joblib.load("log.h5")
scaler = joblib.load("scaler.h5")

# --- Features (exact 9) ---
features_list = [
    "daily_time","age","area_income","daily_internet","male",
    "country_US","country_UK","country_Egypt","source_Facebook"
]

# --- Sidebar Inputs ---
st.sidebar.header("👤 Enter User Data")
user_input = {}

# Numeric features
user_input["daily_time"] = st.sidebar.number_input("⏱ Daily Time Spent on Site", value=30.0, min_value=0.0, max_value=300.0)
user_input["age"] = st.sidebar.number_input("🎂 Age", value=30, min_value=10, max_value=100)
user_input["area_income"] = st.sidebar.number_input("💰 Area Income ($)", value=50000, min_value=0)
user_input["daily_internet"] = st.sidebar.number_input("🌐 Daily Internet Usage (minutes)", value=60, min_value=0, max_value=720)

# Gender
user_input["male"] = st.sidebar.selectbox("👨 Gender", [0,1])

# Country dropdown
country = st.sidebar.selectbox("🌍 Select Country", ["USA","UK","Egypt"])
user_input["country_US"] = 1 if country=="USA" else 0
user_input["country_UK"] = 1 if country=="UK" else 0
user_input["country_Egypt"] = 1 if country=="Egypt" else 0

# Source dropdown
source = st.sidebar.selectbox("🌐 Source Website", ["Other","Facebook"])
user_input["source_Facebook"] = 1 if source=="Facebook" else 0

# --- Individual Prediction ---
st.header("📊 Individual Ad Click Prediction")
if st.sidebar.button("Predict"):
    df = pd.DataFrame([user_input])
    try:
        df_scaled = scaler.transform(df[features_list])
        probability = model.predict_proba(df_scaled)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Click Probability", f"{probability:.2%}")
        with col2:
            if probability >= 0.5:
                st.success("User WILL Click the Ad 🎯")
            else:
                st.error("User will NOT Click the Ad ❌")
    except ValueError as e:
        st.error(f"Error transforming input features: {e}")

st.markdown("---")

# --- Bulk Prediction ---
st.header("📂 Bulk Prediction (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Fill missing features with 0
    for feat in features_list:
        if feat not in df.columns:
            df[feat] = 0

    df_ordered = df[features_list]

    try:
        df_scaled = scaler.transform(df_ordered)
        preds = model.predict_proba(df_scaled)[:,1]
        df["Click_Probability"] = preds
        df["Prediction"] = (preds > 0.5).astype(int)

        st.subheader("Preview of Predictions")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📊 Prediction Distribution")
        st.bar_chart(df["Prediction"].value_counts())
    except ValueError as e:
        st.error(f"Error transforming CSV data: {e}")


