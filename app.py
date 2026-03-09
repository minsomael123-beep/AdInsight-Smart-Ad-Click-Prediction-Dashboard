import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Ad Click Predictor",
    layout="wide"
)

# --- Header ---
st.title("📊 Ad Click Prediction Dashboard")
st.markdown("Predict whether a user will click an ad based on their behavior and profile.")

# --- Load Model & Scaler ---
model = joblib.load("log.h5")
scaler = joblib.load("scaler.h5")

# --- Define Features manually (replace with your 9 features) ---
scaler_features = [
    "daily_time", "age", "area_income", "daily_internet", "male",
    "country_US", "country_UK", "country_CA", "source_Facebook"
]

# --- Sidebar: Individual Input ---
st.sidebar.header("👤 Enter User Data")
user_input = {}
for feat in scaler_features:
    # For binary features like gender/country/source
    if feat.startswith("gender_") or feat.startswith("country_") or feat.startswith("source_") or feat == "male":
        val = st.sidebar.selectbox(f"{feat}", [0,1])
    else:
        val = st.sidebar.number_input(f"{feat}", value=0.0)
    user_input[feat] = val

if st.sidebar.button("Predict"):
    features_df = pd.DataFrame([user_input])
    try:
        features_scaled = scaler.transform(features_df)
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
        st.error(f"Error transforming input features: {e}")

st.markdown("---")

# --- Bulk Prediction ---
st.subheader("📂 Bulk Prediction (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Add missing columns with 0
    for feat in scaler_features:
        if feat not in df.columns:
            df[feat] = 0

    # Arrange columns in the same order as scaler
    df_ordered = df[scaler_features]

    try:
        df_scaled = scaler.transform(df_ordered)
        preds = model.predict_proba(df_scaled)[:,1]
        df["Click_Probability"] = preds
        df["Prediction"] = (preds > 0.5).astype(int)

        st.write("Preview of Predictions:")
        st.dataframe(df.head())

        st.subheader("📊 Prediction Distribution")
        st.bar_chart(df["Prediction"].value_counts())
    except ValueError as e:
        st.error(f"Error transforming CSV data: {e}")





