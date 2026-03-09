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

# --- Define features ---
numeric_features = ["daily_time", "age", "area_income", "daily_internet"]
binary_features = ["male"]
categorical_features = ["country", "source_website"]  # as integer codes

# --- Friendly Labels ---
feature_labels = {
    "daily_time": "⏱ Daily Time Spent on Site (minutes)",
    "age": "🎂 Age of User",
    "area_income": "💰 Area Income ($)",
    "daily_internet": "🌐 Daily Internet Usage (minutes)",
    "male": "👨 Gender (0=Female,1=Male)",
    "country": "🌍 Select Country",
    "source_website": "🌐 Source Website"
}

# Example countries and sources compatible with your current scaler
countries = ["USA", "UK", "Canada"]
sources = ["Other", "Facebook", "Google"]

# --- Sidebar: Individual Input ---
st.sidebar.header("👤 Enter User Data")
user_input = {}

# Numeric inputs
for feat in numeric_features:
    user_input[feat] = st.sidebar.number_input(feature_labels[feat], value=0.0)

# Binary input
user_input["male"] = st.sidebar.selectbox(feature_labels["male"], [0,1])

# Country dropdown as integer code
selected_country = st.sidebar.selectbox(feature_labels["country"], countries)
user_input["country"] = countries.index(selected_country)  # 0,1,2

# Source dropdown as integer code
selected_source = st.sidebar.selectbox(feature_labels["source_website"], sources)
user_input["source_website"] = sources.index(selected_source)  # 0,1,2

# Predict button
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
    for feat in user_input.keys():
        if feat not in df.columns:
            df[feat] = 0

    # Arrange columns in the same order as model expects
    df_ordered = df[list(user_input.keys())]

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

