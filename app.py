import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model & scaler ---
model = joblib.load("log.h5")
scaler = joblib.load("scaler.h5")

# --- Features (exactly 9, متوافق مع السكالر) ---
features_list = [
    "daily_time","age","area_income","daily_internet","male",
    "country_US","country_UK","country_Egypt","source_Facebook"
]

# --- Sidebar Inputs ---
st.sidebar.header("Enter User Data")

user_input = {}

# Numeric features
user_input["daily_time"] = st.sidebar.number_input("⏱ Daily Time Spent on Site", value=0.0)
user_input["age"] = st.sidebar.number_input("🎂 Age", value=0.0)
user_input["area_income"] = st.sidebar.number_input("💰 Area Income ($)", value=0.0)
user_input["daily_internet"] = st.sidebar.number_input("🌐 Daily Internet Usage (minutes)", value=0.0)

# Gender
user_input["male"] = st.sidebar.selectbox("👨 Gender (0=Female,1=Male)", [0,1])

# Country dropdown (One-Hot)
country = st.sidebar.selectbox("🌍 Select Country", ["USA","UK","Egypt"])
user_input["country_US"] = 1 if country=="USA" else 0
user_input["country_UK"] = 1 if country=="UK" else 0
user_input["country_Egypt"] = 1 if country=="Egypt" else 0

# Source
source = st.sidebar.selectbox("🌐 Source Website", ["Other","Facebook"])
user_input["source_Facebook"] = 1 if source=="Facebook" else 0

# --- Predict button ---
if st.sidebar.button("Predict"):
    df = pd.DataFrame([user_input])

    # Scale & predict
    try:
        df_scaled = scaler.transform(df[features_list])
        probability = model.predict_proba(df_scaled)[0][1]

        st.metric("Click Probability", f"{probability:.2%}")
        if probability >= 0.5:
            st.success("User WILL Click the Ad 🎯")
        else:
            st.error("User will NOT Click the Ad ❌")
    except ValueError as e:
        st.error(f"Error transforming input features: {e}")


