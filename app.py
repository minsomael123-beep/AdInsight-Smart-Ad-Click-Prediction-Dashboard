import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعداد صفحة الـ Streamlit
st.set_page_config(page_title="Ad Click Predictor", layout="wide")

st.title("📊 Ad Click Prediction Dashboard")
st.markdown("Predict whether a user will click an ad based on their behavior and profile.")

# تحميل الموديل والسكالر
model = joblib.load("log.h5")      # موديل scikit-learn محفوظ بـ joblib
scaler = joblib.load("scaler.h5")  # السكالر

# Sidebar لإدخال بيانات المستخدم
st.sidebar.header("Enter User Data")

daily_time = st.sidebar.number_input("Daily Time Spent on Site")
age = st.sidebar.number_input("Age")
area_income = st.sidebar.number_input("Area Income")
daily_internet = st.sidebar.number_input("Daily Internet Usage")
male = st.sidebar.selectbox("Gender", ["Female", "Male"])
male = 1 if male == "Male" else 0

# زر التنبؤ للفرد الواحد
if st.sidebar.button("Predict"):
    features = np.array([[daily_time, age, area_income, daily_internet, male]])
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

st.markdown("---")

# Bulk prediction من CSV
st.subheader("📂 Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV file for bulk prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # تأكد إن الأعمدة بالترتيب: daily_time, age, area_income, daily_internet, male
    df_scaled = scaler.transform(df)
    preds = model.predict_proba(df_scaled)[:, 1]
    df["Prediction"] = (preds > 0.5).astype(int)
    df["Click_Probability"] = preds

    st.write("Preview of Predictions:")
    st.dataframe(df.head())

    st.subheader("📊 Prediction Distribution")
    st.bar_chart(df["Prediction"].value_counts())



