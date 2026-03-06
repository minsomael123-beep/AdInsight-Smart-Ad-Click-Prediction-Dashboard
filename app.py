import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# محاولة استيراد Keras
try:
    from tensorflow.keras.models import load_model
except ImportError:
    st.error("TensorFlow/Keras غير مثبت. رجاءً ثبته باستخدام: pip install tensorflow")
    st.stop()

st.set_page_config(page_title="Ad Click Predictor", layout="wide")

# تحديد مسارات الملفات
MODEL_H5 = "log.h5"       # إذا الموديل HDF5
MODEL_SAVED = "log"       # إذا الموديل SavedModel (فولدر)
SCALER_PATH = "scaler.h5" # ملف السكالر

# دالة لتحميل الموديل بأمان
def safe_load_model():
    if os.path.exists(MODEL_H5):
        try:
            model = load_model(MODEL_H5)
            st.success("تم تحميل موديل HDF5 بنجاح ✅")
            return model
        except Exception as e:
            st.warning(f"فشل تحميل HDF5: {e}")
    if os.path.exists(MODEL_SAVED):
        try:
            model = load_model(MODEL_SAVED)
            st.success("تم تحميل موديل SavedModel بنجاح ✅")
            return model
        except Exception as e:
            st.warning(f"فشل تحميل SavedModel: {e}")
    st.error("لم يتم العثور على أي موديل صالح!")
    st.stop()

# تحميل الموديل
model = safe_load_model()

# تحميل السكالر
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        st.success("تم تحميل السكالر بنجاح ✅")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل السكالر: {e}")
        st.stop()
else:
    st.error("ملف السكالر غير موجود!")
    st.stop()

# واجهة Streamlit
st.title("📊 Ad Click Prediction Dashboard")
st.sidebar.header("Enter User Data")

daily_time = st.sidebar.number_input("Daily Time Spent on Site")
age = st.sidebar.number_input("Age")
area_income = st.sidebar.number_input("Area Income")
daily_internet = st.sidebar.number_input("Daily Internet Usage")
male = st.sidebar.selectbox("Gender", ["Female", "Male"])
male = 1 if male == "Male" else 0

# زر التنبؤ
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

# التنبؤ بالجماعي CSV
uploaded_file = st.file_uploader("Upload CSV for Bulk Prediction")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    df["Prediction"] = (preds > 0.5).astype(int)

    st.write(df.head())
    st.bar_chart(df["Prediction"].value_counts())

