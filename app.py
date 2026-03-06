import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
import tempfile
import zipfile
import shutil

# محاولة استيراد Keras
try:
    from tensorflow.keras.models import load_model
except ImportError:
    st.error("TensorFlow/Keras غير مثبت. رجاءً ثبته باستخدام: pip install tensorflow")
    st.stop()

st.set_page_config(page_title="Ad Click Predictor", layout="wide")

st.title("📊 Ad Click Prediction Dashboard")
st.sidebar.header("Upload Your Model & Scaler")

# رفع الموديل HDF5 أو فولدر مضغوط SavedModel
uploaded_model = st.sidebar.file_uploader("Upload Model (.h5 or .zip for SavedModel folder)", type=["h5", "zip"])
uploaded_scaler = st.sidebar.file_uploader("Upload Scaler (.h5)", type=["h5"])

model = None
scaler = None

# دالة لفك فولدر مضغوط SavedModel مؤقتًا
def unzip_to_temp(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

# تحميل الموديل بعد رفعه
if uploaded_model is not None:
    if uploaded_model.name.endswith(".h5"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                tmp_file.write(uploaded_model.read())
                tmp_file.flush()
                model = load_model(tmp_file.name)
            st.success("تم تحميل موديل HDF5 بنجاح ✅")
        except Exception as e:
            st.error(f"فشل تحميل HDF5: {e}")
    elif uploaded_model.name.endswith(".zip"):
        try:
            temp_folder = unzip_to_temp(uploaded_model)
            model = load_model(temp_folder)
            st.success("تم تحميل موديل SavedModel من ZIP بنجاح ✅")
            shutil.rmtree(temp_folder)  # تنظيف الملفات المؤقتة
        except Exception as e:
            st.error(f"فشل تحميل SavedModel: {e}")
    else:
        st.error("صيغة الملف غير مدعومة!")
else:
    st.warning("يرجى رفع موديل للعمل على التنبؤ.")

# تحميل السكالر
if uploaded_scaler is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_scaler:
            tmp_scaler.write(uploaded_scaler.read())
            tmp_scaler.flush()
            scaler = joblib.load(tmp_scaler.name)
        st.success("تم تحميل السكالر بنجاح ✅")
    except Exception as e:
        st.error(f"فشل تحميل السكالر: {e}")
else:
    st.warning("يرجى رفع السكالر للعمل على التنبؤ.")

# واجهة المستخدم للتنبؤ
if model is not None and scaler is not None:
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

