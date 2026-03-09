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

# --- Define numeric and other features manually ---
numeric_features = ["daily_time", "age", "area_income", "daily_internet"]
binary_features = ["male", "source_Facebook"]

# --- List of all countries (used for dropdown & One-Hot) ---
countries = [
    "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua and Barbuda","Argentina",
    "Armenia","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados",
    "Belarus","Belgium","Belize","Benin","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana",
    "Brazil","Brunei","Bulgaria","Burkina Faso","Burundi","Côte d'Ivoire","Cabo Verde","Cambodia",
    "Cameroon","Canada","Central African Republic","Chad","Chile","China","Colombia","Comoros",
    "Congo (Congo-Brazzaville)","Costa Rica","Croatia","Cuba","Cyprus","Czechia","Democratic Republic of the Congo",
    "Denmark","Djibouti","Dominica","Dominican Republic","Ecuador","Egypt","El Salvador","Equatorial Guinea",
    "Eritrea","Estonia","Eswatini","Ethiopia","Fiji","Finland","France","Gabon","Gambia","Georgia",
    "Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti",
    "Holy See","Honduras","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Israel",
    "Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kiribati","Kuwait","Kyrgyzstan",
    "Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg",
    "Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands","Mauritania","Mauritius",
    "Mexico","Micronesia","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique","Myanmar",
    "Namibia","Nauru","Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Korea",
    "North Macedonia","Norway","Oman","Pakistan","Palau","Palestine State","Panama","Papua New Guinea",
    "Paraguay","Peru","Philippines","Poland","Portugal","Qatar","Romania","Russia","Rwanda","Saint Kitts and Nevis",
    "Saint Lucia","Saint Vincent and the Grenadines","Samoa","San Marino","Sao Tome and Principe","Saudi Arabia",
    "Senegal","Serbia","Seychelles","Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","Somalia",
    "South Africa","South Korea","South Sudan","Spain","Sri Lanka","Sudan","Suriname","Sweden","Switzerland","Syria",
    "Tajikistan","Tanzania","Thailand","Timor-Leste","Togo","Tonga","Trinidad and Tobago","Tunisia","Turkey",
    "Turkmenistan","Tuvalu","Uganda","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay",
    "Uzbekistan","Vanuatu","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"
]

# --- Friendly Labels for numeric/binary features ---
feature_labels = {
    "daily_time": "⏱ Daily Time Spent on Site (minutes)",
    "age": "🎂 Age of User",
    "area_income": "💰 Area Income ($)",
    "daily_internet": "🌐 Daily Internet Usage (minutes)",
    "male": "👨 Gender (0=Female,1=Male)",
    "source_Facebook": "📘 Source: Facebook (0=No,1=Yes)"
}

# --- Sidebar: Individual Input ---
st.sidebar.header("👤 Enter User Data")
user_input = {}

# Gender
user_input["male"] = st.sidebar.selectbox(feature_labels["male"], [0,1])

# Country Dropdown
selected_country = st.sidebar.selectbox("🌍 Select Country", countries)

# One-Hot Encoding for countries
for c in countries:
    col_name = f"country_{c.replace(' ','_')}"
    user_input[col_name] = 1 if c == selected_country else 0

# Other numeric inputs
for feat in numeric_features:
    user_input[feat] = st.sidebar.number_input(feature_labels[feat], value=0.0)

# Source
user_input["source_Facebook"] = st.sidebar.selectbox(feature_labels["source_Facebook"], [0,1])

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
