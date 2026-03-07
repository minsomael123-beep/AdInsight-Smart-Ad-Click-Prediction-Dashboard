import streamlit as st
import numpy as np
import pickle

# ===================== إعدادات الصفحة =====================
st.set_page_config(
    page_title="نموذج التنبؤ",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 نموذج التنبؤ")
st.markdown("---")

# ===================== تحميل النموذج والـ Scaler =====================
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    model = load_model("model.h5")
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.h5", "rb") as f:
        scaler = pickle.load(f)
    return scaler

try:
    model = load_model()
    scaler = load_scaler()

    # استخراج عدد الـ features من الـ scaler أو الموديل
    if hasattr(scaler, "n_features_in_"):
        n_features = scaler.n_features_in_
    elif hasattr(scaler, "scale_"):
        n_features = len(scaler.scale_)
    else:
        n_features = model.input_shape[-1]

    # أسماء الـ features (غيّرها حسب بياناتك)
    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

    st.success(f"✅ تم تحميل النموذج بنجاح | عدد المدخلات: {n_features}")
    st.markdown("---")

    # ===================== إدخال البيانات =====================
    st.subheader("📝 أدخل القيم")

    cols = st.columns(2)
    inputs = []

    for i, name in enumerate(feature_names):
        col = cols[i % 2]
        val = col.number_input(
            label=name,
            value=0.0,
            format="%.4f",
            key=f"input_{i}"
        )
        inputs.append(val)

    st.markdown("---")

    # ===================== التنبؤ =====================
    if st.button("🔮 تنبأ", use_container_width=True, type="primary"):
        input_array = np.array([inputs])

        # تطبيق الـ Scaler
        input_scaled = scaler.transform(input_array)

        # التنبؤ
        prediction = model.predict(input_scaled)

        st.markdown("### 📊 النتيجة")

        # إذا كان التصنيف (classification)
        if prediction.shape[-1] > 1:
            pred_class = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction)) * 100

            st.metric(label="الفئة المتوقعة", value=f"Class {pred_class}")
            st.metric(label="نسبة الثقة", value=f"{confidence:.2f}%")

            # عرض جميع الاحتمالات
            with st.expander("عرض جميع الاحتمالات"):
                for idx, prob in enumerate(prediction[0]):
                    st.write(f"Class {idx}: {prob*100:.2f}%")
                    st.progress(float(prob))

        # إذا كان regression
        else:
            result = float(prediction[0][0])
            st.metric(label="القيمة المتوقعة", value=f"{result:.4f}")

    # ===================== رفع ملف CSV =====================
    st.markdown("---")
    st.subheader("📁 أو ارفع ملف CSV للتنبؤ بدفعة")

    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])

    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write("معاينة البيانات:", df.head())

        if st.button("🚀 تنبأ بالكل", use_container_width=True):
            try:
                X = df.values
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)

                if preds.shape[-1] > 1:
                    df["Prediction"] = np.argmax(preds, axis=1)
                    df["Confidence"] = np.max(preds, axis=1)
                else:
                    df["Prediction"] = preds.flatten()

                st.success(f"✅ تم التنبؤ بـ {len(df)} صف")
                st.dataframe(df)

                # تحميل النتائج
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ تحميل النتائج",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"خطأ في البيانات: {e}")

except Exception as e:
    st.error(f"❌ خطأ في تحميل النموذج: {e}")
    st.info("تأكد إن ملفات `model.h5` و `scaler.h5` موجودة في نفس مجلد الـ app.py")


