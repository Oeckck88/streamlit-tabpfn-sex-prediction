
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import os

st.set_page_config(page_title="Sex Prediction (TabPFNv2)", layout="centered")

st.title("🧠 Sex Prediction using TabPFNv2")
st.markdown("입력한 해부학적 측정값을 기반으로 성별을 예측하고, LIME으로 해석을 제공합니다.")

# ✅ 모델 로드
model_path = "/content/drive/MyDrive/ex/new_experiment/clf/exp2_final_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ 모델 파일이 존재하지 않습니다.")
    st.stop()
model = joblib.load(model_path)

# ✅ Feature 정의
feature_names = ['avg_height', 'avg_manubrium', 'avg_body', 'sternum(M+B)']
class_names = ['Female', 'Male']

# ✅ 사용자 입력 받기
st.subheader("🔢 Input Features")

user_input = {}
for feat in feature_names:
    user_input[feat] = st.number_input(f"{feat}", value=50.0, min_value=0.0, max_value=250.0)

input_df = pd.DataFrame([user_input])

# ✅ 예측 실행
if st.button("🔍 Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted Sex:** {'Male' if pred == 1 else 'Female'}")
    st.write(f"Confidence (Female): {proba[0]:.3f}")
    st.write(f"Confidence (Male): {proba[1]:.3f}")

    # ✅ LIME 해석
    st.subheader("🧩 Explanation with LIME")

    explainer = LimeTabularExplainer(
        training_data=np.array(input_df),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    exp = explainer.explain_instance(
        data_row=input_df.values[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )

    st.pyplot(exp.as_pyplot_figure())
