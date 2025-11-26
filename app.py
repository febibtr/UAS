
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
feature_cols = model_data["features"]
target_names = ["0", "1"]

st.set_page_config(
    page_title="Sistem Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Sistem Pendukung Keputusan – Prediksi Diabetes")

st.sidebar.header("Masukkan Data Pasien")
user_input = {}
for col in feature_cols:
    user_input[col] = st.sidebar.number_input(col, min_value=0.0, max_value=300.0, value=50.0)

df_input = pd.DataFrame([user_input])
df_scaled = scaler.transform(df_input)

pred = model.predict(df_scaled)[0]
proba = model.predict_proba(df_scaled)[0]

st.subheader("Hasil Prediksi")
if pred == 1:
    st.markdown("<h2 style='color:red;'>💀 Berisiko Diabetes</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='color:green;'>💚 Tidak Berisiko</h2>", unsafe_allow_html=True)

st.write(f"Probabilitas Tidak Diabetes: {proba[0]:.4f}")
st.write(f"Probabilitas Diabetes: {proba[1]:.4f}")

st.subheader("Grafik Probabilitas")
fig1, ax1 = plt.subplots()
sns.barplot(x=["Tidak", "Diabetes"], y=proba)
st.pyplot(fig1)

st.subheader("Confusion Matrix (Saat Training)")
cm = model_data["confusion_matrix"]
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax2)
st.pyplot(fig2)
