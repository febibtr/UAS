import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="DSS Klasifikasi Kanker Payudara",
    layout="wide",
    page_icon="🩺"
)

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = Path("model.pkl")
if not MODEL_PATH.exists():
    st.error("❗ model.pkl tidak ditemukan. Jalankan `python train.py` terlebih dahulu.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
model = data["model"]
COLUMNS = data["columns"]
TARGET_NAMES = data["target_names"]

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🧪 Prediksi Manual", "📂 Prediksi dari CSV", "📊 Visualisasi Model", "📄 Sample Input"]
)

# ---------------------------
# HOME
# ---------------------------
if menu == "🏠 Home":
    st.title("🩺 Sistem Pendukung Keputusan - Klasifikasi Kanker Payudara")
    st.markdown("""
    Aplikasi ini menggunakan **Random Forest Classification** untuk memprediksi apakah sampel 
    kanker payudara bersifat *benign* atau *malignant*.
    
    ### Fitur Aplikasi:
    - Prediksi satu data secara manual
    - Prediksi dari file CSV
    - Visualisasi confusion matrix
    - Visualisasi probabilitas prediksi
    - Contoh input dari sample_input.csv

    **Model telah dilatih menggunakan dataset bawaan scikit-learn `breast_cancer`.**
    """)

# ---------------------------
# PREdIkSI MANUAL
# ---------------------------
elif menu == "🧪 Prediksi Manual":
    st.title("🧪 Prediksi Satu Data (Manual)")

    st.info("Isi nilai fitur di bawah. Jika tidak yakin, gunakan nilai default.")

    values = {}
    with st.form("manual_form"):
        for col in COLUMNS:
            values[col] = st.number_input(col, value=0.0)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        X = pd.DataFrame([values], columns=COLUMNS)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        st.success(f"📌 **Hasil Prediksi: {TARGET_NAMES[pred].upper()}**")
        st.write("Probabilitas:", prob)

        st.progress(float(prob[1]))

# ---------------------------
# PREDIKSI DARI CSV
# ---------------------------
elif menu == "📂 Prediksi dari CSV":
    st.title("📂 Prediksi dari File CSV")

    file = st.file_uploader("Unggah file CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Data dibaca:")
        st.dataframe(df.head())

        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            st.error("Kolom berikut hilang: " + ", ".join(missing))
        else:
            X = df[COLUMNS]
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

            df["prediksi"] = preds
            df["prob_malignant"] = probs

            st.success("Prediksi selesai!")
            st.dataframe(df)

            st.download_button(
                "Download hasil prediksi",
                df.to_csv(index=False),
                "hasil_prediksi.csv",
                "text/csv"
            )

# ---------------------------
# VISUALISASI
# ---------------------------
elif menu == "📊 Visualisasi Model":
    st.title("📊 Visualisasi Model")

    st.subheader("Confusion Matrix")

    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_breast_cancer
        
        data_bc = load_breast_cancer()
        X, y = data_bc.data, data_bc.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preds = model.predict(X_test)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error("Gagal memuat confusion matrix: " + str(e))

# ---------------------------
# SAMPLE INPUT
# ---------------------------
elif menu == "📄 Sample Input":
    st.title("📄 Sample Input (sample_input.csv)")

    path = Path("sample_input.csv")
    if not path.exists():
        st.error("sample_input.csv tidak ditemukan.")
    else:
        df = pd.read_csv(path)
        st.dataframe(df)
