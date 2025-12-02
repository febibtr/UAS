# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Clustering DSS", layout="wide")

st.title("🍷 Sistem Pendukung Keputusan — Wine Clustering")
st.write("""
Aplikasi ini melakukan **Clustering (K-Means)** pada dataset wine dan memberikan rekomendasi berdasarkan 
cluster yang terbentuk. Anda dapat upload dataset atau menggunakan dataset default `wine-clustering.csv`.
""")

# =======================
# UPLOAD ATAU GUNAKAN DEFAULT
# =======================
uploaded = st.sidebar.file_uploader("Upload dataset wine (.csv)", type=["csv"])
default_path = "/mnt/data/wine-clustering.csv"

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Dataset berhasil diupload.")
else:
    try:
        df = pd.read_csv(default_path)
        st.sidebar.info("Menggunakan dataset default: wine-clustering.csv")
    except:
        st.error("Tidak ada file default. Silakan upload CSV.")
        st.stop()

# =======================
# PREVIEW DATASET
# =======================
st.subheader("Preview Dataset")
st.dataframe(df.head())

# Pilih hanya kolom numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.error("Dataset tidak memiliki kolom numerik untuk clustering.")
    st.stop()

st.write("Kolom numerik yang digunakan:", numeric_cols)

# =======================
# PERSIAPAN DATA
# =======================
data = df[numeric_cols].fillna(df[numeric_cols].mean())

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

# =======================
# ELBOW METHOD
# =======================
st.sidebar.subheader("Elbow Method")
do_elbow = st.sidebar.checkbox("Tampilkan Elbow Method", value=True)

if do_elbow:
    max_k = st.sidebar.slider("Max k", 2, 12, 8)
    inertias = []
    ks = range(1, max_k+1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_title("Elbow Method (SSE vs k)")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia (SSE)")
    st.pyplot(fig)

# =======================
# K-MEANS CLUSTERING
# =======================
st.sidebar.subheader("Clustering")
k = st.sidebar.slider("Jumlah cluster (k)", 2, 8, 3)

km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(scaled)

df["cluster"] = labels

st.subheader("Hasil Clustering")
st.dataframe(df.head())

# =======================
# CLUSTER SUMMARY
# =======================
st.subheader("Ringkasan Cluster")
cluster_summary = df.groupby("cluster")[numeric_cols].mean()
st.dataframe(cluster_summary)

# =======================
# VISUALISASI PCA
# =======================
st.subheader("Visualisasi PCA (2D)")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

fig2, ax2 = plt.subplots(figsize=(8,6))
scatter = ax2.scatter(pca_result[:,0], pca_result[:,1], c=labels, cmap='tab10')
ax2.set_title("PCA Visualization")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")

handles, _ = scatter.legend_elements()
ax2.legend(handles, [f"Cluster {i}" for i in range(k)])
st.pyplot(fig2)

# =======================
# INTERPRETASI CLUSTER
# =======================
st.subheader("Interpretasi Cluster")

target_feature = numeric_cols[0]  # contoh: alcohol

order = cluster_summary[target_feature].sort_values().index

interpretation = {}
for rank, c in enumerate(order):
    if rank == 0:
        tag = "rendah"
    elif rank == len(order)-1:
        tag = "tinggi"
    else:
        tag = "sedang"

    interpretation[c] = f"Cluster {c} → nilai {target_feature} {tag}"

for i in range(k):
    st.write(interpretation[i])

# =======================
# SISTEM PENDUKUNG KEPUTUSAN (REKOMENDASI)
# =======================
st.subheader("🟢 Sistem Rekomendasi (SPK)")

st.write("Masukkan nilai fitur untuk menentukan cluster:")

input_values = []
for col in numeric_cols:
    val = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_values.append(val)

if st.button("Prediksi Cluster Kasus Baru"):
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    predicted_cluster = km.predict(arr_scaled)[0]

    st.success(f"Kasus baru masuk **Cluster {predicted_cluster}**")
    st.write("Interpretasi:", interpretation[predicted_cluster])

    # rekomendasi sederhana SPK
    st.write("Rekomendasi:")
    if target_feature:
        val = arr[0][numeric_cols.index(target_feature)]
        mean_feature = df.groupby("cluster")[target_feature].mean()[predicted_cluster]

        if val > mean_feature:
            st.write("• Nilai lebih tinggi dari rata-rata cluster → cocok untuk kategori premium.")
        else:
            st.write("• Nilai lebih rendah → cocok untuk kategori standard atau ekonomis.")

# =======================
# DOWNLOAD HASIL
# =======================
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Hasil Cluster (CSV)", csv, "wine_clustered.csv", "text/csv")
