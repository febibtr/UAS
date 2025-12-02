import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Aplikasi Clustering K-Means")
st.write("Upload dataset CSV untuk dilakukan clustering.")

# ============================================
# 1. UPLOAD FILE CSV
# ============================================
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data berhasil dimuat:")
    st.dataframe(df)

    # ============================================
    # 2. PILIH KOLOM NUMERIK UNTUK CLUSTERING
    # ============================================
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numerical_cols) < 2:
        st.warning("Dataset harus memiliki minimal 2 kolom numerik untuk clustering.")
    else:
        st.write("Kolom numerik yang tersedia:", numerical_cols)

        data = df[numerical_cols]

        # =====================================
        # 3. NORMALISASI DATA
        # =====================================
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # =====================================
        # 4. ELBOW METHOD
        # =====================================
        st.subheader("Elbow Method")
        sse = []
        K = range(1, 11)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            sse.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, sse, marker='o')
        ax.set_xlabel("Jumlah Cluster (k)")
        ax.set_ylabel("SSE")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # =====================================
        # 5. PILIH JUMLAH CLUSTER
        # =====================================
        st.subheader("Clustering")
        k = st.slider("Pilih jumlah cluster:", 2, 10, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled_data)

        st.write("Hasil Clustering:")
        st.dataframe(df)

        # =====================================
        # 6. VISUALISASI CLUSTER
        # =====================================
        if len(numerical_cols) >= 2:
            fig2, ax2 = plt.subplots()
            ax2.scatter(scaled_data[:, 0], scaled_data[:, 1], c=df["Cluster"], cmap="viridis")
            ax2.set_xlabel(numerical_cols[0])
            ax2.set_ylabel(numerical_cols[1])
            ax2.set_title("Visualisasi Cluster (2 fitur pertama)")
            st.pyplot(fig2)
else:
    st.info("Silakan upload file CSV untuk memulai.")
