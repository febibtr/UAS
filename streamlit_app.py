import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Aplikasi Clustering K-Means dari GitHub RAW CSV")

# ============================================
# 1. LOAD CSV OTOMATIS DARI GITHUB
# ============================================
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/febibtr/UAS/refs/heads/main/wine-clustering.csv", sep=None, engine="python")
print(df.head())

st.write("Dataset berhasil dimuat:")
st.dataframe(df)

# ============================================
# 2. PILIH KOLOM NUMERIK
# ============================================
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
data = df[numerical_cols]

# ============================================
# 3. NORMALISASI
# ============================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# ============================================
# 4. ELBOW METHOD
# ============================================
st.subheader("Elbow Method")
sse = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(K, sse, marker='o')
st.pyplot(fig)

# ============================================
# 5. CLUSTERING
# ============================================
k = st.slider("Pilih jumlah cluster:", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

st.write("Hasil Clustering:")
st.dataframe(df)

# ============================================
# 6. VISUALISASI
# ============================================
fig2, ax2 = plt.subplots()
ax2.scatter(scaled_data[:, 0], scaled_data[:, 1], c=df["Cluster"])
ax2.set_title("Visualisasi Cluster")
st.pyplot(fig2)
