import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Clustering Wine")

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

# ============================================
# RINGKASAN HASIL CLUSTERING
# ============================================
st.subheader("Ringkasan Hasil Clustering")

summary = df.groupby("Cluster")[numerical_cols].mean()

st.write("""
Berikut adalah ringkasan karakteristik masing-masing cluster berdasarkan nilai rata-rata
dari fitur numerik. Ringkasan ini membantu memahami pola dan perbedaan antar kelompok wine.
""")

st.dataframe(summary)

# Interpretasi simple
st.write("### Interpretasi Singkat")
for cluster in summary.index:
    st.write(f"**Cluster {cluster}**:")
    dominant_features = summary.loc[cluster].sort_values(ascending=False).head(3)
    st.write(f"- Memiliki nilai tinggi pada: {', '.join(dominant_features.index)}")
    st.write(f"- Rata-rata nilai terbesar: {dominant_features.iloc[0]:.2f}")
    st.write("---")

