import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ============================
# 1. LOAD DATA CSV (langsung)
# ============================
path_csv = "data_saya.csv"     # ← ganti sesuai lokasi file Anda
df = pd.read_csv(path_csv)

print("Data Loaded:")
print(df.head())

# ============================================
# 2. PILIH KOLOM YANG MAU DI-CLUSTERING
# ============================================
# Misalnya hanya kolom numerik
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\nKolom yang digunakan untuk clustering:", numerical_cols)

data = df[numerical_cols]

# =====================================
# 3. NORMALISASI DATA (sangat penting)
# =====================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# =====================================
# 4. MENENTUKAN JUMLAH CLUSTER (ELBOW)
# =====================================
sse = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(K, sse, 'o-')
plt.title("Metode Elbow")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("SSE / Inertia")
plt.grid(True)
plt.show()

# =====================================================
# 5. K-MEANS CLUSTERING (misal pilih 3 cluster)
# =====================================================
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print("\nHasil Clustering:")
print(df[['Cluster'] + numerical_cols].head())

# =====================================================
# 6. VISUALISASI CLUSTER (jika fitur >= 2)
# =====================================================
if len(numerical_cols) >= 2:
    plt.figure(figsize=(7,5))
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=df['Cluster'], s=50)
    plt.title("Visualisasi Clustering (2 Fitur Pertama)")
    plt.xlabel(numerical_cols[0])
    plt.ylabel(numerical_cols[1])
    plt.show()
