
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

st.set_page_config(page_title="Penguins: CBR / Classification / Clustering", layout="wide")

st.title("Penguins: CBR, Classification, and Clustering — Streamlit App")
st.markdown("""
Implementasi 3 metode:
- **Classification**: Random Forest untuk memprediksi `species`.
- **Clustering**: KMeans untuk menemukan kelompok.
- **Case-Based Reasoning (CBR)**: k-NN retrieval untuk menampilkan kasus paling mirip.
""")

# Load data (default: uploaded file or built-in penguins.csv)
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

uploaded = st.sidebar.file_uploader("Upload CSV (penguins.csv recommended)", type=["csv"])
default_path = "/mnt/data/penguins.csv"
data_file = uploaded if uploaded is not None else default_path

try:
    df = load_data(data_file)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

st.sidebar.write("Dataset preview")
st.sidebar.dataframe(df.head())

# Basic preprocessing
def preprocess(df, dropna=True):
    df = df.copy()
    # Keep commonly used numeric features
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island']
    # Some penguins datasets have slightly different names; try to select what's available
    available = [c for c in features if c in df.columns]
    # For classification we need species
    if 'species' in df.columns:
        available_with_target = available + ['species']
    else:
        available_with_target = available
    df = df[[c for c in available_with_target if c in df.columns]]
    if dropna:
        df = df.dropna().reset_index(drop=True)
    # Encode categorical
    le_sex = LabelEncoder()
    le_island = LabelEncoder()
    encoders = {}
    if 'sex' in df.columns:
        df['sex_enc'] = le_sex.fit_transform(df['sex'].astype(str))
        encoders['sex'] = le_sex
    if 'island' in df.columns:
        df['island_enc'] = le_island.fit_transform(df['island'].astype(str))
        encoders['island'] = le_island
    return df, encoders

df_clean, encoders = preprocess(df, dropna=True)
st.write(f"Loaded {len(df)} rows — after dropna: {len(df_clean)} rows")
st.dataframe(df_clean.head())

mode = st.sidebar.selectbox("Pilih mode", ["Classification", "Clustering", "Case-Based Reasoning (CBR)"])

numeric_features = [c for c in df_clean.columns if df_clean[c].dtype in [np.float64, np.int64]]
# Keep only columns that are meaningful
feature_cols = [c for c in numeric_features if c not in ['species']]

if mode == "Classification":
    st.header("Classification — Predict `species`")
    if 'species' not in df_clean.columns:
        st.error("Kolom `species` tidak ditemukan dalam dataset. Classification tidak tersedia.")
    else:
        X = df_clean[feature_cols]
        y = df_clean['species']
        # Encode target
        le_target = LabelEncoder()
        y_enc = le_target.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model: RandomForest — Accuracy (test): **{acc:.3f}**")
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix (rows=true, cols=pred):")
        st.write(pd.DataFrame(cm, index=le_target.classes_, columns=le_target.classes_))
        # User input for prediction
        st.subheader("Predict a custom penguin")
        input_vals = {}
        for col in feature_cols:
            mn = float(df_clean[col].min())
            mx = float(df_clean[col].max())
            mean = float(df_clean[col].mean())
            input_vals[col] = st.sidebar.slider(col, mn, mx, mean)
        inp = np.array([input_vals[c] for c in feature_cols]).reshape(1, -1)
        inp_s = scaler.transform(inp)
        pred = clf.predict(inp_s)[0]
        pred_proba = clf.predict_proba(inp_s)[0]
        st.write("Predicted species:", le_target.inverse_transform([pred])[0])
        proba_df = pd.DataFrame([pred_proba], columns=le_target.classes_)
        st.write("Probabilities:")
        st.dataframe(proba_df.T)

elif mode == "Clustering":
    st.header("Clustering — KMeans")
    st.write("KMeans clustering on numeric features.")
    X = df_clean[feature_cols].copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    k = st.sidebar.slider("Number of clusters (k)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_s)
    st.write(f"Assigned clusters (k={k})")
    df_clean['cluster'] = labels
    st.dataframe(df_clean[[*feature_cols, 'cluster']].head(10))
    # PCA scatter
    pca = PCA(n_components=2)
    X_p = pca.fit_transform(X_s)
    fig, ax = plt.subplots()
    ax.scatter(X_p[:,0], X_p[:,1], c=labels)
    ax.set_title("PCA projection of clusters")
    st.pyplot(fig)

    st.subheader("Cluster centers (inverse scaled)")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=feature_cols)
    st.dataframe(centers_df)

elif mode.startswith("Case-Based"):
    st.header("Case-Based Reasoning (CBR) — k-NN retrieval")
    st.write("Find the k most similar penguins to a query case (using Euclidean on numeric features).")
    X = df_clean[feature_cols].copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_s)
    # user input case
    st.subheader("Query case (use sidebar)")
    query_vals = {}
    for col in feature_cols:
        mn = float(df_clean[col].min())
        mx = float(df_clean[col].max())
        mean = float(df_clean[col].mean())
        query_vals[col] = st.sidebar.slider(f"Query — {col}", mn, mx, mean)
    q = np.array([query_vals[c] for c in feature_cols]).reshape(1, -1)
    q_s = scaler.transform(q)
    dists, idxs = nbrs.kneighbors(q_s, n_neighbors=st.sidebar.slider("k (neighbors)", 1, 10, 5))
    st.write("Closest cases (distance, index):")
    res = []
    for dist, idx in zip(dists[0], idxs[0]):
        row = df_clean.iloc[idx].to_dict()
        row['_dist'] = float(dist)
        res.append(row)
    st.dataframe(pd.DataFrame(res))

st.sidebar.markdown("---")
st.sidebar.write("Deployment:")
st.sidebar.markdown("""
To deploy this app publicly:
1. Install dependencies (requirements.txt).
2. Run `streamlit run streamlit_app.py`.
3. Use Streamlit Cloud, Heroku, or any VM to host the app.
""")
