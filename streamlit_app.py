import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =====================================================
# CSS CUSTOM BIAR TAMPILAN LEBIH KEKINIAN
# =====================================================
st.markdown("""
<style>
.main {
    background-color: #f3f7fd;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.pred-box {
    background: linear-gradient(135deg, #4CAF50, #2E7D32);
    color: white;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD PKL
# =====================================================
df = pickle.load(open("data_penguins.pkl", "rb"))
X_train = pickle.load(open("X_train.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_train = pickle.load(open("y_train.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

scaler = pickle.load(open("scaler_penguin.pkl", "rb"))

le_species = pickle.load(open("encoder_species.pkl", "rb"))
le_island = pickle.load(open("encoder_island.pkl", "rb"))
le_sex = pickle.load(open("encoder_sex.pkl", "rb"))

# =====================================================
# TITLE
# =====================================================
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🐧 Prediksi Spesies Penguin</h1>", unsafe_allow_html=True)

# =====================================================
# TRAIN MODEL
# =====================================================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

# =====================================================
# FORM INPUT
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Input Data")

col1, col2 = st.columns(2)

with col1:
    island_input = st.selectbox("🏝️ Island:", le_island.classes_)
    bill_length = st.number_input("📏 bill_length_mm:", min_value=0.0, step=0.1)
    flipper = st.number_input("🦴 flipper_length_mm:", min_value=0.0, step=1.0)

with col2:
    bill_depth = st.number_input("📐 bill_depth_mm:", min_value=0.0, step=0.1)
    body_mass = st.number_input("⚖️ body_mass_g:", min_value=0.0, step=50.0)
    sex_input = st.selectbox("♂️ / ♀️ Sex:", le_sex.classes_)

st.markdown("</div>", unsafe_allow_html=True)



# =====================================================
# PREDIKSI
# =====================================================
input_df = pd.DataFrame([{
    "island": le_island.transform([island_input])[0],
    "bill_length_mm": bill_length,
    "bill_depth_mm": bill_depth,
    "flipper_length_mm": flipper,
    "body_mass_g": body_mass,
    "sex": le_sex.transform([sex_input])[0]
}])

input_scaled = scaler.transform(input_df)


st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Prediksi Spesies", use_container_width=True):
    pred = model.predict(input_scaled)[0]
    nama = le_species.inverse_transform([pred])[0]

    st.markdown(f"<div class='pred-box'>🐧 Hasil Prediksi: {nama}</div>", unsafe_allow_html=True)
