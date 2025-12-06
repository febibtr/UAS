import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load CSV
df = pd.read_csv("penguins.csv")

# Hapus NaN
df = df.dropna()

# Label Encoder
le_species = LabelEncoder()
le_island = LabelEncoder()
le_sex = LabelEncoder()

df["species"] = le_species.fit_transform(df["species"])
df["island"] = le_island.fit_transform(df["island"])
df["sex"] = le_sex.fit_transform(df["sex"])

# Fitur dan Target
X = df.drop("species", axis=1)
y = df["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan SEMUA ke PKL
pickle.dump(df, open("data_penguins.pkl", "wb"))
pickle.dump(X_train_scaled, open("X_train.pkl", "wb"))
pickle.dump(X_test_scaled, open("X_test.pkl", "wb"))
pickle.dump(y_train, open("y_train.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))
pickle.dump(scaler, open("scaler_penguin.pkl", "wb"))

# Simpan encoder
pickle.dump(le_species, open("encoder_species.pkl", "wb"))
pickle.dump(le_island, open("encoder_island.pkl", "wb"))
pickle.dump(le_sex, open("encoder_sex.pkl", "wb"))

print("Semua data berhasil disimpan ke PKL!")
