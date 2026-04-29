import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# PATHS (IGUAL QUE OTROS SCRIPTS)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "..", "data", "final", "final_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "final")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_FILE)

# =========================
# TARGET
# =========================

df["target"] = df["FTR"].map({
    "H": 0,
    "D": 1,
    "A": 2
})

# =========================
# DROP COLUMNAS NO USABLES
# =========================

df = df.drop(columns=[
    "FTR",
    "Date",
    "HomeTeam",
    "AwayTeam"
])

# =========================
# FEATURES / TARGET
# =========================

X = df.drop(columns=["target"])
y = df["target"]

# =========================
# TRAIN / TEST (TEMPORAL)
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# =========================
# ESCALADO
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# (OPCIONAL) GUARDAR OUTPUT
# =========================

import numpy as np

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# =========================
# INFO
# =========================

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)

print("✅ Dataset listo para ML")