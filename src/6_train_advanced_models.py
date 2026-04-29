import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "final")

# =========================
# LOAD NUMPY
# =========================

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# =========================
# RECUPERAR DATAFRAME ORIGINAL (para crear nuevas features)
# =========================

df = pd.read_csv(os.path.join(DATA_DIR, "final_dataset.csv"))

# mismo split temporal
split_index = int(len(df) * 0.8)

df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()

# =========================
# NUEVAS FEATURES (CLAVE)
# =========================

def add_diff_features(df):
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["form_diff"] = df["home_points_5"] - df["away_points_5"]
    df["goal_diff_diff"] = df["home_diff_5"] - df["away_diff_5"]
    df["h2h_balance"] = df["h2h_home_wins_5"] - df["h2h_away_wins_5"]
    return df

df_train = add_diff_features(df_train)
df_test = add_diff_features(df_test)

# =========================
# LIMPIAR COLUMNAS
# =========================

drop_cols = ["FTR", "Date", "HomeTeam", "AwayTeam"]

df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)

# target
y_train = df_train["target"]
y_test = df_test["target"]

X_train = df_train.drop(columns=["target"])
X_test = df_test.drop(columns=["target"])

# =========================
# EVALUACIÓN
# =========================

def evaluate(name, model):
    print("\n" + "="*50)
    print(f"Modelo: {name}")
    print("="*50)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# =========================
# MODELO 1 → LOGISTIC BALANCED
# =========================

model_lr = LogisticRegression(max_iter=1000, class_weight="balanced")

evaluate("Logistic Regression (balanced + diff features)", model_lr)


# =========================
# MODELO 2 → XGBOOST 🔥
# =========================

model_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

evaluate("XGBoost", model_xgb)

print("\n🔥 Modelos avanzados ejecutados")