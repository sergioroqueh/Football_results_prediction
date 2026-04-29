import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "final")

# =========================
# LOAD DATA
# =========================

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Datos cargados ✅")
print("Train:", X_train.shape, "| Test:", X_test.shape)

# =========================
# FUNCIÓN EVALUACIÓN
# =========================

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print("\n" + "="*40)
    print(f"Modelo: {name}")
    print("="*40)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# =========================
# MODELO 1 → LOGISTIC REGRESSION
# =========================

model_lr = LogisticRegression(max_iter=1000)

evaluate_model("Logistic Regression", model_lr, X_train, y_train, X_test, y_test)


# =========================
# MODELO 2 → RANDOM FOREST
# =========================

model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

evaluate_model("Random Forest", model_rf, X_train, y_train, X_test, y_test)


# =========================
# (OPCIONAL) MODELO 3 → EXTRA RANDOM FOREST
# =========================

model_rf_big = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

evaluate_model("Random Forest (big)", model_rf_big, X_train, y_train, X_test, y_test)

print("\n🔥 Entrenamiento completado")