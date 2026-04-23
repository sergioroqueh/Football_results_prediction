import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# Leer todos los CSV procesados
files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_features.csv")]

dfs = []

for file in files:
    path = os.path.join(PROCESSED_DIR, file)
    df = pd.read_csv(path)

    # Asegurar fecha bien
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    dfs.append(df)

# Unir todo
df_all = pd.concat(dfs, ignore_index=True)

# Ordenar por fecha (CLAVE)
df_all = df_all.sort_values("Date").reset_index(drop=True)

# Guardar
output_path = os.path.join(PROCESSED_DIR, "all_seasons_combined.csv")
df_all.to_csv(output_path, index=False)

print("Dataset unificado creado ✅")