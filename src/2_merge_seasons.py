import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "final")

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_features.csv")]

dfs = []

for file in files:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, file))
    df["Date"] = pd.to_datetime(df["Date"])
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sort_values("Date").reset_index(drop=True)

output_path = os.path.join(OUTPUT_DIR, "dataset_no_h2h.csv")
df_all.to_csv(output_path, index=False)

print("✅ Dataset unificado (sin H2H)")