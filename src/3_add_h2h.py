import os
import pandas as pd
from collections import defaultdict, deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "data", "final", "dataset_no_h2h.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "final", "final_dataset.csv")

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# H2H GLOBAL
# =========================

h2h_results = defaultdict(lambda: deque(maxlen=5))
h2h_gf = defaultdict(lambda: deque(maxlen=5))
h2h_ga = defaultdict(lambda: deque(maxlen=5))

features = []

for _, row in df.iterrows():

    home, away = row["HomeTeam"], row["AwayTeam"]

    key = tuple(sorted([home, away]))
    past = h2h_results[key]

    features.append({
        "h2h_matches_5": len(past),
        "h2h_home_wins_5": sum(1 for r in past if r=="H"),
        "h2h_away_wins_5": sum(1 for r in past if r=="A"),
        "h2h_draws_5": sum(1 for r in past if r=="D"),
        "h2h_goal_diff_5": sum(h2h_gf[key]) - sum(h2h_ga[key])
    })

    # actualizar
    if row["FTR"] == "H":
        res = "H"
        gf, ga = 1, 0
    elif row["FTR"] == "A":
        res = "A"
        gf, ga = 0, 1
    else:
        res = "D"
        gf = ga = 0

    h2h_results[key].append(res)
    h2h_gf[key].append(gf)
    h2h_ga[key].append(ga)

df_h2h = pd.DataFrame(features)
df_final = pd.concat([df, df_h2h], axis=1)

df_final.to_csv(OUTPUT_FILE, index=False)

print("🔥 Dataset final con H2H creado")