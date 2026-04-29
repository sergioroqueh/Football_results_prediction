import os
import pandas as pd
from collections import defaultdict, deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================

def result_to_points(r):
    return 3 if r == "W" else 1 if r == "D" else 0

def weighted_points(results):
    weights = [1,2,3,4,5]
    return sum(result_to_points(r)*w for r, w in zip(results, weights[-len(results):]))

def elo_expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

def update_elo(a, b, score_a, k=20):
    return a + k * (score_a - elo_expected(a, b))

# =========================
# PROCESS FILES
# =========================

files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

for file in files:
    print(f"Procesando {file}...")

    path = os.path.join(RAW_DIR, file)
    df = pd.read_csv(path, sep=";")

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"])
    # df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # =========================
    # RESET POR TEMPORADA
    # =========================

    hist_results = defaultdict(lambda: deque(maxlen=5))
    hist_gf = defaultdict(lambda: deque(maxlen=5))
    hist_ga = defaultdict(lambda: deque(maxlen=5))
    hist_home = defaultdict(lambda: deque(maxlen=5))
    hist_away = defaultdict(lambda: deque(maxlen=5))

    elo = defaultdict(lambda: 1500)
    last_match = {}

    features = []

    # =========================
    # LOOP TEMPORADA
    # =========================

    for _, row in df.iterrows():

        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        date = row["Date"]

        # resultado
        if hg > ag:
            hr, ar = "W", "L"
            score_home = 1
        elif hg < ag:
            hr, ar = "L", "W"
            score_home = 0
        else:
            hr = ar = "D"
            score_home = 0.5

        # features
        features.append({
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,

            "home_points_5": sum(result_to_points(r) for r in hist_results[home]),
            "away_points_5": sum(result_to_points(r) for r in hist_results[away]),

            "home_weighted": weighted_points(hist_results[home]),
            "away_weighted": weighted_points(hist_results[away]),

            "home_diff_5": sum(hist_gf[home]) - sum(hist_ga[home]),
            "away_diff_5": sum(hist_gf[away]) - sum(hist_ga[away]),

            "home_home_points": sum(result_to_points(r) for r in hist_home[home]),
            "away_away_points": sum(result_to_points(r) for r in hist_away[away]),

            "elo_home": elo[home],
            "elo_away": elo[away],

            "rest_home": (date - last_match.get(home, date)).days,
            "rest_away": (date - last_match.get(away, date)).days,

            "FTR": "H" if hg>ag else "A" if hg<ag else "D"
        })

        # update
        hist_results[home].append(hr)
        hist_results[away].append(ar)

        hist_gf[home].append(hg)
        hist_gf[away].append(ag)

        hist_ga[home].append(ag)
        hist_ga[away].append(hg)

        hist_home[home].append(hr)
        hist_away[away].append(ar)

        elo[home] = update_elo(elo[home], elo[away], score_home)
        elo[away] = update_elo(elo[away], elo[home], 1-score_home)

        last_match[home] = date
        last_match[away] = date

    df_out = pd.DataFrame(features)

    out_path = os.path.join(OUTPUT_DIR, file.replace(".csv", "_features.csv"))
    df_out.to_csv(out_path, index=False)

print("✅ Temporadas procesadas")