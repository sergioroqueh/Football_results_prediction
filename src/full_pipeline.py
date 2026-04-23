import os
import pandas as pd
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "final_dataset.csv")

# =========================
# FUNCIONES AUXILIARES
# =========================

def result_to_points(r):
    return 3 if r == "W" else 1 if r == "D" else 0

def weighted_points(results):
    weights = [1, 2, 3, 4, 5]
    return sum(result_to_points(r) * w for r, w in zip(results, weights[-len(results):]))

def elo_expected(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(r_a, r_b, score_a, k=20):
    exp_a = elo_expected(r_a, r_b)
    return r_a + k * (score_a - exp_a)

# =========================
# CARGAR Y UNIR RAW
# =========================

files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
dfs = []

for file in files:
    path = os.path.join(RAW_DIR, file)
    df = pd.read_csv(path, sep=";")

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.drop_duplicates()
    df = df.dropna(how='all')

    required_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{file} → falta columna {col}")

    df = df.dropna(subset=required_cols)

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values("Date").reset_index(drop=True)

print(f"Total partidos: {len(df)}")

# =========================
# ESTADOS
# =========================

hist_results = defaultdict(lambda: deque(maxlen=5))
hist_goals_for = defaultdict(lambda: deque(maxlen=5))
hist_goals_against = defaultdict(lambda: deque(maxlen=5))

hist_home = defaultdict(lambda: deque(maxlen=5))
hist_away = defaultdict(lambda: deque(maxlen=5))

elo = defaultdict(lambda: 1500)
last_match_date = {}

# H2H
h2h_results = defaultdict(lambda: deque(maxlen=5))
h2h_gf = defaultdict(lambda: deque(maxlen=5))
h2h_ga = defaultdict(lambda: deque(maxlen=5))

features = []

# =========================
# LOOP GLOBAL (CLAVE)
# =========================

for _, row in df.iterrows():

    home = row["HomeTeam"]
    away = row["AwayTeam"]
    hg = row["FTHG"]
    ag = row["FTAG"]
    date = row["Date"]

    # Resultado match
    if hg > ag:
        hr, ar = "W", "L"
        score_home = 1
        h2h_r = "H"
    elif hg < ag:
        hr, ar = "L", "W"
        score_home = 0
        h2h_r = "A"
    else:
        hr, ar = "D", "D"
        score_home = 0.5
        h2h_r = "D"

    # =====================
    # FEATURES EQUIPO
    # =====================

    home_points_5 = sum(result_to_points(r) for r in hist_results[home])
    away_points_5 = sum(result_to_points(r) for r in hist_results[away])

    home_weighted = weighted_points(hist_results[home])
    away_weighted = weighted_points(hist_results[away])

    home_diff_5 = sum(hist_goals_for[home]) - sum(hist_goals_against[home])
    away_diff_5 = sum(hist_goals_for[away]) - sum(hist_goals_against[away])

    home_home_points = sum(result_to_points(r) for r in hist_home[home])
    away_away_points = sum(result_to_points(r) for r in hist_away[away])

    elo_home = elo[home]
    elo_away = elo[away]

    rest_home = (date - last_match_date.get(home, date)).days
    rest_away = (date - last_match_date.get(away, date)).days

    # =====================
    # FEATURES H2H
    # =====================

    key = tuple(sorted([home, away]))

    past = h2h_results[key]

    h2h_matches = len(past)
    h2h_home_wins = sum(1 for r in past if r == "H")
    h2h_away_wins = sum(1 for r in past if r == "A")
    h2h_draws = sum(1 for r in past if r == "D")

    h2h_goal_diff = sum(h2h_gf[key]) - sum(h2h_ga[key])

    weights = [1,2,3,4,5]
    h2h_weighted = 0

    for r, w in zip(past, weights[-len(past):]):
        if r == "H":
            h2h_weighted += 3*w
        elif r == "D":
            h2h_weighted += 1*w

    # =====================
    # GUARDAR FEATURES
    # =====================

    features.append({
        "home_points_5": home_points_5,
        "away_points_5": away_points_5,
        "home_weighted": home_weighted,
        "away_weighted": away_weighted,
        "home_diff_5": home_diff_5,
        "away_diff_5": away_diff_5,
        "home_home_points": home_home_points,
        "away_away_points": away_away_points,
        "elo_home": elo_home,
        "elo_away": elo_away,
        "rest_home": rest_home,
        "rest_away": rest_away,

        "h2h_matches_5": h2h_matches,
        "h2h_home_wins_5": h2h_home_wins,
        "h2h_away_wins_5": h2h_away_wins,
        "h2h_draws_5": h2h_draws,
        "h2h_goal_diff_5": h2h_goal_diff,
        "h2h_weighted_score_5": h2h_weighted
    })

    # =====================
    # ACTUALIZAR HISTORIAL
    # =====================

    hist_results[home].append(hr)
    hist_results[away].append(ar)

    hist_goals_for[home].append(hg)
    hist_goals_for[away].append(ag)

    hist_goals_against[home].append(ag)
    hist_goals_against[away].append(hg)

    hist_home[home].append(hr)
    hist_away[away].append(ar)

    elo[home] = update_elo(elo[home], elo[away], score_home)
    elo[away] = update_elo(elo[away], elo[home], 1 - score_home)

    last_match_date[home] = date
    last_match_date[away] = date

    h2h_results[key].append(h2h_r)
    h2h_gf[key].append(hg)
    h2h_ga[key].append(ag)

# =========================
# FINAL DATASET
# =========================

df_features = pd.DataFrame(features)
df_final = pd.concat([df, df_features], axis=1)

df_final.to_csv(OUTPUT_FILE, index=False)

print("🔥 PIPELINE COMPLETO EJECUTADO")
print(f"Archivo final: {OUTPUT_FILE}")