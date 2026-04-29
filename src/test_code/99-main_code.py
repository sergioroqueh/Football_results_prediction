import os
import pandas as pd
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

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
# PIPELINE POR TEMPORADA
# =========================

def process_season(file_path):

    df = pd.read_csv(file_path, sep=";")

    # -------- LIMPIEZA --------
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.drop_duplicates()
    df = df.dropna(how='all')

    required_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna {col}")

    df = df.dropna(subset=required_cols)

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Fecha flexible (por si cambia formato)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # -------- ESTADO --------
    hist_results = defaultdict(lambda: deque(maxlen=5))
    hist_goals_for = defaultdict(lambda: deque(maxlen=5))
    hist_goals_against = defaultdict(lambda: deque(maxlen=5))

    hist_home = defaultdict(lambda: deque(maxlen=5))
    hist_away = defaultdict(lambda: deque(maxlen=5))

    elo = defaultdict(lambda: 1500)
    last_match_date = {}

    features = []

    # -------- LOOP --------
    for _, row in df.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg = row["FTHG"]
        ag = row["FTAG"]
        date = row["Date"]

        # Resultado
        if hg > ag:
            hr, ar = "W", "L"
            score_home = 1
        elif hg < ag:
            hr, ar = "L", "W"
            score_home = 0
        else:
            hr, ar = "D", "D"
            score_home = 0.5

        # FEATURES (antes)
        home_points_5 = sum(result_to_points(r) for r in hist_results[home])
        away_points_5 = sum(result_to_points(r) for r in hist_results[away])

        home_weighted = weighted_points(hist_results[home])
        away_weighted = weighted_points(hist_results[away])

        home_gf_5 = sum(hist_goals_for[home])
        home_ga_5 = sum(hist_goals_against[home])
        away_gf_5 = sum(hist_goals_for[away])
        away_ga_5 = sum(hist_goals_against[away])

        home_diff_5 = home_gf_5 - home_ga_5
        away_diff_5 = away_gf_5 - away_ga_5

        home_home_points = sum(result_to_points(r) for r in hist_home[home])
        away_away_points = sum(result_to_points(r) for r in hist_away[away])

        elo_home = elo[home]
        elo_away = elo[away]

        rest_home = (date - last_match_date.get(home, date)).days
        rest_away = (date - last_match_date.get(away, date)).days

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
        })

        # ACTUALIZAR
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

    df_features = pd.DataFrame(features)
    df_final = pd.concat([df, df_features], axis=1)

    return df_final


# =========================
# PROCESAR TODOS LOS CSV
# =========================

files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

print(f"Encontrados {len(files)} archivos")

for file in files:

    input_path = os.path.join(RAW_DIR, file)

    try:
        print(f"Procesando: {file}")

        df_processed = process_season(input_path)

        output_name = file.replace(".csv", "_features.csv")
        output_path = os.path.join(PROCESSED_DIR, output_name)

        df_processed.to_csv(output_path, index=False)

        print(f"OK → {output_name}")

    except Exception as e:
        print(f"ERROR en {file}: {e}")