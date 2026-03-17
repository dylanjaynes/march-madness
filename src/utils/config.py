import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "db" / "mm_model.db"
MODELS_DIR = BASE_DIR / "models"

# Ensure dirs exist
for d in [RAW_DIR, PROCESSED_DIR, DB_PATH.parent, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Tournament years (2020 cancelled)
TOURNAMENT_YEARS = [
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2021, 2022, 2023, 2024, 2025, 2026,
]

# Selection Sunday dates (MMDD) — used to pull Torvik ratings as-of that date
# preventing post-Selection Sunday data leakage in training
SELECTION_SUNDAY = {
    2008: "0316",
    2009: "0315",
    2010: "0314",
    2011: "0313",
    2012: "0311",
    2013: "0317",
    2014: "0316",
    2015: "0315",
    2016: "0313",
    2017: "0312",
    2018: "0311",
    2019: "0317",
    2021: "0314",
    2022: "0313",
    2023: "0312",
    2024: "0317",
    2025: "0316",
    2026: "0315",
}

# Model parameters
SPREAD_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "random_state": 42,
}

TOTAL_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}

# Hybrid model thresholds
MISMATCH_SEED_DIFF_THRESHOLD = 5
MISMATCH_BARTHAG_THRESHOLD = 0.3

# A game is "competitive" for ATS betting purposes when the market spread
# is within a range the model can meaningfully compete with.
# Backtest shows: |mkt| <= 14 → 65.3% ATS (n=320, p<0.0001)
#                 |mkt| > 14  → 59.6% ATS (n=52, no signal)
COMPETITIVE_SPREAD_THRESHOLD = 14.0

# Competitive game model (XGBoost, same architecture as SPREAD_MODEL_PARAMS)
COMPETITIVE_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "random_state": 42,
}

# Mismatch game model (Ridge — no prediction ceiling)
MISMATCH_MODEL_PARAMS = {"alpha": 1.0}

# Adjustments
NEUTRAL_SITE_ADJUSTMENT = 0
TOURNAMENT_PACE_HAIRCUT = -2.5
SPREAD_STD_DEV = 12.0  # Residual std for win probability — matches walk-forward backtest RMSE
# COVERAGE_STD = 12.0  # legacy single-std constant (replaced by two-tier below)
COMPETITIVE_COVERAGE_STD = 10.5   # competitive games are more predictable
MISMATCH_COVERAGE_STD = 16.0      # blowout games are highly variable

# Conference power conferences
POWER_CONFERENCES = {"ACC", "Big Ten", "SEC", "Big 12", "Big East", "Pac-12"}

# Edge thresholds for display
EDGE_THRESHOLD_LOW = 2.0
EDGE_THRESHOLD_MED = 3.0
EDGE_THRESHOLD_HIGH = 5.0

# Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "basketball_ncaab"

# BartTorvik
TORVIK_BASE = "https://barttorvik.com"

# Round name mapping
ROUND_NAMES = {
    1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"
}
ROUND_NUMBERS = {v: k for k, v in ROUND_NAMES.items()}
