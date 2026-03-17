import pandas as pd
import numpy as np
from src.utils.config import POWER_CONFERENCES, SELECTION_SUNDAY
from src.utils.db import query_df, db_conn, upsert_df
from src.features.team_ratings import build_team_feature_vector
from src.features.adjustments import get_rest_days, apply_neutral_site_correction

MATCHUP_FEATURES = [
    "adj_o_diff",
    "adj_d_diff",
    "avg_tempo",
    "tempo_diff",
    "barthag_diff",
    "efg_o_diff",
    "efg_d_diff",
    "to_rate_diff",
    "or_rate_diff",
    "ft_rate_diff",
    "three_pt_rate_diff",
    "three_pt_pct_diff",
    "two_pt_pct_diff",
    "sos_diff",
    "conf_power_a",
    "conf_power_b",
    "days_rest_diff",
    "round_number",
    "seed_diff",
]


def _conf_power(conf: str) -> float:
    """1.0 if power conference, 0.0 otherwise."""
    if not conf:
        return 0.0
    return 1.0 if any(pc.lower() in conf.lower() for pc in POWER_CONFERENCES) else 0.0


def build_matchup_features(team_a: str, team_b: str,
                            year: int, round_num: int,
                            seed_a: int = None, seed_b: int = None,
                            game_date: str = None,
                            as_of_date: str = None) -> pd.Series:
    """
    Construct feature row for a matchup. Team A = higher seed (favorite).
    Returns pd.Series with MATCHUP_FEATURES columns.
    """
    ra = build_team_feature_vector(team_a, year, as_of_date)
    rb = build_team_feature_vector(team_b, year, as_of_date)

    if not ra or not rb:
        # Return NaN series if we can't find ratings
        return pd.Series({f: np.nan for f in MATCHUP_FEATURES})

    ra = apply_neutral_site_correction(ra)
    rb = apply_neutral_site_correction(rb)

    # Rest days
    rest_a = get_rest_days(team_a, game_date, year, round_num=round_num) if game_date else 7
    rest_b = get_rest_days(team_b, game_date, year, round_num=round_num) if game_date else 7

    # Seed differential
    sa = seed_a if seed_a is not None else (ra.get("seed") or 8)
    sb = seed_b if seed_b is not None else (rb.get("seed") or 8)

    features = {
        # Efficiency differentials (Team A perspective)
        "adj_o_diff": (ra.get("adj_o", 100) or 100) - (rb.get("adj_d", 100) or 100),
        "adj_d_diff": (ra.get("adj_d", 100) or 100) - (rb.get("adj_o", 100) or 100),
        "avg_tempo": ((ra.get("adj_t", 68) or 68) + (rb.get("adj_t", 68) or 68)) / 2,
        "tempo_diff": abs((ra.get("adj_t", 68) or 68) - (rb.get("adj_t", 68) or 68)),
        "barthag_diff": (ra.get("barthag", 0.5) or 0.5) - (rb.get("barthag", 0.5) or 0.5),

        # Four Factors
        "efg_o_diff": (ra.get("efg_o", 50) or 50) - (rb.get("efg_d", 50) or 50),
        "efg_d_diff": (ra.get("efg_d", 50) or 50) - (rb.get("efg_o", 50) or 50),
        "to_rate_diff": (ra.get("to_rate_o", 18) or 18) - (rb.get("to_rate_d", 18) or 18),
        "or_rate_diff": (ra.get("or_rate_o", 28) or 28) - (rb.get("or_rate_d", 28) or 28),
        "ft_rate_diff": (ra.get("ft_rate_o", 30) or 30) - (rb.get("ft_rate_d", 30) or 30),

        # Shot profile
        "three_pt_rate_diff": (ra.get("three_pt_rate_o", 35) or 35) - (rb.get("three_pt_rate_d", 35) or 35),
        "three_pt_pct_diff": (ra.get("three_pt_pct_o", 33) or 33) - (rb.get("three_pt_pct_d", 33) or 33),
        "two_pt_pct_diff": (ra.get("two_pt_pct_o", 50) or 50) - (rb.get("two_pt_pct_d", 50) or 50),

        # Context
        "sos_diff": (ra.get("sos", 0) or 0) - (rb.get("sos", 0) or 0),
        "conf_power_a": _conf_power(ra.get("conf", "")),
        "conf_power_b": _conf_power(rb.get("conf", "")),
        "days_rest_diff": rest_a - rest_b,
        "round_number": round_num,
        "seed_diff": sa - sb,  # negative = team_a is the bigger favorite (team_a has lower seed number)
    }

    return pd.Series(features)


def build_training_matrix(historical_df: pd.DataFrame = None):
    """
    Build X, y_spread, y_total from historical_results + torvik_ratings.
    If historical_df is None, reads from DB.
    Returns (X, y_spread, y_total) — all aligned DataFrames/Series.
    """
    if historical_df is None:
        historical_df = query_df("SELECT * FROM historical_results")

    if historical_df.empty:
        raise ValueError("No historical data in DB. Run build_historical_dataset() first.")

    # Only use games with scores
    df = historical_df.dropna(subset=["score1", "score2"]).copy()
    df = df[df["score1"] > 0]

    rows = []
    for _, game in df.iterrows():
        year = int(game["year"])
        round_num = int(game["round_number"]) if pd.notna(game["round_number"]) else 1

        # Team A = winner or team with better seed (lower number = better)
        seed1 = game.get("seed1") or 8
        seed2 = game.get("seed2") or 8
        score1 = float(game["score1"])
        score2 = float(game["score2"])

        # Assign team_a as the team with better (lower) seed
        if seed1 <= seed2:
            team_a, team_b = game["team1"], game["team2"]
            sa, sb = int(seed1), int(seed2)
            actual_margin = score1 - score2
        else:
            team_a, team_b = game["team2"], game["team1"]
            sa, sb = int(seed2), int(seed1)
            actual_margin = score2 - score1

        actual_total = score1 + score2
        game_date = game.get("game_date") or f"{year}-03-15"

        feats = build_matchup_features(
            team_a, team_b, year, round_num,
            seed_a=sa, seed_b=sb,
            game_date=str(game_date),
            as_of_date=SELECTION_SUNDAY.get(year),
        )

        if feats.isna().all():
            continue

        row = feats.to_dict()
        row.update({
            "year": year,
            "game_date": game_date,
            "team_a": team_a,
            "team_b": team_b,
            "seed_a": sa,
            "seed_b": sb,
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "market_spread": None,
            "market_total": None,
        })
        rows.append(row)

    if not rows:
        raise ValueError("No feature rows built. Check that torvik_ratings table is populated.")

    result_df = pd.DataFrame(rows)

    # Store in mm_training_data
    with db_conn() as conn:
        conn.execute("DELETE FROM mm_training_data")
    training_cols = [
        "year", "game_date", "team_a", "team_b", "seed_a", "seed_b", "round_number",
        "adj_o_diff", "adj_d_diff", "avg_tempo", "tempo_diff", "barthag_diff",
        "efg_o_diff", "efg_d_diff", "to_rate_diff", "or_rate_diff", "ft_rate_diff",
        "three_pt_rate_diff", "three_pt_pct_diff", "two_pt_pct_diff",
        "sos_diff", "conf_power_a", "conf_power_b", "days_rest_diff",
        "actual_margin", "actual_total", "market_spread", "market_total"
    ]
    store_df = result_df[[c for c in training_cols if c in result_df.columns]].copy()
    upsert_df(store_df, "mm_training_data", if_exists="append")

    X = result_df[MATCHUP_FEATURES].copy()
    y_spread = result_df["actual_margin"]
    y_total = result_df["actual_total"]

    return X, y_spread, y_total
