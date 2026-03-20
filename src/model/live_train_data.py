"""
live_train_data.py
------------------
Assembles training data for the live in-game spread model by joining:
  halftime_scores + historical_results + historical_lines + torvik_ratings

Target: actual_final_margin = score1 - score2 (team1 perspective, better seed = team1)

All optional box-score columns (efg, orb, to) are left as NaN when absent
from the halftime_scores table — they are NOT imputed here.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import TOURNAMENT_YEARS
from src.utils.db import query_df

# ---------------------------------------------------------------------------
# Feature list — ORDER MATTERS (must match model training and inference)
# ---------------------------------------------------------------------------
LIVE_FEATURES: list[str] = [
    "pregame_spread",
    "h1_margin",
    "h1_combined",
    "time_elapsed_pct",
    "time_remaining_pct",
    "efg_pct_diff",
    "orb_margin",
    "to_margin",
    "pace_surprise",
    "margin_surprise",
    "barthag_diff",
    "adj_o_diff",
    "adj_d_diff",
    "seed_diff",
    "round_number",
]


def build_live_training_data(
    train_years: list[int] | None = None,
    val_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df) where val_year games are held out.
    train_years defaults to all TOURNAMENT_YEARS except val_year.
    Both DataFrames contain all LIVE_FEATURES columns plus the target
    column 'actual_final_margin'.

    Convention: team1 = better seed (lower seed number).
    Where historical_results has seed1 <= seed2, team1 is already the
    better seed and the margin is score1 - score2.  When seed1 > seed2
    the roles are swapped and the sign of the margin is inverted so that
    a positive final margin always means the better seed won.
    """
    if train_years is None:
        train_years = [y for y in TOURNAMENT_YEARS if y != val_year]

    all_years = sorted(set(train_years) | {val_year})

    # ------------------------------------------------------------------
    # 1. Load source tables
    # ------------------------------------------------------------------
    placeholders = ",".join("?" * len(all_years))

    hs = query_df(
        f"SELECT * FROM halftime_scores WHERE year IN ({placeholders})",
        params=all_years,
    )
    hr = query_df(
        f"SELECT * FROM historical_results WHERE year IN ({placeholders})",
        params=all_years,
    )
    hl = query_df(
        f"SELECT * FROM historical_lines WHERE year IN ({placeholders})",
        params=all_years,
    )
    tr = query_df(
        f"SELECT year, team, barthag, adj_o, adj_d FROM torvik_ratings WHERE year IN ({placeholders})",
        params=all_years,
    )

    if hs.empty:
        raise ValueError(
            "halftime_scores table is empty — run live data ingestion first."
        )
    if hr.empty:
        raise ValueError(
            "historical_results table is empty — run build_historical_dataset() first."
        )

    # ------------------------------------------------------------------
    # 2. Merge halftime_scores + historical_results on (year, game_date, team1, team2)
    # ------------------------------------------------------------------
    join_keys = ["year", "game_date", "team1", "team2"]

    hr_slim = hr[
        ["year", "game_date", "team1", "team2",
         "score1", "score2", "seed1", "seed2", "round_number"]
    ].copy()

    merged = hs.merge(hr_slim, on=join_keys, how="inner")
    if merged.empty:
        raise ValueError(
            "No rows after joining halftime_scores to historical_results. "
            "Verify that team1/team2 naming is consistent across tables."
        )

    # ------------------------------------------------------------------
    # 3. Merge historical_lines
    # ------------------------------------------------------------------
    hl_slim = hl[
        ["year", "game_date", "team1", "team2",
         "spread_favorite", "spread_line", "total_line"]
    ].copy()

    merged = merged.merge(hl_slim, on=join_keys, how="left")

    # ------------------------------------------------------------------
    # 4. Merge torvik_ratings for team1 and team2
    # ------------------------------------------------------------------
    tr_t1 = tr.rename(columns={
        "team": "team1",
        "barthag": "barthag1",
        "adj_o": "adj_o1",
        "adj_d": "adj_d1",
    })
    tr_t2 = tr.rename(columns={
        "team": "team2",
        "barthag": "barthag2",
        "adj_o": "adj_o2",
        "adj_d": "adj_d2",
    })

    merged = merged.merge(tr_t1[["year", "team1", "barthag1", "adj_o1", "adj_d1"]],
                          on=["year", "team1"], how="left")
    merged = merged.merge(tr_t2[["year", "team2", "barthag2", "adj_o2", "adj_d2"]],
                          on=["year", "team2"], how="left")

    # ------------------------------------------------------------------
    # 5. Normalise orientation: team1 = better seed (lower seed number)
    # ------------------------------------------------------------------
    # seed1, seed2 come from historical_results; default to 8 if missing
    merged["seed1"] = merged["seed1"].fillna(8).astype(int)
    merged["seed2"] = merged["seed2"].fillna(8).astype(int)

    # When seed1 > seed2, team2 is actually the better seed.
    # Flip all team1/team2 columns so team1 is always the better seed.
    needs_flip = merged["seed1"] > merged["seed2"]

    def _swap(df: pd.DataFrame, col1: str, col2: str) -> None:
        """Swap values in col1 and col2 where needs_flip is True (in-place)."""
        tmp = df.loc[needs_flip, col1].copy()
        df.loc[needs_flip, col1] = df.loc[needs_flip, col2]
        df.loc[needs_flip, col2] = tmp

    for c1, c2 in [
        ("seed1", "seed2"),
        ("barthag1", "barthag2"),
        ("adj_o1", "adj_o2"),
        ("adj_d1", "adj_d2"),
        ("h1_score1", "h1_score2"),
        ("h1_efg1", "h1_efg2"),
        ("h1_orb1", "h1_orb2"),
        ("h1_to1", "h1_to2"),
        ("score1", "score2"),
    ]:
        if c1 in merged.columns and c2 in merged.columns:
            _swap(merged, c1, c2)

    # spread_favorite references team name — flip sign of spread_line when needed
    # (spread_line is positive, spread_favorite = name of favored team)
    # After the seed flip, team1 is the better-seeded team.  We want pregame_spread
    # to be positive when team1 is favored (i.e. spread_favorite == team1).
    # Note: team1/team2 columns themselves are NOT swapped — only score/stats columns
    # above are swapped so the orientation is consistent.  We do NOT rename team columns
    # because the join keys have already been used.

    # ------------------------------------------------------------------
    # 6. Compute features
    # ------------------------------------------------------------------
    df = merged.copy()

    # pregame_spread: positive = team1 favored
    df["pregame_spread"] = np.where(
        df["spread_favorite"] == df["team1"],
        df["spread_line"],
        -df["spread_line"],
    )
    # If spread_line is NaN the result will also be NaN — that is intentional
    # (we will drop NaN-target rows later but keep NaN-feature rows for the
    # calibrator which handles NaN features via XGBoost's native missing-value path)

    df["h1_score1"] = pd.to_numeric(df["h1_score1"], errors="coerce")
    df["h1_score2"] = pd.to_numeric(df["h1_score2"], errors="coerce")

    df["h1_margin"] = df["h1_score1"] - df["h1_score2"]
    df["h1_combined"] = df["h1_score1"] + df["h1_score2"]

    # Time — always halftime for historical data
    df["time_elapsed_pct"] = 20.0 / 40.0   # 0.5
    df["time_remaining_pct"] = 1.0 - df["time_elapsed_pct"]  # 0.5

    # Box-score efficiency — left as NaN if absent
    df["h1_efg1"] = pd.to_numeric(df.get("h1_efg1"), errors="coerce")
    df["h1_efg2"] = pd.to_numeric(df.get("h1_efg2"), errors="coerce")
    df["efg_pct_diff"] = df["h1_efg1"] - df["h1_efg2"]

    df["h1_orb1"] = pd.to_numeric(df.get("h1_orb1"), errors="coerce")
    df["h1_orb2"] = pd.to_numeric(df.get("h1_orb2"), errors="coerce")
    df["orb_margin"] = df["h1_orb1"] - df["h1_orb2"]

    df["h1_to1"] = pd.to_numeric(df.get("h1_to1"), errors="coerce")
    df["h1_to2"] = pd.to_numeric(df.get("h1_to2"), errors="coerce")
    # Reversed: positive = team1 protecting the ball
    df["to_margin"] = df["h1_to2"] - df["h1_to1"]

    # Pace surprise: actual H1 combined vs projected H1 pace
    df["pace_surprise"] = df["h1_combined"] - (df["total_line"] / 2.0)

    # Margin surprise: actual H1 margin vs expected H1 margin
    df["margin_surprise"] = df["h1_margin"] - (df["pregame_spread"] * 0.5)

    # Torvik differentials (team1 - team2)
    df["barthag_diff"] = df["barthag1"] - df["barthag2"]
    df["adj_o_diff"] = df["adj_o1"] - df["adj_o2"]
    df["adj_d_diff"] = df["adj_d1"] - df["adj_d2"]   # positive = team1 worse defense

    df["seed_diff"] = df["seed1"] - df["seed2"]  # negative = team1 is bigger favorite

    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")

    # ------------------------------------------------------------------
    # 7. Target variable
    # ------------------------------------------------------------------
    df["actual_final_margin"] = (
        pd.to_numeric(df["score1"], errors="coerce")
        - pd.to_numeric(df["score2"], errors="coerce")
    )

    # ------------------------------------------------------------------
    # 8. Select and drop rows where target is unknown
    # ------------------------------------------------------------------
    output_cols = LIVE_FEATURES + ["actual_final_margin", "year",
                                   "game_date", "team1", "team2"]
    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].copy()
    df = df.dropna(subset=["actual_final_margin"])

    # Sort by year so TimeSeriesSplit works correctly downstream
    df = df.sort_values("year").reset_index(drop=True)

    train_df = df[df["year"].isin(train_years)].copy()
    val_df = df[df["year"] == val_year].copy()

    print(
        f"[live_train_data] train rows: {len(train_df)} "
        f"({min(train_years) if train_years else '?'}–{max(train_years) if train_years else '?'})  "
        f"val rows: {len(val_df)} ({val_year})"
    )
    return train_df, val_df
