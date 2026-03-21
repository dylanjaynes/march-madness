"""
live_train_data.py
------------------
Assembles training data for the live in-game spread model.

New architecture (PBP upgrade):
  - For games with play-by-play data: generates 19 training rows per game
    at 2-minute intervals from minute 2 through minute 38.
  - For games without PBP (e.g. 2019): falls back to single halftime row
    from halftime_scores table.

Target: actual_final_margin = score1 - score2 (team1 perspective, better seed = team1)

All optional box-score columns are left as NaN when absent — XGBoost handles
missing values natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.pbp_parser import compute_game_state_at
from src.utils.config import TOURNAMENT_YEARS
from src.utils.db import query_df

# ---------------------------------------------------------------------------
# Feature list — ORDER MATTERS (must match model training and inference)
# ---------------------------------------------------------------------------
LIVE_FEATURES: list[str] = [
    "pregame_spread",
    "h1_margin",         # current margin at snapshot time (team1 perspective)
    "h1_combined",       # current combined score at snapshot time
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
    "pace_live",
    "momentum_5pos",
    "momentum_10pos",
    "possessions",
]

# Training timepoints: 2-minute intervals from minute 2 through minute 38
# (19 timepoints × ~665 games = ~12,600 training rows when PBP available)
TRAINING_TIMEPOINTS: list[float] = [
    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
    20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0,
]


# ---------------------------------------------------------------------------
# PBP feature helpers
# ---------------------------------------------------------------------------

def build_pbp_features_at_halftime(
    espn_game_id: str,
    home_team: str,
    away_team: str,
) -> dict | None:
    """
    Returns game state at halftime (t=20.0) computed from pbp_plays.
    Returns None if no PBP data exists for this game.
    """
    plays = query_df(
        "SELECT * FROM pbp_plays WHERE espn_game_id = ? ORDER BY time_elapsed ASC",
        params=[espn_game_id],
    )
    if plays.empty:
        return None

    plays_list = plays.to_dict("records")
    return compute_game_state_at(
        plays_list,
        at_time_elapsed=20.0,
        home_team=home_team,
        away_team=away_team,
    )


def _build_pbp_rows_for_game(
    espn_game_id: str,
    game_meta: dict,
    team1_is_home: bool,
    actual_final_margin: float,
) -> list[dict]:
    """
    Build one training row per TRAINING_TIMEPOINT for a game with PBP data.

    team1_is_home: True if the team1 in game_meta is the ESPN home team.
    actual_final_margin: final margin from team1 perspective (positive = team1 won).
    """
    plays = query_df(
        "SELECT * FROM pbp_plays WHERE espn_game_id = ? ORDER BY time_elapsed ASC",
        params=[espn_game_id],
    )
    if plays.empty:
        return []

    plays_list = plays.to_dict("records")
    rows = []

    for t in TRAINING_TIMEPOINTS:
        try:
            # Pass dummy team names — we orient manually below
            state = compute_game_state_at(
                plays_list,
                at_time_elapsed=t,
                home_team="home",
                away_team="away",
            )

            # Orient from home perspective → team1 perspective
            sign = 1.0 if team1_is_home else -1.0

            current_margin = sign * state["current_margin"]
            h1_combined    = state["score_home"] + state["score_away"]
            pace_surprise  = h1_combined - (game_meta.get("total_line", np.nan) / 2.0)
            margin_surprise = current_margin - game_meta["pregame_spread"] * (t / 40.0)

            # eFG% diff (team1 - team2)
            if state["efg_diff"] is not None:
                efg_pct_diff = sign * state["efg_diff"]
            else:
                efg_pct_diff = np.nan

            # ORB margin (team1 - team2 offensive rebounds)
            orb_margin = sign * state["orb_margin"]
            if orb_margin == 0 and state["orb_home"] == 0 and state["orb_away"] == 0:
                orb_margin = np.nan  # no data yet

            # TO margin: positive = team1 protecting ball (team2 has more TOs)
            # From compute_game_state_at: to_margin = to_away - to_home (positive = home protecting ball)
            to_margin = sign * state["to_margin"]
            if to_margin == 0 and state["to_home"] == 0 and state["to_away"] == 0:
                to_margin = np.nan

            # Momentum: positive = team1 gaining margin
            momentum_5pos  = sign * state["momentum_5pos"]
            momentum_10pos = sign * state["momentum_10pos"]

            possessions = state["possessions_home"] + state["possessions_away"]

            row = {
                # Identifiers (not features)
                "year":         game_meta["year"],
                "game_date":    game_meta["game_date"],
                "team1":        game_meta["team1"],
                "team2":        game_meta["team2"],
                "time_elapsed": t,
                # LIVE_FEATURES
                "pregame_spread":    game_meta["pregame_spread"],
                "h1_margin":         current_margin,
                "h1_combined":       float(h1_combined),
                "time_elapsed_pct":  t / 40.0,
                "time_remaining_pct": 1.0 - t / 40.0,
                "efg_pct_diff":      efg_pct_diff,
                "orb_margin":        float(orb_margin),
                "to_margin":         float(to_margin),
                "pace_surprise":     float(pace_surprise) if not np.isnan(pace_surprise) else np.nan,
                "margin_surprise":   float(margin_surprise),
                "barthag_diff":      game_meta.get("barthag_diff", np.nan),
                "adj_o_diff":        game_meta.get("adj_o_diff", np.nan),
                "adj_d_diff":        game_meta.get("adj_d_diff", np.nan),
                "seed_diff":         game_meta.get("seed_diff", np.nan),
                "round_number":      game_meta.get("round_number", np.nan),
                "pace_live":         float(state["pace_live"]) if state["pace_live"] else np.nan,
                "momentum_5pos":     float(momentum_5pos),
                "momentum_10pos":    float(momentum_10pos),
                "possessions":       float(possessions) if possessions > 0 else np.nan,
                # Target
                "actual_final_margin": actual_final_margin,
            }
            rows.append(row)
        except Exception:
            continue

    return rows


def _build_halftime_row_from_hs(hs_row: pd.Series, game_meta: dict) -> dict | None:
    """
    Build a single halftime training row from halftime_scores (fallback for games without PBP).
    """
    try:
        h1_score1  = float(hs_row.get("h1_score1", np.nan))
        h1_score2  = float(hs_row.get("h1_score2", np.nan))
        if np.isnan(h1_score1) or np.isnan(h1_score2):
            return None

        h1_margin   = h1_score1 - h1_score2
        h1_combined = h1_score1 + h1_score2
        t           = 20.0

        efg1 = pd.to_numeric(hs_row.get("h1_efg1"), errors="coerce")
        efg2 = pd.to_numeric(hs_row.get("h1_efg2"), errors="coerce")
        efg_pct_diff = (float(efg1) - float(efg2)) if (not np.isnan(float(efg1) if efg1 is not None else np.nan) and not np.isnan(float(efg2) if efg2 is not None else np.nan)) else np.nan

        orb1 = pd.to_numeric(hs_row.get("h1_orb1"), errors="coerce")
        orb2 = pd.to_numeric(hs_row.get("h1_orb2"), errors="coerce")
        orb_margin = (float(orb1) - float(orb2)) if (orb1 is not None and orb2 is not None) else np.nan

        to1 = pd.to_numeric(hs_row.get("h1_to1"), errors="coerce")
        to2 = pd.to_numeric(hs_row.get("h1_to2"), errors="coerce")
        to_margin = (float(to2) - float(to1)) if (to1 is not None and to2 is not None) else np.nan

        pregame_spread = game_meta["pregame_spread"]
        total_line     = game_meta.get("total_line", np.nan)
        pace_surprise  = h1_combined - (total_line / 2.0) if not np.isnan(total_line) else np.nan
        margin_surprise = h1_margin - pregame_spread * 0.5

        return {
            "year":         game_meta["year"],
            "game_date":    game_meta["game_date"],
            "team1":        game_meta["team1"],
            "team2":        game_meta["team2"],
            "time_elapsed": t,
            "pregame_spread":     pregame_spread,
            "h1_margin":          h1_margin,
            "h1_combined":        h1_combined,
            "time_elapsed_pct":   t / 40.0,
            "time_remaining_pct": 1.0 - t / 40.0,
            "efg_pct_diff":       efg_pct_diff,
            "orb_margin":         orb_margin,
            "to_margin":          to_margin,
            "pace_surprise":      pace_surprise,
            "margin_surprise":    margin_surprise,
            "barthag_diff":       game_meta.get("barthag_diff", np.nan),
            "adj_o_diff":         game_meta.get("adj_o_diff", np.nan),
            "adj_d_diff":         game_meta.get("adj_d_diff", np.nan),
            "seed_diff":          game_meta.get("seed_diff", np.nan),
            "round_number":       game_meta.get("round_number", np.nan),
            # PBP-only features are NaN for fallback rows
            "pace_live":          np.nan,
            "momentum_5pos":      np.nan,
            "momentum_10pos":     np.nan,
            "possessions":        np.nan,
            "actual_final_margin": game_meta["actual_final_margin"],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def build_live_training_data(
    train_years: list[int] | None = None,
    val_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df) where val_year games are held out.

    For games with PBP data: generates 19 training rows per game at
    2-minute intervals (TRAINING_TIMEPOINTS).
    For games without PBP: generates 1 halftime row from halftime_scores.

    Convention: team1 = better seed (lower seed number), positive margin = team1 won.
    """
    if train_years is None:
        train_years = [y for y in TOURNAMENT_YEARS if y != val_year]

    all_years = sorted(set(train_years) | {val_year})
    placeholders = ",".join("?" * len(all_years))

    # ------------------------------------------------------------------
    # 1. Load source tables
    # ------------------------------------------------------------------
    hr = query_df(
        f"SELECT * FROM historical_results WHERE year IN ({placeholders})",
        params=all_years,
    )
    hl = query_df(
        f"SELECT * FROM historical_lines WHERE year IN ({placeholders})",
        params=all_years,
    )
    hs = query_df(
        f"SELECT * FROM halftime_scores WHERE year IN ({placeholders})",
        params=all_years,
    )
    tr = query_df(
        f"SELECT year, team, barthag, adj_o, adj_d FROM torvik_ratings WHERE year IN ({placeholders})",
        params=all_years,
    )

    if hr.empty:
        raise ValueError("historical_results table is empty.")

    # ------------------------------------------------------------------
    # 2. Merge historical_results + historical_lines + torvik_ratings
    # ------------------------------------------------------------------
    join_keys = ["year", "game_date", "team1", "team2"]

    hr_slim = hr[
        ["year", "game_date", "team1", "team2",
         "score1", "score2", "seed1", "seed2", "round_number", "espn_game_id"]
    ].copy()

    if not hl.empty:
        hl_slim = hl[
            ["year", "game_date", "team1", "team2",
             "spread_favorite", "spread_line", "total_line"]
        ].copy()
        merged = hr_slim.merge(hl_slim, on=join_keys, how="left")
    else:
        merged = hr_slim.copy()
        merged["spread_favorite"] = np.nan
        merged["spread_line"]     = np.nan
        merged["total_line"]      = np.nan

    if not hs.empty:
        # Drop espn_game_id from halftime_scores to avoid column collision
        # (espn_game_id already comes from historical_results)
        hs_merge = hs.drop(columns=["espn_game_id"], errors="ignore")
        merged = merged.merge(hs_merge, on=join_keys, how="left")

    # Torvik ratings
    if not tr.empty:
        tr_t1 = tr.rename(columns={"team": "team1", "barthag": "barthag1",
                                    "adj_o": "adj_o1", "adj_d": "adj_d1"})
        tr_t2 = tr.rename(columns={"team": "team2", "barthag": "barthag2",
                                    "adj_o": "adj_o2", "adj_d": "adj_d2"})
        merged = merged.merge(tr_t1[["year", "team1", "barthag1", "adj_o1", "adj_d1"]],
                              on=["year", "team1"], how="left")
        merged = merged.merge(tr_t2[["year", "team2", "barthag2", "adj_o2", "adj_d2"]],
                              on=["year", "team2"], how="left")
    else:
        for col in ["barthag1", "barthag2", "adj_o1", "adj_o2", "adj_d1", "adj_d2"]:
            merged[col] = np.nan

    # ------------------------------------------------------------------
    # 3. Orientation: team1 = better seed (lower seed number)
    # ------------------------------------------------------------------
    merged["seed1"] = merged["seed1"].fillna(8).astype(int)
    merged["seed2"] = merged["seed2"].fillna(8).astype(int)
    needs_flip = merged["seed1"] > merged["seed2"]

    def _swap(col1: str, col2: str) -> None:
        tmp = merged.loc[needs_flip, col1].copy()
        merged.loc[needs_flip, col1] = merged.loc[needs_flip, col2]
        merged.loc[needs_flip, col2] = tmp

    for c1, c2 in [
        ("seed1", "seed2"),
        ("barthag1", "barthag2"),
        ("adj_o1", "adj_o2"),
        ("adj_d1", "adj_d2"),
        ("score1", "score2"),
    ]:
        if c1 in merged.columns and c2 in merged.columns:
            _swap(c1, c2)

    # After flip, h1_score1/h1_score2 in halftime_scores need same treatment
    for c1, c2 in [("h1_score1", "h1_score2"), ("h1_efg1", "h1_efg2"),
                   ("h1_orb1", "h1_orb2"), ("h1_to1", "h1_to2")]:
        if c1 in merged.columns and c2 in merged.columns:
            _swap(c1, c2)

    # pregame_spread: positive = team1 (better seed) favored
    merged["pregame_spread"] = np.where(
        merged.get("spread_favorite") == merged["team1"],
        merged.get("spread_line", np.nan),
        -merged.get("spread_line", np.nan),
    )

    # Torvik differentials (team1 - team2)
    merged["barthag_diff"] = merged["barthag1"] - merged["barthag2"]
    merged["adj_o_diff"]   = merged["adj_o1"]   - merged["adj_o2"]
    merged["adj_d_diff"]   = merged["adj_d1"]   - merged["adj_d2"]
    merged["seed_diff"]    = merged["seed1"]     - merged["seed2"]

    merged["actual_final_margin"] = (
        pd.to_numeric(merged["score1"], errors="coerce")
        - pd.to_numeric(merged["score2"], errors="coerce")
    )

    # Drop rows with no final score
    merged = merged.dropna(subset=["actual_final_margin"])

    # ------------------------------------------------------------------
    # 4. Check which games have PBP data
    # ------------------------------------------------------------------
    pbp_game_ids = set()
    try:
        pbp_available = query_df(
            "SELECT DISTINCT espn_game_id FROM pbp_plays WHERE espn_game_id IS NOT NULL"
        )
        if not pbp_available.empty:
            pbp_game_ids = set(pbp_available["espn_game_id"].astype(str).tolist())
    except Exception:
        pass

    print(f"[live_train_data] PBP data available for {len(pbp_game_ids)} games")

    # ------------------------------------------------------------------
    # 5. Build training rows game by game
    # ------------------------------------------------------------------
    all_rows = []
    pbp_game_count = 0
    hs_game_count  = 0

    hs_index = {}
    if not hs.empty and "espn_game_id" in hs.columns:
        for _, row in hs.iterrows():
            hs_index[str(row.get("espn_game_id", ""))] = row
    # Also index hs by (year, game_date, team1, team2) as fallback
    hs_key_index = {}
    if not hs.empty:
        for _, row in hs.iterrows():
            k = (int(row["year"]), str(row["game_date"]), str(row["team1"]), str(row["team2"]))
            hs_key_index[k] = row

    for _, game in merged.iterrows():
        espn_id = str(game.get("espn_game_id", "") or "")

        game_meta = {
            "year":                int(game["year"]),
            "game_date":           str(game["game_date"]),
            "team1":               str(game["team1"]),
            "team2":               str(game["team2"]),
            "pregame_spread":      float(game["pregame_spread"]) if not pd.isna(game["pregame_spread"]) else 0.0,
            "total_line":          float(game["total_line"]) if not pd.isna(game.get("total_line", np.nan)) else np.nan,
            "barthag_diff":        float(game["barthag_diff"]) if not pd.isna(game["barthag_diff"]) else np.nan,
            "adj_o_diff":          float(game["adj_o_diff"])   if not pd.isna(game["adj_o_diff"])   else np.nan,
            "adj_d_diff":          float(game["adj_d_diff"])   if not pd.isna(game["adj_d_diff"])   else np.nan,
            "seed_diff":           float(game["seed_diff"]),
            "round_number":        float(game["round_number"]) if not pd.isna(game.get("round_number", np.nan)) else np.nan,
            "actual_final_margin": float(game["actual_final_margin"]),
        }

        # Try PBP path
        if espn_id and espn_id != "nan" and espn_id in pbp_game_ids:
            # Determine if team1 is the ESPN home team
            # Use final score comparison: compare ESPN home_score to score1
            try:
                last_play = query_df(
                    "SELECT home_score, away_score FROM pbp_plays "
                    "WHERE espn_game_id = ? ORDER BY time_elapsed DESC LIMIT 1",
                    params=[espn_id],
                )
                if not last_play.empty:
                    espn_home_final = int(last_play.iloc[0]["home_score"])
                    espn_away_final = int(last_play.iloc[0]["away_score"])
                    score1 = int(game["score1"])
                    score2 = int(game["score2"])
                    # If espn_home_final matches score1, team1=home; else team1=away
                    team1_is_home = abs(espn_home_final - score1) < abs(espn_home_final - score2)
                else:
                    team1_is_home = True
            except Exception:
                team1_is_home = True

            rows = _build_pbp_rows_for_game(
                espn_game_id=espn_id,
                game_meta=game_meta,
                team1_is_home=team1_is_home,
                actual_final_margin=game_meta["actual_final_margin"],
            )
            if rows:
                all_rows.extend(rows)
                pbp_game_count += 1
                continue

        # Fallback: halftime_scores
        hs_row = None
        if espn_id and espn_id in hs_index:
            hs_row = hs_index[espn_id]
        else:
            k = (game_meta["year"], game_meta["game_date"], game_meta["team1"], game_meta["team2"])
            hs_row = hs_key_index.get(k)

        if hs_row is not None:
            row = _build_halftime_row_from_hs(hs_row, game_meta)
            if row is not None:
                all_rows.append(row)
                hs_game_count += 1

    print(
        f"[live_train_data] {pbp_game_count} games × 19 timepoints (PBP)  "
        f"+ {hs_game_count} games × 1 timepoint (halftime fallback)  "
        f"= {len(all_rows)} total training rows"
    )

    if not all_rows:
        raise ValueError(
            "No training rows assembled — ensure pbp_plays or halftime_scores is populated."
        )

    # ------------------------------------------------------------------
    # 6. Build DataFrame and split train/val
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["year", "game_date"]).reset_index(drop=True)

    output_cols = LIVE_FEATURES + ["actual_final_margin", "year", "game_date", "team1", "team2"]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].copy()
    df = df.dropna(subset=["actual_final_margin"])

    train_df = df[df["year"].isin(train_years)].copy()
    val_df   = df[df["year"] == val_year].copy()

    print(
        f"[live_train_data] train rows: {len(train_df)} "
        f"({min(train_years) if train_years else '?'}–{max(train_years) if train_years else '?'})  "
        f"val rows: {len(val_df)} ({val_year})"
    )
    return train_df, val_df
