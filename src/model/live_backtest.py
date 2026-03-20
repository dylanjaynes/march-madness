"""
live_backtest.py
----------------
Backtests the live in-game spread model against held-out 2025 data.

IMPORTANT CAVEAT (logged prominently and included in all outputs):
  Historical backtest uses closing pre-game lines as a proxy for live halftime
  lines.  Live lines at halftime are typically tighter (less value available)
  than pre-game closing lines.  Treat the cover rates produced here as an
  *upper bound* on true expected performance.  The first valid backtest using
  actual live halftime odds will come from 2026 Odds API live spreads.

Scenario classification:
  FAV_AHEAD  : team1 (better seed, lower seed number) is currently leading
  DOG_WINNING: team2 is currently leading

Edge thresholds: |edge| >= 3, >= 5, >= 7
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR, PROCESSED_DIR
from src.utils.db import query_df
from src.model.live_train_data import LIVE_FEATURES, build_live_training_data
from src.model.live_train import formula_projected_margin


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKTEST_CAVEAT = (
    "IMPORTANT — UPPER BOUND: This backtest uses closing pre-game lines as a proxy "
    "for live halftime lines. Live halftime lines are typically tighter than pre-game "
    "closing lines, meaning less edge is available in practice. Cover rates shown here "
    "are an upper bound on real-world expected performance. The first valid backtest "
    "using actual live halftime odds (Odds API) will be from the 2026 tournament."
)

LOW_SAMPLE_THRESHOLD = 20
EDGE_THRESHOLDS = [3, 5, 7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_live_model():
    """
    Attempt to load the trained live model and calibrator.
    Raises FileNotFoundError if either artefact is missing.
    """
    model_path = MODELS_DIR / "live_spread_model.pkl"
    cal_path = MODELS_DIR / "live_spread_calibrator.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not cal_path.exists():
        raise FileNotFoundError(f"Trained calibrator not found: {cal_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)
    return model, calibrator


def _formula_predict_row(row: pd.Series) -> float:
    """
    Apply formula stub to a single data row, using NaN-safe defaults for
    missing box-score columns.
    """
    return formula_projected_margin(
        current_margin=float(row["h1_margin"]) if pd.notna(row.get("h1_margin")) else 0.0,
        pregame_spread=float(row["pregame_spread"]) if pd.notna(row.get("pregame_spread")) else 0.0,
        time_elapsed=20.0,
        time_remaining=20.0,
        efg_pct_diff=float(row["efg_pct_diff"]) if pd.notna(row.get("efg_pct_diff")) else 0.0,
        orb_margin=float(row["orb_margin"]) if pd.notna(row.get("orb_margin")) else 0.0,
        to_margin=float(row["to_margin"]) if pd.notna(row.get("to_margin")) else 0.0,
    )


def _cover_rate(df: pd.DataFrame) -> float:
    """
    Fraction of rows where the model's projected side covered the actual margin.

    bet_team1 is True when the model projects team1 margin > market spread,
    i.e. the model recommends taking team1 (the better seed).

    A team1 bet covers when: actual_final_margin > pregame_spread (proxy market)
    A team2 bet covers when: actual_final_margin < pregame_spread
    """
    if df.empty:
        return float("nan")

    # We bet the side the model favours vs the market proxy
    bet_team1 = df["projected_margin"] > df["pregame_spread"]

    actual_covers_team1 = df["actual_final_margin"] > df["pregame_spread"]

    covered = (bet_team1 & actual_covers_team1) | (~bet_team1 & ~actual_covers_team1)
    return float(covered.mean())


def _sample_flag(n: int) -> str:
    return " [low sample]" if n < LOW_SAMPLE_THRESHOLD else ""


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def run_live_backtest(val_year: int = 2025) -> dict:
    """
    For each game in halftime_training_data (val_year):
      - Compute projected_margin using trained live model at halftime (time_elapsed=20)
      - Compare to historical_lines.spread_line as proxy for live market
      - Grade: did bet side cover actual result?

    Returns results dict and saves to data/processed/live_backtest_results.json.

    IMPORTANT CAVEAT (log prominently and include in output):
    Historical backtest uses closing pre-game lines as proxy for live halftime lines.
    Live lines at halftime are typically tighter — treat cover rates as upper bound.
    2026 live data (actual Odds API live spreads) will be first valid backtest.
    """
    print("=" * 70)
    print("LIVE MODEL BACKTEST")
    print("=" * 70)
    print(f"\n*** {BACKTEST_CAVEAT}\n")

    # ------------------------------------------------------------------
    # 1. Load validation data
    # ------------------------------------------------------------------
    try:
        _, val_df = build_live_training_data(val_year=val_year)
    except ValueError as exc:
        # halftime_scores table is empty
        msg = str(exc)
        print(f"WARNING: {msg}")
        result = {
            "error": msg,
            "caveat": BACKTEST_CAVEAT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_results(result)
        return result

    if val_df.empty:
        msg = (
            f"No halftime data found for val_year={val_year}. "
            "Cannot run backtest — populate halftime_scores for that year first."
        )
        print(f"WARNING: {msg}")
        result = {
            "error": msg,
            "caveat": BACKTEST_CAVEAT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_results(result)
        return result

    print(f"Validation games (year={val_year}): {len(val_df)}")

    # ------------------------------------------------------------------
    # 2. Generate projected margins
    # ------------------------------------------------------------------
    use_model = True
    try:
        model, calibrator = _load_live_model()
        print("  Using trained XGBoost live model + isotonic calibrator.")
    except FileNotFoundError as exc:
        print(f"  WARNING: {exc}")
        print("  Falling back to formula stub (NaN-safe defaults for missing features).")
        use_model = False

    val_df = val_df.copy().reset_index(drop=True)

    if use_model:
        X_val = val_df[LIVE_FEATURES].copy()
        raw_preds = model.predict(X_val)
        val_df["projected_margin"] = calibrator.predict(raw_preds)
    else:
        val_df["projected_margin"] = val_df.apply(_formula_predict_row, axis=1)

    # ------------------------------------------------------------------
    # 3. Drop games where proxy market line is unavailable
    # ------------------------------------------------------------------
    n_before = len(val_df)
    val_df = val_df.dropna(subset=["pregame_spread", "actual_final_margin"])
    n_dropped = n_before - len(val_df)
    if n_dropped:
        print(f"  Dropped {n_dropped} games with missing pregame_spread or target.")

    n_games = len(val_df)
    if n_games == 0:
        msg = "All games dropped after removing missing-line rows — cannot compute cover rates."
        print(f"WARNING: {msg}")
        result = {
            "error": msg,
            "caveat": BACKTEST_CAVEAT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_results(result)
        return result

    # ------------------------------------------------------------------
    # 4. Compute edge: model projection vs market proxy
    # ------------------------------------------------------------------
    val_df["edge"] = val_df["projected_margin"] - val_df["pregame_spread"]

    # ------------------------------------------------------------------
    # 5. Scenario classification
    # ------------------------------------------------------------------
    # h1_margin: positive = team1 (better seed) is currently leading
    val_df["scenario"] = np.where(
        val_df["h1_margin"] > 0,
        "FAV_AHEAD",
        "DOG_WINNING",
    )

    # ------------------------------------------------------------------
    # 6. Grading
    # ------------------------------------------------------------------
    overall_cover = _cover_rate(val_df)

    # By edge threshold
    edge_results: dict[str, object] = {}
    for thresh in EDGE_THRESHOLDS:
        edge_mask = val_df["edge"].abs() >= thresh
        n_edge = int(edge_mask.sum())
        cr = _cover_rate(val_df[edge_mask])
        flag = _sample_flag(n_edge)
        edge_results[thresh] = {"n": n_edge, "cover_rate": cr, "flag": flag}
        print(
            f"  Edge >= {thresh:2d}: n={n_edge}{flag}  cover_rate={cr:.3f}"
            if not np.isnan(cr) else
            f"  Edge >= {thresh:2d}: n={n_edge}{flag}  cover_rate=N/A"
        )

    # By scenario
    scenario_results: dict[str, dict] = {}
    for scenario in ["FAV_AHEAD", "DOG_WINNING"]:
        sub = val_df[val_df["scenario"] == scenario]
        n_s = len(sub)
        cr_s = _cover_rate(sub)
        flag_s = _sample_flag(n_s)
        scenario_results[scenario] = {
            "n": n_s,
            "cover_rate": cr_s if not np.isnan(cr_s) else None,
            "flag": flag_s,
        }
        print(
            f"  Scenario {scenario}: n={n_s}{flag_s}  cover_rate="
            f"{cr_s:.3f}" if not np.isnan(cr_s) else
            f"  Scenario {scenario}: n={n_s}{flag_s}  cover_rate=N/A"
        )

    n_with_edge = int((val_df["edge"].abs() >= min(EDGE_THRESHOLDS)).sum())

    # ------------------------------------------------------------------
    # 7. Build output dict
    # ------------------------------------------------------------------
    result = {
        "halftime": {
            "n_games": n_games,
            "n_with_edge": n_with_edge,
            "cover_rate": float(overall_cover) if not np.isnan(overall_cover) else None,
            "cover_rate_edge_3": (
                float(edge_results[3]["cover_rate"])
                if not np.isnan(edge_results[3]["cover_rate"]) else None
            ),
            "cover_rate_edge_5": (
                float(edge_results[5]["cover_rate"])
                if not np.isnan(edge_results[5]["cover_rate"]) else None
            ),
            "n_edge_3": edge_results[3]["n"],
            "n_edge_5": edge_results[5]["n"],
            "n_edge_7": edge_results[7]["n"],
            "cover_rate_edge_7": (
                float(edge_results[7]["cover_rate"])
                if not np.isnan(edge_results[7]["cover_rate"]) else None
            ),
            "low_sample_flags": {
                f"edge_{t}": edge_results[t]["flag"] for t in EDGE_THRESHOLDS
            },
            "by_scenario": {
                sc: {
                    "n": scenario_results[sc]["n"],
                    "cover_rate": scenario_results[sc]["cover_rate"],
                    "flag": scenario_results[sc]["flag"],
                }
                for sc in ["FAV_AHEAD", "DOG_WINNING"]
            },
        },
        "caveat": BACKTEST_CAVEAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": "xgboost_live_spread" if use_model else "formula_stub",
        "val_year": val_year,
    }

    print(f"\n  Overall cover rate (all games): {overall_cover:.3f}  (n={n_games})")
    print(f"\n*** REMINDER: {BACKTEST_CAVEAT[:120]}…")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    _save_results(result)
    return result


def _save_results(result: dict) -> None:
    out_path = PROCESSED_DIR / "live_backtest_results.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_live_backtest()
