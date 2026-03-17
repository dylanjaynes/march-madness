"""
Walk-forward NIT backtest.
Same hybrid Ridge+XGBoost methodology as backtest.py.
Trains on prior NCAA + NIT data combined, tests on NIT games.
NIT teams have no seeds — barthag_diff is used for mismatch routing instead.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

from src.utils.config import (
    TOTAL_MODEL_PARAMS, SELECTION_SUNDAY, COMPETITIVE_SPREAD_THRESHOLD,
    MISMATCH_BARTHAG_THRESHOLD,
)
from src.utils.db import query_df
from src.features.matchup import MATCHUP_FEATURES, build_matchup_features
from src.features.team_ratings import build_team_feature_vector
from src.model.backtest import (
    _ats_record, calculate_roi, _compute_aggregate_metrics,
    _train_competitive_model, _train_mismatch_model,
)


def _mismatch_mask_nit(df: pd.DataFrame) -> np.ndarray:
    """
    NIT has no seeds — use only barthag_diff for mismatch routing.
    Games where |barthag_diff| >= MISMATCH_BARTHAG_THRESHOLD are mismatches.
    """
    if "barthag_diff" in df.columns:
        return (df["barthag_diff"].abs() >= MISMATCH_BARTHAG_THRESHOLD).values
    return np.zeros(len(df), dtype=bool)


def build_nit_training_matrix() -> pd.DataFrame:
    """Build feature rows for all NIT games. No seeds — use barthag for team_a assignment."""
    results = query_df("SELECT * FROM nit_results ORDER BY year, game_date")
    lines = query_df(
        "SELECT year, game_date, team1, team2, spread_line, total_line, spread_favorite "
        "FROM nit_lines"
    )

    if results.empty:
        raise ValueError("No NIT results — run ingest_nit_results() first")

    df = results.merge(lines, on=["year", "game_date", "team1", "team2"], how="left")
    rows = []

    for _, game in df.iterrows():
        year = int(game["year"])
        round_num = int(game.get("round_number") or 1)
        t1, t2 = str(game["team1"]), str(game["team2"])
        s1, s2 = float(game["score1"]), float(game["score2"])

        as_of = SELECTION_SUNDAY.get(year)
        r1 = build_team_feature_vector(t1, year, as_of)
        r2 = build_team_feature_vector(t2, year, as_of)

        b1 = float(r1.get("barthag", 0.5) or 0.5) if r1 else 0.5
        b2 = float(r2.get("barthag", 0.5) or 0.5) if r2 else 0.5

        # team_a = higher barthag (better team)
        if b1 >= b2:
            team_a, team_b, actual_margin = t1, t2, s1 - s2
        else:
            team_a, team_b, actual_margin = t2, t1, s2 - s1

        feats = build_matchup_features(
            team_a, team_b, year, round_num,
            game_date=str(game.get("game_date") or f"{year}-03-20"),
            as_of_date=as_of,
        )
        if feats.isna().all():
            continue

        # Market spread: convert to model convention (positive = team_a favored)
        mkt_spread = None
        if pd.notna(game.get("spread_line")):
            raw = float(game["spread_line"])
            fav = str(game.get("spread_favorite") or "").strip().lower()
            mkt_spread = raw if fav == team_a.strip().lower() else -raw

        row = feats.to_dict()
        row.update({
            "year": year,
            "game_date": str(game.get("game_date") or ""),
            "team_a": team_a,
            "team_b": team_b,
            "seed_a": 8,   # NIT default (mid-seed for model convention)
            "seed_b": 8,
            "actual_margin": actual_margin,
            "actual_total": s1 + s2,
            "market_spread": mkt_spread,
            "market_total": None,
            "round_number": round_num,
            "barthag_diff": abs(b1 - b2),
        })
        rows.append(row)

    if not rows:
        raise ValueError("No NIT feature rows built")

    out = pd.DataFrame(rows)
    n_lines = out["market_spread"].notna().sum()
    print(f"  NIT matrix: {len(out)} games, {n_lines} with lines")
    return out


def run_nit_backtest(nit_df: pd.DataFrame = None) -> dict:
    """Walk-forward NIT backtest. Trains on prior NCAA + NIT, tests on NIT per year."""
    if nit_df is None:
        nit_df = build_nit_training_matrix()

    ncaa_df = query_df("SELECT * FROM mm_training_data")
    if ncaa_df.empty:
        raise ValueError("No NCAA training data in mm_training_data")

    # Add seed_diff = 0 for NIT (no seeds)
    nit_df = nit_df.copy()
    if "seed_diff" not in nit_df.columns:
        nit_df["seed_diff"] = 0

    # Ensure NCAA df has barthag_diff if missing
    if "barthag_diff" not in ncaa_df.columns:
        ncaa_df["barthag_diff"] = 0.0

    years_with_nit = sorted(nit_df["year"].unique())
    all_predictions = []
    per_year_metrics = {}

    for test_year in years_with_nit[1:]:
        train_years = [y for y in years_with_nit if y < test_year]

        ncaa_train = ncaa_df[ncaa_df["year"].isin(train_years)].copy()
        nit_train  = nit_df[nit_df["year"].isin(train_years)].copy()
        df_test    = nit_df[nit_df["year"] == test_year].copy()

        # Combine train sets with common features
        for df_part in [ncaa_train, nit_train]:
            for col in MATCHUP_FEATURES:
                if col not in df_part.columns:
                    df_part[col] = 0.0

        df_train = pd.concat([ncaa_train, nit_train], ignore_index=True)

        X_train = df_train[MATCHUP_FEATURES].fillna(0)
        y_s_train = df_train["actual_margin"].fillna(0)
        y_t_train = df_train["actual_total"].fillna(130)

        for col in MATCHUP_FEATURES:
            if col not in df_test.columns:
                df_test[col] = 0.0
        X_test = df_test[MATCHUP_FEATURES].fillna(0)
        y_s_test = df_test["actual_margin"].values
        y_t_test = df_test["actual_total"].values

        if len(X_train) < 10 or len(X_test) < 3:
            print(f"  NIT {test_year}: skip (train={len(X_train)}, test={len(X_test)})")
            continue

        print(f"  NIT {test_year}: train={len(X_train)}, test={len(X_test)}")

        # Hybrid spread model (mismatch routed by barthag_diff only for NIT)
        mis_tr = _mismatch_mask_nit(df_train)
        mis_te = _mismatch_mask_nit(df_test.reset_index(drop=True))

        comp_model = _train_competitive_model(
            X_train.iloc[~mis_tr], y_s_train.iloc[~mis_tr]
        )
        mis_model = _train_mismatch_model(
            X_train.iloc[mis_tr], y_s_train.iloc[mis_tr]
        )

        pred_spread = np.zeros(len(X_test))
        for idx_arr, model in [
            (np.where(~mis_te)[0], comp_model),
            (np.where(mis_te)[0],  mis_model),
        ]:
            if len(idx_arr) > 0:
                pred_spread[idx_arr] = model.predict(X_test.iloc[idx_arr])

        # Total model
        total_model = xgb.XGBRegressor(**TOTAL_MODEL_PARAMS)
        total_model.fit(X_train, y_t_train)
        pred_total = total_model.predict(X_test)
        try:
            cv = TimeSeriesSplit(n_splits=min(5, len(X_train) // 20))
            oof = np.full(len(y_t_train), np.nan)
            for tr, va in cv.split(X_train):
                if len(tr) < 5:
                    continue
                m = xgb.XGBRegressor(**TOTAL_MODEL_PARAMS)
                m.fit(X_train.iloc[tr], y_t_train.iloc[tr])
                oof[va] = m.predict(X_train.iloc[va])
            v = ~np.isnan(oof)
            if v.sum() >= 10:
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(oof[v], y_t_train.values[v])
                pred_total = cal.predict(pred_total)
        except Exception:
            pass

        market_spreads = df_test["market_spread"].values
        ats_all, ats_3, ats_5 = [], [], []

        for j in range(len(X_test)):
            mkt = market_spreads[j]
            has_line = (
                mkt is not None
                and not (isinstance(mkt, float) and np.isnan(mkt))
            )

            if has_line:
                mkt = float(mkt)
                edge = pred_spread[j] - mkt
                actual = y_s_test[j]
                result = (
                    "WIN" if (edge >= 0 and actual > mkt) or (edge < 0 and actual < mkt)
                    else "PUSH" if actual == mkt else "LOSS"
                )
                ats_all.append(result)
                if abs(edge) >= 3:
                    ats_3.append(result)
                if abs(edge) >= 5:
                    ats_5.append(result)
                is_comp = abs(mkt) <= COMPETITIVE_SPREAD_THRESHOLD
            else:
                result = "NO_LINE"
                edge = float("nan")
                mkt = float("nan")
                is_comp = True

            all_predictions.append({
                "year": test_year,
                "team_a": df_test.iloc[j]["team_a"],
                "team_b": df_test.iloc[j]["team_b"],
                "round_number": int(df_test.iloc[j]["round_number"]),
                "is_mismatch": bool(mis_te[j]),
                "model_spread": float(pred_spread[j]),
                "model_total": float(pred_total[j]),
                "market_spread": float(mkt) if not (isinstance(mkt, float) and np.isnan(mkt)) else None,
                "spread_edge": float(edge) if not (isinstance(edge, float) and np.isnan(edge)) else None,
                "actual_margin": float(y_s_test[j]),
                "actual_total": float(y_t_test[j]),
                "ats_result": result,
                "ou_result": "NO_LINE",
                "is_competitive": is_comp,
            })

        rmse = float(np.sqrt(mean_squared_error(y_s_test, pred_spread)))
        has = np.array([
            m is not None and not (isinstance(m, float) and np.isnan(m))
            for m in market_spreads
        ])
        mkt_rmse = (
            float(np.sqrt(mean_squared_error(y_s_test[has], market_spreads[has].astype(float))))
            if has.sum() >= 2 else float("nan")
        )

        per_year_metrics[test_year] = {
            "spread_rmse": rmse,
            "vs_market_spread_rmse": mkt_rmse,
            "ats_record_all": _ats_record(ats_all),
            "ats_record_edge_3": _ats_record(ats_3),
            "ats_record_edge_5": _ats_record(ats_5),
            "ats_roi_edge_3": calculate_roi([r for r in ats_3 if r != "PUSH"]),
            "n_games": len(X_test),
            "n_games_with_lines": len(ats_all),
        }

    preds_df = pd.DataFrame(all_predictions)
    agg = _compute_aggregate_metrics(preds_df) if not preds_df.empty else {}
    agg["per_year"] = per_year_metrics
    agg["predictions_df"] = preds_df
    return agg


def print_nit_report(results: dict):
    print("\n" + "=" * 65)
    print("NIT WALK-FORWARD BACKTEST REPORT")
    print("=" * 65)
    for year, m in sorted(results.get("per_year", {}).items()):
        w, l, p = m.get("ats_record_all", (0, 0, 0))
        w3, l3, _ = m.get("ats_record_edge_3", (0, 0, 0))
        pct = f"{w/(w+l)*100:.1f}%" if w + l > 0 else "—"
        p3  = f"{w3/(w3+l3)*100:.1f}%" if w3 + l3 > 0 else "—"
        mr = m.get("vs_market_spread_rmse", float("nan"))
        mr_s = f"{mr:.2f}" if not (isinstance(mr, float) and np.isnan(mr)) else "—"
        print(
            f"  {year}: {m['n_games_with_lines']}/{m['n_games']} lined "
            f"| RMSE {m['spread_rmse']:.2f} mkt {mr_s} "
            f"| ATS {w}-{l} ({pct}) "
            f"| Edge≥3 {w3}-{l3} ({p3}) ROI {m.get('ats_roi_edge_3', 0):.1f}%"
        )

    print()
    aw, al, ap = results.get("ats_record_all", (0, 0, 0))
    aw3, al3, _ = results.get("ats_record_edge_3", (0, 0, 0))
    pt  = f"{aw/(aw+al)*100:.1f}%" if aw + al > 0 else "—"
    p3t = f"{aw3/(aw3+al3)*100:.1f}%" if aw3 + al3 > 0 else "—"
    print(
        f"  TOTAL | ATS {aw}-{al} ({pt}) "
        f"| Edge≥3 {aw3}-{al3} ({p3t}) "
        f"| ROI {results.get('ats_roi_edge_3', 0):.1f}%"
    )

    df = results.get("predictions_df", pd.DataFrame())
    if not df.empty and "is_competitive" in df.columns:
        c = df[df["is_competitive"] & df["ats_result"].isin(["WIN", "LOSS"])]
        c3 = c[c["spread_edge"].abs() >= 3] if not c.empty else c
        cw = (c3["ats_result"] == "WIN").sum()
        cl = (c3["ats_result"] == "LOSS").sum()
        pct_comp = f"{cw/(cw+cl)*100:.1f}%" if cw + cl > 0 else "—"
        print(
            f"\n  Competitive only (|spread|≤{COMPETITIVE_SPREAD_THRESHOLD:.0f}): "
            f"Edge≥3 {cw}-{cl} ({pct_comp})"
        )
