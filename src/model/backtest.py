import copy
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.utils.config import (
    TOURNAMENT_YEARS, MODELS_DIR, ROUND_NAMES,
    MISMATCH_SEED_DIFF_THRESHOLD, MISMATCH_BARTHAG_THRESHOLD,
    COMPETITIVE_MODEL_PARAMS, MISMATCH_MODEL_PARAMS,
)
from src.utils.db import query_df
from src.features.matchup import MATCHUP_FEATURES, build_training_matrix, build_matchup_features


def _train_year_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    model_type: str = "competitive",
) -> object:
    """
    Train a single model for one walk-forward fold.

    model_type:
      "competitive" → XGBoost (default, same as legacy behaviour)
      "mismatch"    → Ridge regression
      "auto"        → XGBoost (caller handles routing externally)
    """
    if model_type == "mismatch":
        model = Ridge(**params)
    else:
        model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def _mismatch_mask_series(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series marking mismatch games."""
    mask = df["seed_diff"].abs() >= MISMATCH_SEED_DIFF_THRESHOLD
    if "barthag_diff" in df.columns:
        mask = mask | (df["barthag_diff"].abs() >= MISMATCH_BARTHAG_THRESHOLD)
    return mask


def _fit_oof_calibrator(model_proto, X: pd.DataFrame, y: pd.Series) -> IsotonicRegression:
    """OOF isotonic calibrator using TimeSeriesSplit(n_splits=5)."""
    cv = TimeSeriesSplit(n_splits=5)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in cv.split(X):
        if len(train_idx) < 3:
            continue
        m = copy.deepcopy(model_proto)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = m.predict(X.iloc[val_idx])
    valid = ~np.isnan(oof)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof[valid], y.values[valid])
    return cal


def _ats_result(actual_margin: float, spread: float) -> str:
    """
    ATS result from team_a perspective.
    spread > 0 means team_a is favored by spread pts.
    team_a covers if actual_margin > -spread (i.e. beats the line).
    """
    if actual_margin is None or spread is None:
        return "PUSH"
    cover_margin = actual_margin + spread  # positive = team_a covered
    if cover_margin > 0:
        return "WIN"
    elif cover_margin < 0:
        return "LOSS"
    else:
        return "PUSH"


def _ou_result(actual_total: float, total_line: float) -> str:
    if actual_total is None or total_line is None:
        return "PUSH"
    if actual_total > total_line:
        return "OVER"
    elif actual_total < total_line:
        return "UNDER"
    else:
        return "PUSH"


def calculate_roi(results: list, juice: float = -110) -> float:
    """
    Calculate ROI on a list of bet results.
    results: list of "WIN"/"LOSS"/"PUSH"
    Returns ROI as a percentage.
    """
    if not results:
        return 0.0
    wins = results.count("WIN")
    losses = results.count("LOSS")
    pushes = results.count("PUSH")
    total_bets = wins + losses + pushes

    if total_bets == 0:
        return 0.0

    # Juice: -110 means bet $110 to win $100
    stake = abs(juice)
    payout = 100.0
    total_wagered = total_bets * stake
    total_returned = wins * (stake + payout) + pushes * stake
    roi = (total_returned - total_wagered) / total_wagered * 100
    return roi


def _ats_record(results: list) -> tuple:
    w = results.count("WIN")
    l = results.count("LOSS")
    p = results.count("PUSH")
    return (w, l, p)


def run_backtest(years: list = None) -> dict:
    """
    Walk-forward backtest. For each test year, train on all prior years.
    Returns per-year and aggregate metrics dict.

    Now uses the two-stage hybrid model:
      - Mismatch games (|seed_diff| >= threshold): Ridge + isotonic calibrator
      - Competitive games (otherwise):             XGBoost + isotonic calibrator

    Per-year metrics include separate RMSE and ATS breakdowns for each bucket
    so the impact of the hybrid split is directly visible.

    KEY: market_spread and market_total are kept as NaN when no real line exists.
    ATS and O/U grading ONLY happens on games with real market lines.
    Games without lines still contribute to RMSE metrics.
    """
    from src.utils.config import TOTAL_MODEL_PARAMS

    if years is None:
        years = TOURNAMENT_YEARS

    # Load full training data from DB
    df = query_df("SELECT * FROM mm_training_data")
    if df.empty:
        raise ValueError("No training data. Run build_training_matrix() first.")

    # Ensure seed_diff is available for routing
    if "seed_diff" not in df.columns:
        df["seed_diff"] = df["seed_a"].fillna(8) - df["seed_b"].fillna(8)

    all_predictions = []
    per_year_metrics = {}

    test_years = years[1:]  # Need at least 1 training year
    for i, test_year in enumerate(test_years):
        train_years = years[:years.index(test_year)]
        df_train = df[df["year"].isin(train_years)].copy()
        df_test = df[df["year"] == test_year].copy()

        if df_train.empty or df_test.empty:
            continue

        # ── Feature matrices ─────────────────────────────────────────────
        X_train_full = df_train[MATCHUP_FEATURES].dropna()
        y_spread_train_full = df_train.loc[X_train_full.index, "actual_margin"]
        y_total_train = df_train.loc[X_train_full.index, "actual_total"]

        X_test_full = df_test[MATCHUP_FEATURES].dropna()
        y_spread_test_full = df_test.loc[X_test_full.index, "actual_margin"]
        y_total_test = df_test.loc[X_test_full.index, "actual_total"]

        if len(X_train_full) < 10 or len(X_test_full) < 5:
            continue

        # ── Mismatch routing ─────────────────────────────────────────────
        train_mis_mask = _mismatch_mask_series(df_train.loc[X_train_full.index])
        test_mis_mask = _mismatch_mask_series(df_test.loc[X_test_full.index])

        X_train_comp = X_train_full[~train_mis_mask]
        y_train_comp = y_spread_train_full[~train_mis_mask]
        X_train_mis = X_train_full[train_mis_mask]
        y_train_mis = y_spread_train_full[train_mis_mask]

        X_test_comp = X_test_full[~test_mis_mask]
        X_test_mis = X_test_full[test_mis_mask]

        n_train = len(X_train_full)
        n_comp_test = (~test_mis_mask).sum()
        n_mis_test = test_mis_mask.sum()
        print(
            f"  Backtest {test_year}: train={n_train}  "
            f"test_competitive={n_comp_test}  test_mismatch={n_mis_test}"
        )

        # ── Train competitive model (XGBoost) ────────────────────────────
        comp_model = _train_year_model(X_train_comp, y_train_comp, COMPETITIVE_MODEL_PARAMS, "competitive")
        comp_cal = None
        if len(X_train_comp) >= 10:
            comp_cal = _fit_oof_calibrator(
                xgb.XGBRegressor(**COMPETITIVE_MODEL_PARAMS), X_train_comp, y_train_comp
            )

        # ── Train mismatch model (Ridge) ─────────────────────────────────
        mis_model = _train_year_model(X_train_mis, y_train_mis, MISMATCH_MODEL_PARAMS, "mismatch")
        mis_cal = None
        if len(X_train_mis) >= 10:
            mis_cal = _fit_oof_calibrator(
                Ridge(**MISMATCH_MODEL_PARAMS), X_train_mis, y_train_mis
            )

        # ── Train total model (unchanged: XGBoost) ───────────────────────
        total_model = _train_year_model(X_train_full, y_total_train, TOTAL_MODEL_PARAMS, "competitive")

        # ── Predict spreads via hybrid routing ───────────────────────────
        pred_spread = np.zeros(len(X_test_full))

        if len(X_test_comp) > 0:
            comp_idx = np.where(~test_mis_mask.values)[0]
            raw_comp = comp_model.predict(X_test_comp)
            if comp_cal is not None:
                pred_spread[comp_idx] = comp_cal.predict(raw_comp)
            else:
                pred_spread[comp_idx] = raw_comp

        if len(X_test_mis) > 0:
            mis_idx = np.where(test_mis_mask.values)[0]
            raw_mis = mis_model.predict(X_test_mis)
            if mis_cal is not None:
                pred_spread[mis_idx] = mis_cal.predict(raw_mis)
            else:
                pred_spread[mis_idx] = raw_mis

        pred_total = total_model.predict(X_test_full)

        # ── Metrics ──────────────────────────────────────────────────────
        actual_spreads = y_spread_test_full.values
        actual_totals = y_total_test.values
        market_spreads = df_test.loc[X_test_full.index, "market_spread"].values
        market_totals = df_test.loc[X_test_full.index, "market_total"].values
        round_nums = df_test.loc[X_test_full.index, "round_number"].values
        is_mismatch_arr = test_mis_mask.values  # bool array aligned to X_test_full

        spread_rmse = np.sqrt(mean_squared_error(actual_spreads, pred_spread))
        total_rmse = np.sqrt(mean_squared_error(actual_totals, pred_total))

        # Per-bucket RMSE
        if n_comp_test > 0:
            comp_rmse = np.sqrt(mean_squared_error(
                actual_spreads[~is_mismatch_arr], pred_spread[~is_mismatch_arr]
            ))
        else:
            comp_rmse = float("nan")

        if n_mis_test > 0:
            mis_rmse = np.sqrt(mean_squared_error(
                actual_spreads[is_mismatch_arr], pred_spread[is_mismatch_arr]
            ))
        else:
            mis_rmse = float("nan")

        has_real_spread = ~np.isnan(market_spreads.astype(float))
        if has_real_spread.sum() >= 2:
            market_spread_rmse = np.sqrt(mean_squared_error(
                actual_spreads[has_real_spread], market_spreads[has_real_spread]
            ))
        else:
            market_spread_rmse = float("nan")

        # ── ATS / O/U grading ────────────────────────────────────────────
        ats_all, ats_2, ats_3, ats_5 = [], [], [], []
        ats_comp, ats_mis = [], []   # per-bucket ATS
        ou_all, ou_2, ou_3 = [], [], []

        for j in range(len(X_test_full)):
            model_spread = pred_spread[j]
            model_total = pred_total[j]
            mkt_spread = market_spreads[j]
            mkt_total = market_totals[j]
            actual_margin = actual_spreads[j]
            actual_total = actual_totals[j]
            is_mis_game = bool(is_mismatch_arr[j])

            has_spread_line = not (mkt_spread is None or (isinstance(mkt_spread, float) and np.isnan(mkt_spread)))
            has_total_line = not (mkt_total is None or (isinstance(mkt_total, float) and np.isnan(mkt_total)))

            if has_spread_line:
                mkt_spread = float(mkt_spread)
                spread_edge = model_spread - mkt_spread

                if spread_edge < 0:
                    ats_result = "WIN" if actual_margin < mkt_spread else (
                        "PUSH" if actual_margin == mkt_spread else "LOSS"
                    )
                else:
                    ats_result = "WIN" if actual_margin > mkt_spread else (
                        "PUSH" if actual_margin == mkt_spread else "LOSS"
                    )

                ats_all.append(ats_result)
                abs_edge = abs(spread_edge)
                if abs_edge >= 2:
                    ats_2.append(ats_result)
                if abs_edge >= 3:
                    ats_3.append(ats_result)
                if abs_edge >= 5:
                    ats_5.append(ats_result)

                # Per-bucket ATS
                if is_mis_game:
                    ats_mis.append(ats_result)
                else:
                    ats_comp.append(ats_result)
            else:
                ats_result = "NO_LINE"
                spread_edge = float("nan")
                mkt_spread = float("nan")

            if has_total_line:
                mkt_total = float(mkt_total)
                total_edge = model_total - mkt_total
                ou_result = "WIN" if (
                    (model_total > mkt_total and actual_total > mkt_total) or
                    (model_total < mkt_total and actual_total < mkt_total)
                ) else "LOSS"
                ou_all.append(ou_result)
                if abs(total_edge) >= 2:
                    ou_2.append(ou_result)
                if abs(total_edge) >= 3:
                    ou_3.append(ou_result)
            else:
                total_edge = float("nan")
                mkt_total = float("nan")
                ou_result = "NO_LINE"

            all_predictions.append({
                "year": test_year,
                "game_idx": j,
                "round_number": round_nums[j] if j < len(round_nums) else None,
                "is_mismatch": is_mis_game,
                "model_spread": model_spread,
                "model_total": model_total,
                "market_spread": mkt_spread,
                "market_total": mkt_total,
                "spread_edge": spread_edge,
                "total_edge": total_edge,
                "actual_margin": actual_margin,
                "actual_total": actual_total,
                "ats_result": ats_result,
                "ou_result": ou_result,
            })

        # By-round breakdown
        by_round = {}
        preds_df = pd.DataFrame(all_predictions[-len(X_test_full):])
        for rn in range(1, 7):
            rdf = preds_df[preds_df["round_number"] == rn]
            if not rdf.empty:
                by_round[ROUND_NAMES.get(rn, rn)] = {
                    "n_games": len(rdf),
                    "spread_rmse": np.sqrt(mean_squared_error(
                        rdf["actual_margin"], rdf["model_spread"]
                    )) if len(rdf) > 0 else None,
                    "ats_record": _ats_record(
                        rdf[rdf["ats_result"].isin(["WIN", "LOSS", "PUSH"])]["ats_result"].tolist()
                    ),
                }

        n_games_with_lines = len(ats_all)
        per_year_metrics[test_year] = {
            "spread_rmse": spread_rmse,
            "total_rmse": total_rmse,
            "competitive_rmse": comp_rmse,
            "mismatch_rmse": mis_rmse,
            "vs_market_spread_rmse": market_spread_rmse,
            "ats_record_all": _ats_record(ats_all),
            "ats_record_edge_2": _ats_record(ats_2),
            "ats_record_edge_3": _ats_record(ats_3),
            "ats_record_edge_5": _ats_record(ats_5),
            "ats_roi_edge_3": calculate_roi([r for r in ats_3 if r != "PUSH"]),
            "ats_record_competitive": _ats_record(ats_comp),
            "ats_record_mismatch": _ats_record(ats_mis),
            "ou_record_all": _ats_record(ou_all),
            "ou_record_edge_2": _ats_record(ou_2),
            "ou_record_edge_3": _ats_record(ou_3),
            "ou_roi_edge_3": calculate_roi([r for r in ou_3 if r != "PUSH"]),
            "by_round": by_round,
            "n_games": len(X_test_full),
            "n_games_competitive": n_comp_test,
            "n_games_mismatch": n_mis_test,
            "n_games_with_lines": n_games_with_lines,
            "train_years": train_years,
        }

    # Aggregate metrics
    all_preds_df = pd.DataFrame(all_predictions)
    aggregate = _compute_aggregate_metrics(all_preds_df)
    aggregate["per_year"] = per_year_metrics

    return aggregate


def _compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    spread_rmse = np.sqrt(mean_squared_error(df["actual_margin"], df["model_spread"]))
    total_rmse = np.sqrt(mean_squared_error(df["actual_total"], df["model_total"]))

    # Per-bucket aggregate RMSE
    comp_df = df[~df["is_mismatch"]] if "is_mismatch" in df.columns else df
    mis_df = df[df["is_mismatch"]] if "is_mismatch" in df.columns else pd.DataFrame()
    comp_rmse = float(np.sqrt(mean_squared_error(comp_df["actual_margin"], comp_df["model_spread"]))) if len(comp_df) > 0 else float("nan")
    mis_rmse = float(np.sqrt(mean_squared_error(mis_df["actual_margin"], mis_df["model_spread"]))) if len(mis_df) > 0 else float("nan")

    # Market RMSE only over games with real lines (non-NaN market_spread)
    real_lines = df[df["market_spread"].notna() & (df["market_spread"] != 0)]
    market_rmse = np.sqrt(mean_squared_error(
        real_lines["actual_margin"], real_lines["market_spread"]
    )) if len(real_lines) >= 2 else float("nan")

    # ATS: only games with real market lines
    ats_df = df[df["ats_result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    ats_3_df = ats_df[ats_df["spread_edge"].abs() >= 3] if not ats_df.empty else ats_df
    ats_5_df = ats_df[ats_df["spread_edge"].abs() >= 5] if not ats_df.empty else ats_df

    ats_all = ats_df["ats_result"].tolist()
    ats_3 = ats_3_df["ats_result"].tolist()
    ats_5 = ats_5_df["ats_result"].tolist()

    # Per-bucket ATS aggregate
    ats_comp_df = ats_df[~ats_df["is_mismatch"]] if "is_mismatch" in ats_df.columns else ats_df
    ats_mis_df = ats_df[ats_df["is_mismatch"]] if "is_mismatch" in ats_df.columns else pd.DataFrame()

    # O/U: only games with real total lines
    ou_all_df = df[df["ou_result"].isin(["WIN", "LOSS"])]
    ou_3_df = ou_all_df[ou_all_df["total_edge"].abs() >= 3] if not ou_all_df.empty else ou_all_df
    ou_all = ou_all_df["ou_result"].tolist()
    ou_3 = ou_3_df["ou_result"].tolist()

    return {
        "spread_rmse": spread_rmse,
        "total_rmse": total_rmse,
        "competitive_rmse": comp_rmse,
        "mismatch_rmse": mis_rmse,
        "vs_market_spread_rmse": market_rmse,
        "ats_record_all": _ats_record(ats_all),
        "ats_record_edge_3": _ats_record(ats_3),
        "ats_record_edge_5": _ats_record(ats_5),
        "ats_roi_edge_3": calculate_roi([r for r in ats_3 if r != "PUSH"]),
        "ats_record_competitive": _ats_record(ats_comp_df["ats_result"].tolist()),
        "ats_record_mismatch": _ats_record(ats_mis_df["ats_result"].tolist() if not ats_mis_df.empty else []),
        "ou_record_all": _ats_record(ou_all),
        "ou_record_edge_3": _ats_record(ou_3),
        "ou_roi_edge_3": calculate_roi([r for r in ou_3 if r != "PUSH"]),
        "n_games_total": len(df),
        "predictions_df": df,
    }


def generate_backtest_report(results: dict) -> pd.DataFrame:
    """
    Format backtest results into a display-ready DataFrame.
    Includes separate RMSE and ATS rows for competitive vs mismatch games.
    ATS columns reflect TRUE ATS vs real market lines (games without lines excluded).
    """
    rows = []
    per_year = results.get("per_year", {})

    def _pct(w, l):
        return f"{w/(w+l)*100:.1f}%" if (w + l) > 0 else "—"

    def _rmse_str(v):
        return f"{v:.2f}" if not np.isnan(v) else "—"

    for year, metrics in per_year.items():
        ats_all = metrics.get("ats_record_all", (0, 0, 0))
        ats_3 = metrics.get("ats_record_edge_3", (0, 0, 0))
        ou_all = metrics.get("ou_record_all", (0, 0, 0))
        n_lines = metrics.get("n_games_with_lines", sum(ats_all))
        mkt_rmse = metrics.get("vs_market_spread_rmse", float("nan"))
        comp_rmse = metrics.get("competitive_rmse", float("nan"))
        mis_rmse = metrics.get("mismatch_rmse", float("nan"))
        ats_comp = metrics.get("ats_record_competitive", (0, 0, 0))
        ats_mis = metrics.get("ats_record_mismatch", (0, 0, 0))
        n_comp = metrics.get("n_games_competitive", 0)
        n_mis = metrics.get("n_games_mismatch", 0)

        w, l, p = ats_all
        w3, l3, p3 = ats_3
        wc, lc, pc = ats_comp
        wm, lm, pm = ats_mis

        rows.append({
            "Year": str(year),
            "Games": metrics.get("n_games", 0),
            "Comp/Mis": f"{n_comp}/{n_mis}",
            "w/ Lines": n_lines,
            "Spread RMSE": round(metrics.get("spread_rmse", 0), 2),
            "Comp RMSE": _rmse_str(comp_rmse),
            "Mis RMSE": _rmse_str(mis_rmse),
            "Mkt RMSE": _rmse_str(mkt_rmse),
            "ATS All": f"{w}-{l}-{p} ({_pct(w,l)})" if n_lines > 0 else "no lines",
            "ATS Edge≥3": f"{w3}-{l3}-{p3} ({_pct(w3,l3)})" if (w3+l3) > 0 else "—",
            "ATS ROI≥3": f"{metrics.get('ats_roi_edge_3', 0):.1f}%" if (w3+l3) > 0 else "—",
            "ATS Comp": f"{wc}-{lc}-{pc} ({_pct(wc,lc)})" if (wc+lc) > 0 else "—",
            "ATS Mis": f"{wm}-{lm}-{pm} ({_pct(wm,lm)})" if (wm+lm) > 0 else "—",
            "O/U All": f"{ou_all[0]}-{ou_all[1]}-{ou_all[2]}" if sum(ou_all) > 0 else "no lines",
        })

    # Aggregate row
    ats_all_agg = results.get("ats_record_all", (0, 0, 0))
    ats_3_agg = results.get("ats_record_edge_3", (0, 0, 0))
    ats_comp_agg = results.get("ats_record_competitive", (0, 0, 0))
    ats_mis_agg = results.get("ats_record_mismatch", (0, 0, 0))
    ou_all_agg = results.get("ou_record_all", (0, 0, 0))

    w, l, p = ats_all_agg
    w3, l3, p3 = ats_3_agg
    wc, lc, pc = ats_comp_agg
    wm, lm, pm = ats_mis_agg
    agg_mkt_rmse = results.get("vs_market_spread_rmse", float("nan"))

    rows.append({
        "Year": "TOTAL",
        "Games": results.get("n_games_total", 0),
        "Comp/Mis": "—",
        "w/ Lines": w + l + p,
        "Spread RMSE": round(results.get("spread_rmse", 0), 2),
        "Comp RMSE": _rmse_str(results.get("competitive_rmse", float("nan"))),
        "Mis RMSE": _rmse_str(results.get("mismatch_rmse", float("nan"))),
        "Mkt RMSE": _rmse_str(agg_mkt_rmse),
        "ATS All": f"{w}-{l}-{p} ({_pct(w,l)})",
        "ATS Edge≥3": f"{w3}-{l3}-{p3} ({_pct(w3,l3)})" if (w3+l3) > 0 else "—",
        "ATS ROI≥3": f"{results.get('ats_roi_edge_3', 0):.1f}%",
        "ATS Comp": f"{wc}-{lc}-{pc} ({_pct(wc,lc)})" if (wc+lc) > 0 else "—",
        "ATS Mis": f"{wm}-{lm}-{pm} ({_pct(wm,lm)})" if (wm+lm) > 0 else "—",
        "O/U All": f"{ou_all_agg[0]}-{ou_all_agg[1]}-{ou_all_agg[2]}",
    })

    return pd.DataFrame(rows)
