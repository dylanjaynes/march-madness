import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from src.utils.config import TOURNAMENT_YEARS, MODELS_DIR, ROUND_NAMES
from src.utils.db import query_df
from src.features.matchup import MATCHUP_FEATURES, build_training_matrix, build_matchup_features


def _train_year_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


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
    """
    from src.utils.config import SPREAD_MODEL_PARAMS, TOTAL_MODEL_PARAMS

    if years is None:
        years = TOURNAMENT_YEARS

    # Load full training data from DB
    df = query_df("SELECT * FROM mm_training_data")
    if df.empty:
        raise ValueError("No training data. Run build_training_matrix() first.")

    # Fill NaN market lines with naive KenPom-derived spread as proxy
    # (We'll use model-vs-actual comparison as primary metric)
    df["market_spread"] = df["market_spread"].fillna(0.0)
    df["market_total"] = df["market_total"].fillna(df["actual_total"].mean())

    all_predictions = []
    per_year_metrics = {}

    # Compute seed_diff from stored seed columns (not stored as feature)
    if "seed_diff" not in df.columns:
        df["seed_diff"] = df["seed_a"].fillna(8) - df["seed_b"].fillna(8)

    test_years = years[1:]  # Need at least 1 training year
    for i, test_year in enumerate(test_years):
        train_years = years[:years.index(test_year)]
        df_train = df[df["year"].isin(train_years)].copy()
        df_test = df[df["year"] == test_year].copy()

        if df_train.empty or df_test.empty:
            continue

        X_train = df_train[MATCHUP_FEATURES].dropna()
        y_spread_train = df_train.loc[X_train.index, "actual_margin"]
        y_total_train = df_train.loc[X_train.index, "actual_total"]

        X_test = df_test[MATCHUP_FEATURES].dropna()
        y_spread_test = df_test.loc[X_test.index, "actual_margin"]
        y_total_test = df_test.loc[X_test.index, "actual_total"]

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        print(f"  Backtest {test_year}: train={len(X_train)} games, test={len(X_test)} games")

        # Train models
        spread_model = _train_year_model(X_train, y_spread_train, SPREAD_MODEL_PARAMS)
        total_model = _train_year_model(X_train, y_total_train, TOTAL_MODEL_PARAMS)

        # Predict
        pred_spread = spread_model.predict(X_test)
        pred_total = total_model.predict(X_test)

        # Compute metrics
        actual_spreads = y_spread_test.values
        actual_totals = y_total_test.values
        market_spreads = df_test.loc[X_test.index, "market_spread"].values
        market_totals = df_test.loc[X_test.index, "market_total"].values
        round_nums = df_test.loc[X_test.index, "round_number"].values

        spread_rmse = np.sqrt(mean_squared_error(actual_spreads, pred_spread))
        total_rmse = np.sqrt(mean_squared_error(actual_totals, pred_total))
        market_spread_rmse = np.sqrt(mean_squared_error(actual_spreads, market_spreads))

        # ATS analysis (model vs actual, no market lines)
        # We use model spread as our "pick" — positive model spread means team_a favored
        ats_all, ats_2, ats_3, ats_5 = [], [], [], []
        ou_all, ou_2, ou_3 = [], [], []

        for j in range(len(X_test)):
            model_spread = pred_spread[j]
            model_total = pred_total[j]
            mkt_spread = market_spreads[j]
            mkt_total = market_totals[j]
            actual_margin = actual_spreads[j]
            actual_total = actual_totals[j]

            # ATS: compare model projection to actual (if no market lines, use 0)
            # Positive spread_edge = model thinks team_a covers
            spread_edge = model_spread - mkt_spread
            total_edge = model_total - mkt_total

            # Use actual result vs model as ATS proxy
            # If model says team_a by X and actual > 0 = team_a wins = model was right directionally
            ats_result = "WIN" if (model_spread > 0 and actual_margin > 0) or \
                                  (model_spread < 0 and actual_margin < 0) else "LOSS"
            ats_all.append(ats_result)

            if abs(spread_edge) >= 2:
                ats_2.append(ats_result)
            if abs(spread_edge) >= 3:
                ats_3.append(ats_result)
            if abs(spread_edge) >= 5:
                ats_5.append(ats_result)

            ou_result = "WIN" if (model_total > mkt_total and actual_total > mkt_total) or \
                                 (model_total < mkt_total and actual_total < mkt_total) else "LOSS"
            ou_all.append(ou_result)
            if abs(total_edge) >= 2:
                ou_2.append(ou_result)
            if abs(total_edge) >= 3:
                ou_3.append(ou_result)

            all_predictions.append({
                "year": test_year,
                "game_idx": j,
                "round_number": round_nums[j] if j < len(round_nums) else None,
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
        preds_df = pd.DataFrame(all_predictions[-len(X_test):])
        for rn in range(1, 7):
            rdf = preds_df[preds_df["round_number"] == rn]
            if not rdf.empty:
                by_round[ROUND_NAMES.get(rn, rn)] = {
                    "n_games": len(rdf),
                    "spread_rmse": np.sqrt(mean_squared_error(
                        rdf["actual_margin"], rdf["model_spread"]
                    )) if len(rdf) > 0 else None,
                    "ats_record": _ats_record(rdf["ats_result"].tolist()),
                }

        per_year_metrics[test_year] = {
            "spread_rmse": spread_rmse,
            "total_rmse": total_rmse,
            "vs_market_spread_rmse": market_spread_rmse,
            "ats_record_all": _ats_record(ats_all),
            "ats_record_edge_2": _ats_record(ats_2),
            "ats_record_edge_3": _ats_record(ats_3),
            "ats_record_edge_5": _ats_record(ats_5),
            "ats_roi_edge_3": calculate_roi([r for r in ats_3 if r != "PUSH"]),
            "ou_record_all": _ats_record(ou_all),
            "ou_record_edge_2": _ats_record(ou_2),
            "ou_record_edge_3": _ats_record(ou_3),
            "ou_roi_edge_3": calculate_roi([r for r in ou_3 if r != "PUSH"]),
            "by_round": by_round,
            "n_games": len(X_test),
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
    market_rmse = np.sqrt(mean_squared_error(df["actual_margin"], df["market_spread"]))

    ats_all = df["ats_result"].tolist()
    ats_3 = df[df["spread_edge"].abs() >= 3]["ats_result"].tolist()
    ats_5 = df[df["spread_edge"].abs() >= 5]["ats_result"].tolist()

    ou_all = df["ou_result"].tolist()
    ou_3 = df[df["total_edge"].abs() >= 3]["ou_result"].tolist()

    return {
        "spread_rmse": spread_rmse,
        "total_rmse": total_rmse,
        "vs_market_spread_rmse": market_rmse,
        "ats_record_all": _ats_record(ats_all),
        "ats_record_edge_3": _ats_record(ats_3),
        "ats_record_edge_5": _ats_record(ats_5),
        "ats_roi_edge_3": calculate_roi([r for r in ats_3 if r != "PUSH"]),
        "ou_record_all": _ats_record(ou_all),
        "ou_record_edge_3": _ats_record(ou_3),
        "ou_roi_edge_3": calculate_roi([r for r in ou_3 if r != "PUSH"]),
        "n_games_total": len(df),
        "predictions_df": df,
    }


def generate_backtest_report(results: dict) -> pd.DataFrame:
    """Format backtest results into a display-ready DataFrame."""
    rows = []
    per_year = results.get("per_year", {})

    for year, metrics in per_year.items():
        ats_all = metrics.get("ats_record_all", (0, 0, 0))
        ats_3 = metrics.get("ats_record_edge_3", (0, 0, 0))
        ou_all = metrics.get("ou_record_all", (0, 0, 0))

        rows.append({
            "Year": str(year),
            "Games": metrics.get("n_games", 0),
            "Spread RMSE": round(metrics.get("spread_rmse", 0), 2),
            "Total RMSE": round(metrics.get("total_rmse", 0), 2),
            "Market RMSE": round(metrics.get("vs_market_spread_rmse", 0), 2),
            "ATS (All)": f"{ats_all[0]}-{ats_all[1]}-{ats_all[2]}",
            "ATS (Edge≥3)": f"{ats_3[0]}-{ats_3[1]}-{ats_3[2]}",
            "ATS ROI (Edge≥3)": f"{metrics.get('ats_roi_edge_3', 0):.1f}%",
            "O/U (All)": f"{ou_all[0]}-{ou_all[1]}-{ou_all[2]}",
            "Train Years": str(metrics.get("train_years", [])),
        })

    # Add aggregate row
    ats_all_agg = results.get("ats_record_all", (0, 0, 0))
    ats_3_agg = results.get("ats_record_edge_3", (0, 0, 0))
    ou_all_agg = results.get("ou_record_all", (0, 0, 0))
    rows.append({
        "Year": "TOTAL",
        "Games": results.get("n_games_total", 0),
        "Spread RMSE": round(results.get("spread_rmse", 0), 2),
        "Total RMSE": round(results.get("total_rmse", 0), 2),
        "Market RMSE": round(results.get("vs_market_spread_rmse", 0), 2),
        "ATS (All)": f"{ats_all_agg[0]}-{ats_all_agg[1]}-{ats_all_agg[2]}",
        "ATS (Edge≥3)": f"{ats_3_agg[0]}-{ats_3_agg[1]}-{ats_3_agg[2]}",
        "ATS ROI (Edge≥3)": f"{results.get('ats_roi_edge_3', 0):.1f}%",
        "O/U (All)": f"{ou_all_agg[0]}-{ou_all_agg[1]}-{ou_all_agg[2]}",
        "Train Years": "Walk-Forward",
    })

    return pd.DataFrame(rows)
