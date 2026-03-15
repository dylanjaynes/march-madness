import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from src.utils.config import (
    SPREAD_MODEL_PARAMS, TOTAL_MODEL_PARAMS, MODELS_DIR,
    COMPETITIVE_MODEL_PARAMS, MISMATCH_MODEL_PARAMS,
    MISMATCH_SEED_DIFF_THRESHOLD, MISMATCH_BARTHAG_THRESHOLD,
)
from src.features.matchup import MATCHUP_FEATURES, build_training_matrix


def train_spread_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """Train XGBoost model for spread prediction."""
    model = xgb.XGBRegressor(**SPREAD_MODEL_PARAMS)

    # FIX: Use TimeSeriesSplit (temporal CV) instead of random KFold.
    # Random KFold lets 2024 data leak into a fold that tests 2010 data,
    # which produces an optimistic CV RMSE that doesn't reflect real forward performance.
    # TimeSeriesSplit always trains on earlier data and tests on later data.
    # X must be sorted by year before this call (done in run_full_training_pipeline).
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="neg_root_mean_squared_error")
    cv_rmse = -scores.mean()
    print(f"  Spread model CV RMSE (temporal): {cv_rmse:.3f} pts (±{scores.std():.3f})")

    # Train on full dataset
    model.fit(X, y)

    # Save
    model_path = MODELS_DIR / "spread_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Spread model saved to {model_path}")
    return model


def train_total_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """Train XGBoost model for total points prediction."""
    model = xgb.XGBRegressor(**TOTAL_MODEL_PARAMS)

    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="neg_root_mean_squared_error")
    cv_rmse = -scores.mean()
    print(f"  Total model CV RMSE (temporal): {cv_rmse:.3f} pts (±{scores.std():.3f})")

    model.fit(X, y)

    model_path = MODELS_DIR / "total_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Total model saved to {model_path}")
    return model


def train_baseline_model(X: pd.DataFrame, y_spread: pd.Series, y_total: pd.Series):
    """
    Ridge regression baseline (simulates KenPom formula).
    Used as benchmark for XGBoost.
    """
    ridge_spread = Ridge(alpha=1.0)
    ridge_spread.fit(X, y_spread)
    spread_pred = ridge_spread.predict(X)
    spread_rmse = np.sqrt(mean_squared_error(y_spread, spread_pred))

    ridge_total = Ridge(alpha=1.0)
    ridge_total.fit(X, y_total)
    total_pred = ridge_total.predict(X)
    total_rmse = np.sqrt(mean_squared_error(y_total, total_pred))

    print(f"  Baseline Ridge - Spread in-sample RMSE: {spread_rmse:.3f}")
    print(f"  Baseline Ridge - Total in-sample RMSE:  {total_rmse:.3f}")

    for name, model, target in [
        ("spread_baseline", ridge_spread, "spread"),
        ("total_baseline", ridge_total, "total"),
    ]:
        model_path = MODELS_DIR / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    return ridge_spread, ridge_total


def _mismatch_mask(seed_diff_series: pd.Series,
                   barthag_diff_series: pd.Series = None) -> pd.Series:
    """Boolean mask: True for games classified as mismatches."""
    mask = seed_diff_series.abs() >= MISMATCH_SEED_DIFF_THRESHOLD
    if barthag_diff_series is not None:
        mask = mask | (barthag_diff_series.abs() >= MISMATCH_BARTHAG_THRESHOLD)
    return mask


def _oof_calibrator(model_proto, X: pd.DataFrame, y: pd.Series) -> IsotonicRegression:
    """
    Fit an IsotonicRegression calibrator on OOF predictions.
    model_proto is cloned fresh for each CV fold so the final model is unaffected.
    """
    cv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.full(len(y), np.nan)

    for train_idx, val_idx in cv.split(X):
        if len(train_idx) < 5:
            continue
        fold_model = copy.deepcopy(model_proto)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = fold_model.predict(X.iloc[val_idx])

    valid = ~np.isnan(oof_preds)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds[valid], y.values[valid])
    return calibrator


def train_hybrid_spread_model(
    X: pd.DataFrame,
    y: pd.Series,
    seed_diff_series: pd.Series,
    barthag_diff_series: pd.Series = None,
):
    """
    Two-stage hybrid spread model.

    Routing rule (applied at train AND predict time):
      |seed_diff| >= MISMATCH_SEED_DIFF_THRESHOLD
      OR |barthag_diff| >= MISMATCH_BARTHAG_THRESHOLD
      → mismatch game → Ridge regression
      otherwise → competitive game → XGBoost

    For each branch:
      1. Train the primary model on the full branch subset.
      2. Fit an IsotonicRegression calibrator on OOF predictions (TimeSeriesSplit, 5 folds).

    Saves four files:
      spread_competitive.pkl, spread_mismatch.pkl,
      cal_competitive.pkl,    cal_mismatch.pkl

    Legacy spread_model.pkl is left untouched (fallback path in predict.py).
    """
    is_mismatch = _mismatch_mask(seed_diff_series, barthag_diff_series)
    is_comp = ~is_mismatch

    X_comp = X[is_comp]
    y_comp = y[is_comp]
    X_mis = X[is_mismatch]
    y_mis = y[is_mismatch]

    print(f"  Competitive games: {is_comp.sum()}  |  Mismatch games: {is_mismatch.sum()}")

    # ── Competitive model: XGBoost ──────────────────────────────────────────
    print("\n  [Competitive] XGBoost")
    comp_proto = xgb.XGBRegressor(**COMPETITIVE_MODEL_PARAMS)

    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(comp_proto, X_comp, y_comp, cv=cv,
                             scoring="neg_root_mean_squared_error")
    print(f"    CV RMSE: {-scores.mean():.3f} pts (±{scores.std():.3f})")

    comp_model = xgb.XGBRegressor(**COMPETITIVE_MODEL_PARAMS)
    comp_model.fit(X_comp, y_comp)

    cal_comp = _oof_calibrator(xgb.XGBRegressor(**COMPETITIVE_MODEL_PARAMS), X_comp, y_comp)

    # Sanity: calibrated OOF min/max on competitive subset
    oof_comp_raw = np.full(len(y_comp), np.nan)
    for train_idx, val_idx in cv.split(X_comp):
        if len(train_idx) < 5:
            continue
        m = copy.deepcopy(comp_proto)
        m.fit(X_comp.iloc[train_idx], y_comp.iloc[train_idx])
        oof_comp_raw[val_idx] = m.predict(X_comp.iloc[val_idx])
    valid_c = ~np.isnan(oof_comp_raw)
    oof_comp_cal = cal_comp.predict(oof_comp_raw[valid_c])
    print(f"    OOF spread range (raw):  [{oof_comp_raw[valid_c].min():.1f}, {oof_comp_raw[valid_c].max():.1f}]")
    print(f"    OOF spread range (cal):  [{oof_comp_cal.min():.1f}, {oof_comp_cal.max():.1f}]")

    # ── Mismatch model: Ridge ───────────────────────────────────────────────
    print("\n  [Mismatch] Ridge(alpha=1.0)")
    mis_proto = Ridge(**MISMATCH_MODEL_PARAMS)

    oof_mis_raw = np.full(len(y_mis), np.nan)
    for train_idx, val_idx in cv.split(X_mis):
        if len(train_idx) < 3:
            continue
        m = Ridge(**MISMATCH_MODEL_PARAMS)
        m.fit(X_mis.iloc[train_idx], y_mis.iloc[train_idx])
        oof_mis_raw[val_idx] = m.predict(X_mis.iloc[val_idx])
    valid_m = ~np.isnan(oof_mis_raw)
    mis_oof_rmse = np.sqrt(mean_squared_error(y_mis.values[valid_m], oof_mis_raw[valid_m]))
    print(f"    OOF RMSE: {mis_oof_rmse:.3f} pts")

    mis_model = Ridge(**MISMATCH_MODEL_PARAMS)
    mis_model.fit(X_mis, y_mis)

    cal_mis = IsotonicRegression(out_of_bounds="clip")
    cal_mis.fit(oof_mis_raw[valid_m], y_mis.values[valid_m])

    oof_mis_cal = cal_mis.predict(oof_mis_raw[valid_m])
    print(f"    OOF spread range (raw):  [{oof_mis_raw[valid_m].min():.1f}, {oof_mis_raw[valid_m].max():.1f}]")
    print(f"    OOF spread range (cal):  [{oof_mis_cal.min():.1f}, {oof_mis_cal.max():.1f}]")

    # ── Save ────────────────────────────────────────────────────────────────
    print()
    artifacts = {
        "spread_competitive": comp_model,
        "spread_mismatch": mis_model,
        "cal_competitive": cal_comp,
        "cal_mismatch": cal_mis,
    }
    for name, obj in artifacts.items():
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {path}")

    return comp_model, mis_model, cal_comp, cal_mis


def sanity_check_hybrid(year: int = 2025):
    """
    Load 2025 test data, run both hybrid models, print min/max predicted spreads.
    Call after train_hybrid_spread_model() has been run.
    """
    from src.utils.db import query_df

    df = query_df("SELECT * FROM mm_training_data WHERE year = ?", params=[year])
    if df.empty:
        print(f"No data for year {year}")
        return

    if "seed_diff" not in df.columns:
        df["seed_diff"] = df["seed_a"].fillna(8) - df["seed_b"].fillna(8)

    barthag_col = "barthag_diff" if "barthag_diff" in df.columns else None

    is_mis = _mismatch_mask(
        df["seed_diff"],
        df[barthag_col] if barthag_col else None,
    )

    X = df[MATCHUP_FEATURES].fillna(0)

    comp_model = load_model("spread_competitive")
    mis_model = load_model("spread_mismatch")
    cal_comp = load_model("cal_competitive")
    cal_mis = load_model("cal_mismatch")

    X_comp = X[~is_mis]
    X_mis = X[is_mis]

    print(f"\n=== Sanity Check — {year} Test Set ===")
    print(f"  Competitive games: {len(X_comp)}  |  Mismatch games: {len(X_mis)}")

    if len(X_comp) > 0:
        raw_c = comp_model.predict(X_comp)
        cal_c = cal_comp.predict(raw_c)
        print(f"  Competitive raw spread:  [{raw_c.min():.1f}, {raw_c.max():.1f}]")
        print(f"  Competitive cal spread:  [{cal_c.min():.1f}, {cal_c.max():.1f}]")

    if len(X_mis) > 0:
        raw_m = mis_model.predict(X_mis)
        cal_m = cal_mis.predict(raw_m)
        print(f"  Mismatch raw spread:     [{raw_m.min():.1f}, {raw_m.max():.1f}]")
        print(f"  Mismatch cal spread:     [{cal_m.min():.1f}, {cal_m.max():.1f}]")
        if cal_m.max() > 22:
            print("  ✓ Mismatch model exceeds ±22 cap — prediction compression resolved")
        else:
            print("  ✗ Mismatch model still capped at ±22 — check calibration")


def load_model(name: str):
    """Load a saved model from disk."""
    model_path = MODELS_DIR / f"{name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_feature_importance(model: xgb.XGBRegressor) -> pd.DataFrame:
    """Return feature importances as a sorted DataFrame."""
    importances = model.feature_importances_
    return pd.DataFrame({
        "feature": MATCHUP_FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def run_full_training_pipeline():
    """
    1. Build training matrix from DB
    2. Sort by year (required for TimeSeriesSplit)
    3. Train spread + total XGBoost models (legacy)
    4. Train hybrid spread models (competitive XGBoost + mismatch Ridge + calibrators)
    5. Train baseline
    6. Print CV RMSE summary
    """
    from src.utils.db import query_df

    print("=== Training Pipeline ===")
    print("Building training matrix...")
    X, y_spread, y_total = build_training_matrix()
    print(f"Training rows: {len(X)}")

    # Drop rows with NaN features
    mask = X.notna().all(axis=1)
    X = X[mask]
    y_spread = y_spread[mask]
    y_total = y_total[mask]
    print(f"After NaN drop: {len(X)} rows")

    # Sort by year so TimeSeriesSplit works correctly.
    X = X.sort_index()
    y_spread = y_spread.sort_index()
    y_total = y_total.sort_index()

    print("\nTraining spread model (legacy XGBoost)...")
    spread_model = train_spread_model(X, y_spread)

    print("\nTraining total model...")
    total_model = train_total_model(X, y_total)

    print("\nTraining baseline models...")
    train_baseline_model(X, y_spread, y_total)

    # ── Hybrid model ────────────────────────────────────────────────────────
    print("\n=== Hybrid Spread Model ===")
    # Pull seed_diff and barthag_diff from the full training table (they're not
    # in MATCHUP_FEATURES but are stored as columns in mm_training_data).
    df_full = query_df("SELECT * FROM mm_training_data")
    df_full = df_full.loc[df_full.index.isin(X.index)] if not df_full.empty else df_full

    # Align to X index after NaN dropping
    common_idx = X.index.intersection(df_full.index)
    df_aligned = df_full.loc[common_idx]
    X_aligned = X.loc[common_idx]
    y_aligned = y_spread.loc[common_idx]

    if "seed_diff" not in df_aligned.columns:
        df_aligned = df_aligned.copy()
        df_aligned["seed_diff"] = (
            df_aligned["seed_a"].fillna(8) - df_aligned["seed_b"].fillna(8)
        )

    seed_diff_s = df_aligned["seed_diff"].reindex(X_aligned.index).fillna(0)
    barthag_diff_s = (
        df_aligned["barthag_diff"].reindex(X_aligned.index)
        if "barthag_diff" in df_aligned.columns else None
    )

    train_hybrid_spread_model(X_aligned, y_aligned, seed_diff_s, barthag_diff_s)

    print("\n=== Feature Importances (Spread) ===")
    fi = get_feature_importance(spread_model)
    print(fi.head(10).to_string(index=False))

    # Sanity check on 2025 test set
    sanity_check_hybrid(year=2025)

    return spread_model, total_model
