"""
live_train.py
-------------
Trains the live in-game spread model.

Architecture mirrors src/model/train.py:
  - XGBoost regressor with SPREAD_MODEL_PARAMS hyperparameters
  - TimeSeriesSplit cross-validation (5 folds)
  - IsotonicRegression calibrator fitted on OOF predictions (_oof_calibrator)

Saved artefacts:
  models/live_spread_model.pkl
  models/live_spread_calibrator.pkl
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb

from src.utils.config import MODELS_DIR, SPREAD_MODEL_PARAMS
from src.model.live_train_data import LIVE_FEATURES, build_live_training_data


# ---------------------------------------------------------------------------
# Formula-based stub
# ---------------------------------------------------------------------------

def formula_projected_margin(
    current_margin: float,
    pregame_spread: float,
    time_elapsed: float,
    time_remaining: float,
    efg_pct_diff: float = 0.0,
    orb_margin: float = 0.0,
    to_margin: float = 0.0,
) -> float:
    """
    Time-weighted blend of current score and pre-game model, plus efficiency adjustments.
    Coefficients from sports analytics literature — replaced by trained model when available.
    NOTE: w_model multiplier on adjustments shrinks correction terms to zero as game ends,
    preventing absurd late-game projections.
    """
    w_score = time_elapsed / 40.0
    w_model = time_remaining / 40.0
    base = current_margin * w_score + pregame_spread * w_model
    efg_adj = efg_pct_diff * 0.15 * w_model
    orb_adj = orb_margin   * 0.25 * w_model
    to_adj  = to_margin    * 0.40 * w_model
    return base + efg_adj + orb_adj + to_adj


# ---------------------------------------------------------------------------
# OOF calibrator (mirrors train.py _oof_calibrator)
# ---------------------------------------------------------------------------

def _oof_calibrator(
    model_proto: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> IsotonicRegression:
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


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_live_model() -> None:
    """
    Trains XGBoost live spread model + isotonic calibrator.
    Train years: all except 2025.  Val year: 2025.
    Saves:
        models/live_spread_model.pkl
        models/live_spread_calibrator.pkl
    Logs: feature importances, MAE vs pre-game baseline.
    """
    print("=== Live Spread Model Training ===")

    # ------------------------------------------------------------------
    # 1. Build training data
    # ------------------------------------------------------------------
    train_df, val_df = build_live_training_data(val_year=2025)

    if train_df.empty:
        raise ValueError(
            "Training set is empty — ensure halftime_scores table is populated "
            "for years other than 2025."
        )

    # ------------------------------------------------------------------
    # 2. Prepare feature matrices
    # ------------------------------------------------------------------
    # XGBoost handles NaN natively — do NOT fill missing box-score features.
    X_train = train_df[LIVE_FEATURES].copy()
    y_train = train_df["actual_final_margin"].copy()

    # Sort by year index for temporal CV (already sorted by build_live_training_data,
    # but reset ensures a clean 0-based integer index for iloc in TimeSeriesSplit)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    print(f"  Training rows : {len(X_train)}")
    print(f"  Features      : {LIVE_FEATURES}")

    # ------------------------------------------------------------------
    # 3. Cross-validation (temporal)
    # ------------------------------------------------------------------
    cv = TimeSeriesSplit(n_splits=5)
    proto = xgb.XGBRegressor(**SPREAD_MODEL_PARAMS)
    scores = cross_val_score(
        proto, X_train, y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    cv_rmse = -scores.mean()
    print(f"  Live model CV RMSE (temporal): {cv_rmse:.3f} pts (±{scores.std():.3f})")

    # ------------------------------------------------------------------
    # 4. Train on full training set
    # ------------------------------------------------------------------
    model = xgb.XGBRegressor(**SPREAD_MODEL_PARAMS)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Fit OOF calibrator
    # ------------------------------------------------------------------
    print("  Fitting OOF isotonic calibrator …")
    calibrator = _oof_calibrator(
        xgb.XGBRegressor(**SPREAD_MODEL_PARAMS),
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 6. Feature importances
    # ------------------------------------------------------------------
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": LIVE_FEATURES, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\n  Feature importances (sorted):")
    print(fi_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Validation metrics vs pre-game baseline
    # ------------------------------------------------------------------
    if not val_df.empty:
        X_val = val_df[LIVE_FEATURES].copy()
        y_val = val_df["actual_final_margin"].values

        raw_preds = model.predict(X_val)
        cal_preds = calibrator.predict(raw_preds)

        mae_model = mean_absolute_error(y_val, cal_preds)

        # Pre-game baseline: pregame_spread as projection
        pregame = val_df["pregame_spread"].fillna(0).values
        mae_baseline = mean_absolute_error(y_val, pregame)

        print(f"\n  Validation ({2025}, n={len(val_df)}):")
        print(f"    Live model MAE  (calibrated) : {mae_model:.3f} pts")
        print(f"    Pre-game spread MAE (baseline): {mae_baseline:.3f} pts")
        improvement = mae_baseline - mae_model
        print(
            f"    Improvement vs baseline       : {improvement:+.3f} pts "
            f"({'better' if improvement > 0 else 'worse'})"
        )
    else:
        print(f"  No validation data for year 2025 — skipping val metrics.")

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "live_spread_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Saved {model_path}")

    cal_path = MODELS_DIR / "live_spread_calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  Saved {cal_path}")

    print("\n=== Training complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_live_model()
