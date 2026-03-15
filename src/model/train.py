import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from src.utils.config import SPREAD_MODEL_PARAMS, TOTAL_MODEL_PARAMS, MODELS_DIR
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
    3. Train spread + total XGBoost models
    4. Train baseline
    5. Print CV RMSE summary
    """
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

    # FIX: Sort by year so TimeSeriesSplit works correctly.
    # The year information is on the index of y_spread/y_total (same index as X
    # since they come from build_training_matrix together). We sort by that index
    # which corresponds to chronological order if build_training_matrix processes
    # games in year order — verify this assumption holds in your DB query.
    # If not, join year back in before sorting.
    X = X.sort_index()
    y_spread = y_spread.sort_index()
    y_total = y_total.sort_index()

    print("\nTraining spread model...")
    spread_model = train_spread_model(X, y_spread)

    print("\nTraining total model...")
    total_model = train_total_model(X, y_total)

    print("\nTraining baseline models...")
    train_baseline_model(X, y_spread, y_total)

    print("\n=== Feature Importances (Spread) ===")
    fi = get_feature_importance(spread_model)
    print(fi.head(10).to_string(index=False))

    return spread_model, total_model