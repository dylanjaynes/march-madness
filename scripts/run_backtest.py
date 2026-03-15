"""
Run the full walk-forward backtest and persist results for Streamlit display.

Usage:
    python scripts/run_backtest.py

Outputs:
    data/processed/backtest_results.json   — aggregate + per-year metrics
    data/processed/backtest_predictions.csv — full game-level predictions
"""
import sys
import os
import json
import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.model.backtest import run_backtest, generate_backtest_report
from src.utils.config import TOURNAMENT_YEARS, PROCESSED_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_json_safe(obj):
    """
    Recursively cast numpy/tuple types to JSON-serializable Python natives.
    DataFrames are excluded (None) — they go to the CSV instead.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None  # JSON doesn't support NaN/Inf
    if isinstance(obj, pd.DataFrame):
        return None  # excluded; saved separately as CSV
    return obj


# ── Run backtest ──────────────────────────────────────────────────────────────

print("Running walk-forward backtest...")
results = run_backtest(TOURNAMENT_YEARS)

# ── Standard report (RMSE, directional ATS) ───────────────────────────────────
report = generate_backtest_report(results)
print("\n" + "=" * 80)
print("WALK-FORWARD BACKTEST REPORT")
print("=" * 80)
print(report.to_string(index=False))

# ── True ATS against market spread ────────────────────────────────────────────
df = results.get("predictions_df")
if df is None or df.empty:
    print("\nNo predictions to analyze.")
    sys.exit(0)

real = df[df["market_spread"].abs() > 0.01].copy()
print(f"\n{'='*80}")
print(f"TRUE ATS BACKTEST (games with real market lines: {len(real)} of {len(df)})")
print("=" * 80)

print(f"\nCoverage by year:")
for yr, grp in df.groupby("year"):
    real_yr = grp[grp["market_spread"].abs() > 0.01]
    print(f"  {yr}: {len(real_yr)}/{len(grp)} games with real lines")

real["actual_cover"] = real.apply(
    lambda r: "WIN" if r["actual_margin"] > r["market_spread"]
    else ("PUSH" if r["actual_margin"] == r["market_spread"] else "LOSS"),
    axis=1,
)
ats = real["actual_cover"].value_counts()
aw, al, ap = ats.get("WIN", 0), ats.get("LOSS", 0), ats.get("PUSH", 0)
pct_cover = f"{aw/(aw+al)*100:.1f}%" if (aw + al) > 0 else "n/a"
print(f"\nMarket calibration (fav-side covers): {aw}-{al}-{ap}  ({pct_cover})")

real["edge"] = real["model_spread"] - real["market_spread"]
real["true_model_cover"] = real.apply(
    lambda r: (
        (r["model_spread"] > r["market_spread"] and r["actual_margin"] > r["market_spread"]) or
        (r["model_spread"] < r["market_spread"] and r["actual_margin"] < r["market_spread"])
    ),
    axis=1,
)

print(f"\nModel true ATS vs market spread by edge threshold:")
print(f"{'Threshold':<12} {'Bets':<6} {'W':<5} {'L':<5} {'W%':<8} {'ROI':<8}")
for thresh in [0, 1, 2, 3, 5, 7]:
    sub = real[real["edge"].abs() >= thresh]
    if len(sub) == 0:
        continue
    w = sub["true_model_cover"].sum()
    l = len(sub) - w
    pct = w / len(sub) * 100
    stake = 110; payout = 100
    roi = ((w * (stake + payout) + 0) - len(sub) * stake) / (len(sub) * stake) * 100
    print(f"  >= {thresh:<8} {len(sub):<6} {w:<5} {l:<5} {pct:.1f}%    {roi:.1f}%")

print(f"\nTrue ATS (edge >= 3) by year:")
for yr, grp in real.groupby("year"):
    sub = grp[grp["edge"].abs() >= 3]
    if len(sub) == 0:
        continue
    w = sub["true_model_cover"].sum()
    l = len(sub) - w
    pct = w / len(sub) * 100 if len(sub) > 0 else 0
    print(f"  {yr}: {w}-{l} ({pct:.1f}%)")

ou = real[real["market_total"].abs() > 0.01].copy()
if not ou.empty:
    ou["ou_result"] = ou.apply(
        lambda r: "OVER" if r["actual_total"] > r["market_total"]
        else ("PUSH" if r["actual_total"] == r["market_total"] else "UNDER"),
        axis=1,
    )
    oc = ou["ou_result"].value_counts()
    print(f"\n{'='*80}")
    print(f"TRUE O/U BACKTEST ({len(ou)} games)")
    print("=" * 80)
    print(f"\nMarket calibration: {oc.get('OVER',0)} OVER / {oc.get('UNDER',0)} UNDER / {oc.get('PUSH',0)} PUSH")

    ou["ou_edge"] = ou["model_total"] - ou["market_total"]
    ou["model_ou_cover"] = ou.apply(
        lambda r: (
            (r["model_total"] > r["market_total"] and r["actual_total"] > r["market_total"]) or
            (r["model_total"] < r["market_total"] and r["actual_total"] < r["market_total"])
        ),
        axis=1,
    )

    print(f"\nModel O/U vs market by edge threshold:")
    print(f"{'Threshold':<12} {'Bets':<6} {'W':<5} {'L':<5} {'W%':<8} {'ROI':<8}")
    for thresh in [0, 1, 2, 3, 5]:
        sub = ou[ou["ou_edge"].abs() >= thresh]
        if len(sub) == 0:
            continue
        w = sub["model_ou_cover"].sum()
        l = len(sub) - w
        pct = w / len(sub) * 100
        roi = ((w * 210) - len(sub) * 110) / (len(sub) * 110) * 100
        print(f"  >= {thresh:<8} {len(sub):<6} {w:<5} {l:<5} {pct:.1f}%    {roi:.1f}%")


# ── Persist results ───────────────────────────────────────────────────────────

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
results_path = PROCESSED_DIR / "backtest_results.json"
preds_path = PROCESSED_DIR / "backtest_predictions.csv"

# Build JSON-safe dict (exclude predictions_df — goes to CSV)
json_payload = {k: v for k, v in results.items() if k != "predictions_df"}
json_payload["computed_at"] = datetime.datetime.now().isoformat()
json_payload = _to_json_safe(json_payload)

with open(results_path, "w") as f:
    json.dump(json_payload, f, indent=2)
print(f"\nResults saved   → {results_path}")

# Save predictions CSV
csv_cols = [
    "year", "game_idx", "round_number", "is_mismatch",
    "model_spread", "model_total",
    "market_spread", "market_total",
    "spread_edge", "total_edge",
    "actual_margin", "actual_total",
    "ats_result", "ou_result",
]
save_df = df[[c for c in csv_cols if c in df.columns]].copy()
save_df.to_csv(preds_path, index=False)
print(f"Predictions saved → {preds_path}  ({len(save_df)} rows)")
print(f"\nDone. Computed at: {json_payload['computed_at']}")
