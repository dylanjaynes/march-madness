"""
Run the full walk-forward backtest and print true ATS + O/U results.

Usage:
    python scripts/run_backtest.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.model.backtest import run_backtest, generate_backtest_report
from src.utils.config import TOURNAMENT_YEARS

print("Running walk-forward backtest...")
results = run_backtest(TOURNAMENT_YEARS)

# ── Standard report (RMSE, directional ATS) ─────────────────────────────────
report = generate_backtest_report(results)
print("\n" + "=" * 80)
print("WALK-FORWARD BACKTEST REPORT")
print("=" * 80)
print(report.to_string(index=False))

# ── True ATS against market spread ──────────────────────────────────────────
df = results.get("predictions_df")
if df is None or df.empty:
    print("\nNo predictions to analyze.")
    sys.exit(0)

# Only evaluate games where we have a REAL market spread (not placeholder 0)
real = df[df["market_spread"].abs() > 0.01].copy()
print(f"\n{'='*80}")
print(f"TRUE ATS BACKTEST (games with real market lines: {len(real)} of {len(df)})")
print("=" * 80)

print(f"\nCoverage by year:")
for yr, grp in df.groupby("year"):
    real_yr = grp[grp["market_spread"].abs() > 0.01]
    print(f"  {yr}: {len(real_yr)}/{len(grp)} games with real lines")

# Market calibration: favorite covers (spread_line is from team_a perspective)
real["actual_cover"] = real.apply(
    lambda r: "WIN" if r["actual_margin"] > r["market_spread"]
    else ("PUSH" if r["actual_margin"] == r["market_spread"] else "LOSS"),
    axis=1,
)
ats = real["actual_cover"].value_counts()
aw, al, ap = ats.get("WIN", 0), ats.get("LOSS", 0), ats.get("PUSH", 0)
print(f"\nMarket calibration (fav-side covers): {aw}-{al}-{ap}  ({aw/(aw+al)*100:.1f}%)")

# True model ATS: bet team_a when model_spread > market_spread
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
    # ROI at -110 juice
    stake = 110; payout = 100
    roi = ((w * (stake + payout) + 0) - len(sub) * stake) / (len(sub) * stake) * 100
    print(f"  >= {thresh:<8} {len(sub):<6} {w:<5} {l:<5} {pct:.1f}%    {roi:.1f}%")

# By year (edge >= 3)
print(f"\nTrue ATS (edge >= 3) by year:")
for yr, grp in real.groupby("year"):
    sub = grp[grp["edge"].abs() >= 3]
    if len(sub) == 0:
        continue
    w = sub["true_model_cover"].sum()
    l = len(sub) - w
    pct = w / len(sub) * 100 if len(sub) > 0 else 0
    print(f"  {yr}: {w}-{l} ({pct:.1f}%)")

# ── True O/U ─────────────────────────────────────────────────────────────────
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
