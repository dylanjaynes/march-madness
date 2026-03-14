"""
Join historical betting lines into mm_training_data.

Match logic:
  For each training row (year, team_a, team_b), look in historical_lines
  for a row where (team1 = team_a OR team1 = team_b) AND year matches.
  Normalize team names before matching.

Spread sign convention coming in from SBRO/OddsPortal:
  spread_line is from team1's perspective (negative = team1 favored).
  We convert to team_a perspective: positive = team_a favored.
"""

import pandas as pd
import numpy as np
from src.utils.db import query_df, db_conn
from src.utils.team_map import normalize_team_name, is_known_team
import warnings


def _norm(name: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalize_team_name(str(name)) if is_known_team(str(name)) else str(name)


def join_lines_to_training():
    """
    For every row in mm_training_data that has NULL market_spread,
    try to find a matching row in historical_lines and fill it in.
    Returns count of rows updated.
    """
    training = query_df("SELECT rowid, * FROM mm_training_data")
    lines = query_df("SELECT * FROM historical_lines WHERE spread_line IS NOT NULL")

    if training.empty or lines.empty:
        print(f"  [join_lines] training={len(training)} rows, lines={len(lines)} rows — nothing to join")
        return 0

    # Build lines index: (year, frozenset({team1_norm, team2_norm})) → row
    lines["_t1"] = lines["team1"].apply(_norm).str.lower()
    lines["_t2"] = lines["team2"].apply(_norm).str.lower()
    lines_idx = {}
    for _, lr in lines.iterrows():
        key = (int(lr["year"]), frozenset({lr["_t1"], lr["_t2"]}))
        if key not in lines_idx:
            lines_idx[key] = lr

    updated = 0
    skipped = 0
    for _, tr in training.iterrows():
        if pd.notna(tr.get("market_spread")):
            continue  # already has a line

        year = int(tr["year"])
        ta = _norm(tr["team_a"]).lower()
        tb = _norm(tr["team_b"]).lower()
        key = (year, frozenset({ta, tb}))

        lr = lines_idx.get(key)
        if lr is None:
            skipped += 1
            continue

        spread_line = lr.get("spread_line")
        total_line = lr.get("total_line")

        # Convert spread to team_a perspective
        # In historical_lines: spread_line = team1's spread (neg = team1 favored)
        # We want: positive = team_a favored
        if pd.notna(spread_line):
            t1_norm = lr["_t1"]
            if t1_norm == ta:
                # team1 = team_a, so spread already in team_a perspective
                market_spread_a = float(spread_line)
            else:
                # team1 = team_b, flip sign
                market_spread_a = -float(spread_line)
        else:
            market_spread_a = None

        market_total = float(total_line) if pd.notna(total_line) else None

        with db_conn() as conn:
            conn.execute(
                """UPDATE mm_training_data
                   SET market_spread = ?, market_total = ?
                   WHERE year = ? AND team_a = ? AND team_b = ?""",
                [market_spread_a, market_total,
                 tr["year"], tr["team_a"], tr["team_b"]]
            )
        updated += 1

    print(f"  [join_lines] Updated {updated} training rows with market lines ({skipped} unmatched)")
    return updated


def report_line_coverage():
    """Print how many training rows have market lines by year."""
    df = query_df("""
        SELECT year,
               COUNT(*) as total,
               SUM(CASE WHEN market_spread IS NOT NULL THEN 1 ELSE 0 END) as has_spread,
               SUM(CASE WHEN market_total IS NOT NULL THEN 1 ELSE 0 END) as has_total
        FROM mm_training_data
        GROUP BY year
        ORDER BY year
    """)
    if df.empty:
        print("No training data.")
        return
    print("\nLine coverage by year:")
    print(f"{'Year':<6} {'Total':<8} {'Has Spread':<12} {'Has Total':<10}")
    for _, r in df.iterrows():
        pct = r["has_spread"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  {r['year']:<4} {r['total']:<8} {r['has_spread']:<6} ({pct:.0f}%)  {r['has_total']}")
