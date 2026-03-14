"""
Reload all historical betting lines and join to training data.

Steps:
  1. Fetch historical odds from The Odds API (all years)
  2. Ingest any SBRO Excel files found in data/raw/ (2019/2021 fallback)
  3. Clear + re-join market lines into mm_training_data
  4. Print coverage report

Usage:
    python scripts/reload_lines.py [--years 2022 2023 2024 2025] [--skip-api]

Options:
    --years      Only process specific years (default: all)
    --skip-api   Skip The Odds API fetch (re-join only)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.db import db_conn
from src.ingest.odds_historical import ingest_all_historical_odds
from src.ingest.sbro import ingest_all_sbro
from src.ingest.join_lines import join_lines_to_training, report_line_coverage

# Parse args
skip_api = "--skip-api" in sys.argv
years = None
if "--years" in sys.argv:
    idx = sys.argv.index("--years")
    years = [int(y) for y in sys.argv[idx + 1:] if y.isdigit()]

print("=" * 60)

if not skip_api:
    print("STEP 1: Fetching historical odds from The Odds API")
    print("=" * 60)
    api_results = ingest_all_historical_odds(years=years)
    print(f"API results: {api_results}")
else:
    print("STEP 1: Skipped (--skip-api)")
    api_results = {}

print()
print("=" * 60)
print("STEP 2: Ingesting SBRO lines (2019/2021 fallback if present)")
print("=" * 60)
sbro_results = ingest_all_sbro()
print(f"SBRO results: {sbro_results}")

print()
print("=" * 60)
print("STEP 3: Clearing and re-joining market lines to training data")
print("=" * 60)
if years:
    placeholders = ",".join("?" * len(years))
    with db_conn() as conn:
        conn.execute(
            f"UPDATE mm_training_data SET market_spread=NULL, market_total=NULL WHERE year IN ({placeholders})",
            years,
        )
    print(f"  Cleared lines for years: {years}")
else:
    with db_conn() as conn:
        conn.execute("UPDATE mm_training_data SET market_spread=NULL, market_total=NULL")
    print("  Cleared all market lines")

join_lines_to_training()

print()
print("=" * 60)
print("COVERAGE REPORT")
print("=" * 60)
report_line_coverage()

print()
print("To run full backtest:")
print("  python scripts/run_backtest.py")
