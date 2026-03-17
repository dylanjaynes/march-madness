"""
NIT data pipeline + backtest runner.
Usage:
  python scripts/run_nit_pipeline.py              # full pipeline
  python scripts/run_nit_pipeline.py --ingest-only
  python scripts/run_nit_pipeline.py --backtest-only
"""
import sys
import os
import json
import datetime
import argparse

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import PROCESSED_DIR
from src.ingest.nit import ingest_nit_results, ingest_nit_lines, get_nit_coverage_report
from src.model.nit_backtest import build_nit_training_matrix, run_nit_backtest, print_nit_report


def _safe(obj):
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, pd.DataFrame):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="NIT data pipeline + backtest")
    parser.add_argument("--ingest-only",   action="store_true",
                        help="Only run data ingestion (ESPN results + Odds API lines)")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Only run backtest (skip ingestion)")
    args = parser.parse_args()

    if not args.backtest_only:
        print("\n=== Ingesting NIT results from ESPN ===")
        ingest_nit_results()
        print("\n=== Ingesting NIT lines from Odds API ===")
        ingest_nit_lines()

    get_nit_coverage_report()

    if args.ingest_only:
        return

    print("\n=== Building NIT training matrix ===")
    nit_df = build_nit_training_matrix()

    print("\n=== Running walk-forward NIT backtest ===")
    results = run_nit_backtest(nit_df)
    print_nit_report(results)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds_df = results.pop("predictions_df", pd.DataFrame())

    payload = _safe(results)
    payload["computed_at"] = datetime.datetime.now().isoformat()
    payload["tournament"] = "NIT"

    out = PROCESSED_DIR / "nit_backtest_results.json"
    csv = PROCESSED_DIR / "nit_backtest_predictions.csv"

    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out}")

    if not preds_df.empty:
        preds_df.to_csv(csv, index=False)
        print(f"Saved → {csv} ({len(preds_df)} rows)")


if __name__ == "__main__":
    main()
