"""
pbp_backfill.py
---------------
Fetches and stores play-by-play data for all historical tournament games
that have a known espn_game_id. Run once to backfill 2019-2025.

2020 is excluded (tournament cancelled).
Games that return empty plays[] are logged and skipped.
"""

from __future__ import annotations

import time
import requests

from src.ingest.pbp_parser import parse_plays
from src.utils.db import db_conn, query_df

ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/summary"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_and_store_game_pbp(
    espn_game_id: str,
    game_date: str,
    team1: str,
    team2: str,
) -> int:
    """
    Fetch ESPN summary for one game, parse plays, upsert to pbp_plays.
    Returns number of plays stored (0 if no play data available).
    """
    try:
        resp = requests.get(
            ESPN_SUMMARY,
            params={"event": espn_game_id},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [pbp] fetch error for {espn_game_id}: {e}")
        return 0

    # Silent skip for invalid IDs (e.g. 2019 games returning {"code": 404})
    if "code" in data and "header" not in data:
        print(f"    [pbp] {espn_game_id}: ESPN returned code={data.get('code')} — skipping")
        return 0

    plays_raw = data.get("plays", [])
    if not plays_raw:
        print(f"    [pbp] {espn_game_id}: no plays array — skipping")
        return 0

    # Get home team ID from competitions
    try:
        competitions = data.get("header", {}).get("competitions", data.get("competitions", []))
        competitors  = competitions[0].get("competitors", [])
        home_comp    = next(
            (c for c in competitors if c.get("homeAway") == "home"),
            competitors[0],
        )
        home_id = home_comp.get("team", {}).get("id", "")
    except Exception:
        home_id = ""

    plays_parsed = parse_plays(plays_raw, home_id)

    rows = [
        {
            "play_id":      p["play_id"],
            "espn_game_id": espn_game_id,
            "game_date":    game_date,
            "team1":        team1,
            "team2":        team2,
            "period":       p["period"],
            "clock_secs":   p["clock_secs"],
            "time_elapsed": p["time_elapsed"],
            "event_type":   p["event_type"],
            "team":         p["team"],
            "score_value":  p["score_value"],
            "home_score":   p["home_score"],
            "away_score":   p["away_score"],
            "margin":       p["margin"],
            "is_fg_attempt": int(p["is_fg_attempt"]),
            "is_fg_made":    int(p["is_fg_made"]),
            "is_3pt":        int(p["is_3pt"]),
            "raw_text":      p["raw_text"],
        }
        for p in plays_parsed
    ]

    if not rows:
        return 0

    with db_conn() as conn:
        for row in rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO pbp_plays VALUES (
                    :play_id, :espn_game_id, :game_date, :team1, :team2,
                    :period, :clock_secs, :time_elapsed, :event_type, :team,
                    :score_value, :home_score, :away_score, :margin,
                    :is_fg_attempt, :is_fg_made, :is_3pt, :raw_text
                )
                """,
                row,
            )

    return len(rows)


def run_backfill(years: list = None) -> dict:
    """
    Fetch PBP for all historical tournament games with a known espn_game_id.
    2020 excluded (tournament cancelled).
    """
    if years is None:
        years = [2019, 2021, 2022, 2023, 2024, 2025]

    placeholders = ",".join("?" * len(years))
    games = query_df(
        f"""
        SELECT espn_game_id, game_date, team1, team2, year
        FROM historical_results
        WHERE espn_game_id IS NOT NULL
          AND year IN ({placeholders})
        ORDER BY year, game_date
        """,
        params=years,
    )

    if games.empty:
        print("No games with espn_game_id found — run espn_enrichment first.")
        return {"games": 0, "plays": 0}

    print(f"[pbp_backfill] Starting backfill: {len(games)} games across years {years}")
    total_plays = 0
    skipped     = 0

    for i, row in games.iterrows():
        n = fetch_and_store_game_pbp(
            espn_game_id=str(row["espn_game_id"]),
            game_date=str(row["game_date"]),
            team1=str(row["team1"]),
            team2=str(row["team2"]),
        )
        total_plays += n
        if n == 0:
            skipped += 1

        idx = list(games.index).index(i) + 1
        print(
            f"  [{idx}/{len(games)}] {row['year']} | {row['team1']} vs {row['team2']}: "
            f"{n} plays stored"
        )
        time.sleep(1.5)

    print(
        f"\n[pbp_backfill] Complete: {total_plays} plays across "
        f"{len(games) - skipped}/{len(games)} games "
        f"({skipped} skipped / no data)"
    )
    return {"games": len(games), "plays": total_plays, "skipped": skipped}


if __name__ == "__main__":
    from src.utils.db import init_db
    init_db()
    run_backfill()
