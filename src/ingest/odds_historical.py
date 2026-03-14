"""
Fetch historical NCAA Tournament betting lines from The Odds API.

Uses /v4/historical/sports/basketball_ncaab/odds endpoint to pull
closing-line snapshots for each tournament game date across all years.

Each unique game date costs 1 API request.
Tournament has ~10 unique game dates/year × 6 years = ~60 requests total.
"""

import time
import requests
import pandas as pd
from src.utils.config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, TOURNAMENT_YEARS
from src.utils.db import db_conn, query_df
from src.utils.team_map import normalize_team_name, is_known_team

PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "bovada", "williamhill_us"]
HISTORICAL_URL = f"{ODDS_API_BASE}/historical/sports/{ODDS_SPORT}/odds"


def _norm(name: str) -> str:
    return normalize_team_name(name) if is_known_team(name) else name


def _select_outcomes(bookmakers: list, market_key: str) -> list:
    """Return outcomes list from the best available bookmaker."""
    bk_map = {b["key"]: b for b in bookmakers}
    for pref in PREFERRED_BOOKS:
        if pref not in bk_map:
            continue
        for market in bk_map[pref].get("markets", []):
            if market["key"] == market_key:
                return market.get("outcomes", [])
    # Fallback: first bookmaker that has the market
    for bk in bookmakers:
        for market in bk.get("markets", []):
            if market["key"] == market_key:
                return market.get("outcomes", [])
    return []


def fetch_odds_snapshot(date_str: str) -> list:
    """
    Call the historical endpoint and return the games list.
    date_str: ISO 8601 UTC, e.g. '2022-03-17T23:00:00Z'
    """
    params = {
        "apiKey": ODDS_API_KEY,
        "date": date_str,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(HISTORICAL_URL, params=params, timeout=20)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"      quota: {remaining} remaining / {used} used")
        resp.raise_for_status()
        data = resp.json()
        # Historical endpoint wraps results in {"data": [...]}
        return data.get("data", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"      [odds_historical] API error for {date_str}: {e}")
        return []


def _build_lines_map(games_api: list) -> dict:
    """
    Index API games by frozenset of normalized lower-case team names.
    Returns dict: key → {home, away, spread_home, spread_away, total}
    """
    result = {}
    for game in games_api:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = _norm(home_raw)
        away = _norm(away_raw)
        key = frozenset({home.lower(), away.lower()})

        spread_outs = _select_outcomes(game.get("bookmakers", []), "spreads")
        total_outs = _select_outcomes(game.get("bookmakers", []), "totals")

        spread_home = spread_away = total_line = None
        for o in spread_outs:
            oname = _norm(o.get("name", "")).lower()
            if oname == home.lower():
                spread_home = o.get("point")
            elif oname == away.lower():
                spread_away = o.get("point")
        for o in total_outs:
            if o.get("name") == "Over":
                total_line = o.get("point")

        result[key] = {
            "home": home,
            "away": away,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "total": total_line,
        }
    return result


def _fuzzy_match(t1: str, t2: str, lines_map: dict):
    """Substring fallback if exact frozenset key misses."""
    for key, line in lines_map.items():
        api_teams = list(key)
        hit1 = any(t1 in at or at in t1 for at in api_teams)
        hit2 = any(t2 in at or at in t2 for at in api_teams)
        if hit1 and hit2:
            return line
    return None


def ingest_historical_odds_for_year(year: int) -> int:
    """
    Fetch and store closing lines for all tournament games in year.
    Returns number of rows upserted into historical_lines.
    """
    results = query_df(
        "SELECT game_date, team1, team2 FROM historical_results WHERE year = ? ORDER BY game_date",
        params=[year],
    )
    if results.empty:
        print(f"  No historical_results for {year} — run build_historical_dataset() first")
        return 0

    unique_dates = sorted(results["game_date"].dropna().unique())
    print(f"  {year}: {len(unique_dates)} game dates, {len(results)} games")

    stored = 0
    for game_date in unique_dates:
        # Query at 14:00 UTC ≈ 9-10 AM ET — before any game tips off,
        # so all tournament games still have open pre-game lines in the feed
        date_str = f"{game_date}T14:00:00Z"
        print(f"    {game_date} → snapshot at {date_str}")

        games_api = fetch_odds_snapshot(date_str)
        if not games_api:
            print(f"      No data returned")
            time.sleep(1)
            continue

        print(f"      {len(games_api)} games in API response")
        lines_map = _build_lines_map(games_api)

        day_games = results[results["game_date"] == game_date]
        for _, row in day_games.iterrows():
            t1_raw = row["team1"]
            t2_raw = row["team2"]
            t1 = _norm(t1_raw).lower()
            t2 = _norm(t2_raw).lower()

            line = lines_map.get(frozenset({t1, t2}))
            if line is None:
                line = _fuzzy_match(t1, t2, lines_map)
                if line:
                    print(f"      fuzzy: {t1_raw} vs {t2_raw}")

            if line is None:
                print(f"      NO MATCH: {t1_raw} vs {t2_raw}")
                continue

            # Align spread_line to team1's perspective and convert to
            # "positive = team1 favored" convention (matches SBRO source).
            # The Odds API uses standard convention: negative = favored.
            # Negate so that: positive stored value = team1 is the favorite.
            home_norm = line["home"].lower()
            if home_norm == t1:
                # home team == team1; spread_home is team1's line (neg = favored)
                raw_spread = line["spread_home"]
            elif home_norm == t2:
                # home team == team2; use spread_away for team1's perspective
                raw_spread = line["spread_away"]
            else:
                raw_spread = line["spread_home"]

            # Negate: API -33 (favored) → stored +33 (positive = team1 favored)
            spread_team1 = -raw_spread if raw_spread is not None else None

            spread_fav = t1_raw if (spread_team1 is not None and spread_team1 > 0) else t2_raw

            with db_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO historical_lines
                       (year, game_date, team1, team2, spread_favorite,
                        spread_line, total_line, open_spread, open_total,
                        ats_result, ou_result, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)""",
                    [year, game_date, t1_raw, t2_raw, spread_fav,
                     spread_team1, line["total"], "odds_api_historical"],
                )
            stored += 1

        time.sleep(0.5)  # be polite with the API

    print(f"  Stored {stored} lines for {year}")
    return stored


def ingest_all_historical_odds(years=None) -> dict:
    """
    Ingest historical odds for all (or specified) tournament years.
    Returns {year: lines_stored}.
    """
    if not ODDS_API_KEY:
        print("[odds_historical] No ODDS_API_KEY set")
        return {}

    target = years or TOURNAMENT_YEARS
    totals = {}
    for year in target:
        print(f"\n=== {year} ===")
        totals[year] = ingest_historical_odds_for_year(year)

    print(f"\n[odds_historical] Done. Lines stored: {totals}")
    return totals
