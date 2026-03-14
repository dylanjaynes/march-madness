import requests
import pandas as pd
from datetime import datetime, timezone

from src.utils.config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT
from src.utils.db import db_conn, upsert_df
from src.utils.team_map import normalize_team_name, is_known_team

PREFERRED_BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "bovada"]


def _select_bookmaker(bookmakers: list):
    """Pick the best available bookmaker in preference order."""
    bk_map = {b["key"]: b for b in bookmakers}
    for pref in PREFERRED_BOOKMAKERS:
        if pref in bk_map:
            return bk_map[pref]
    return bookmakers[0] if bookmakers else None


def fetch_current_games() -> pd.DataFrame:
    """
    Fetch current NCAAB odds from The Odds API.
    Returns DataFrame with spread and total for each game.
    """
    if not ODDS_API_KEY:
        print("  [odds] No ODDS_API_KEY set — returning empty DataFrame")
        return pd.DataFrame()

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [odds] API error: {e}")
        return pd.DataFrame()

    rows = []
    for game in data:
        game_id = game.get("id", "")
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        commence_time = game.get("commence_time", "")

        home_team = normalize_team_name(home_raw) if is_known_team(home_raw) else home_raw
        away_team = normalize_team_name(away_raw) if is_known_team(away_raw) else away_raw

        bookmakers = game.get("bookmakers", [])
        bk = _select_bookmaker(bookmakers)
        if not bk:
            continue

        spread_home = None
        spread_away = None
        total_line = None

        for market in bk.get("markets", []):
            if market["key"] == "spreads":
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == home_raw:
                        spread_home = outcome.get("point")
                    elif outcome["name"] == away_raw:
                        spread_away = outcome.get("point")
            elif market["key"] == "totals":
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == "Over":
                        total_line = outcome.get("point")

        rows.append({
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "total_line": total_line,
            "bookmaker": bk.get("key", ""),
        })

    return pd.DataFrame(rows)


def poll_and_store_odds():
    """Fetch current odds and store in odds_history table."""
    df = fetch_current_games()
    if df.empty:
        return

    now = datetime.now(timezone.utc).isoformat()
    df["pull_timestamp"] = now
    df["is_opening"] = False

    upsert_df(df, "odds_history", if_exists="append")
    print(f"  [odds] Stored {len(df)} games at {now}")


def get_latest_odds() -> pd.DataFrame:
    """Get most recent odds from DB for each game."""
    from src.utils.db import query_df
    sql = """
        SELECT oh.*
        FROM odds_history oh
        INNER JOIN (
            SELECT game_id, MAX(pull_timestamp) AS max_ts
            FROM odds_history
            GROUP BY game_id
        ) latest ON oh.game_id = latest.game_id AND oh.pull_timestamp = latest.max_ts
    """
    return query_df(sql)


def fetch_historical_odds(year: int) -> pd.DataFrame:
    """
    Historical odds are not available on the free Odds API tier.
    This stub returns empty DataFrame — historical lines must be manually seeded.
    """
    print(f"  [odds] Historical odds for {year} not available (requires paid API tier)")
    return pd.DataFrame()
