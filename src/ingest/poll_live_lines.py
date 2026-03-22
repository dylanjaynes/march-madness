"""
poll_live_lines.py
------------------
Captures live in-game spreads alongside live game state (score, clock).
Run every 5 minutes during tournament game windows to build a training
dataset of (game_state → live_market_spread) pairs.

Called by GitHub Actions cron and optionally by the Streamlit scheduler.
"""

from __future__ import annotations

import requests
from datetime import datetime, timezone

from src.utils.config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT
from src.utils.db import db_conn
from src.utils.team_map import normalize_team_name, is_known_team

PREFERRED_BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "bovada"]
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _select_bookmaker(bookmakers: list):
    bk_map = {b["key"]: b for b in bookmakers}
    for pref in PREFERRED_BOOKMAKERS:
        if pref in bk_map:
            return bk_map[pref]
    return bookmakers[0] if bookmakers else None


def _norm(name: str) -> str:
    return normalize_team_name(name) if is_known_team(name) else name


def _fetch_live_odds() -> dict:
    """
    Fetch current NCAAB odds (includes in-game lines when games are live).
    Returns dict keyed by frozenset({home_norm, away_norm}).
    """
    if not ODDS_API_KEY:
        return {}

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
        print(f"  [poll_live_lines] Odds API error: {e}")
        return {}

    result = {}
    for game in data:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = _norm(home_raw)
        away = _norm(away_raw)

        bookmakers = game.get("bookmakers", [])
        bk = _select_bookmaker(bookmakers)
        if not bk:
            continue

        spread_home = spread_away = total_line = None
        for market in bk.get("markets", []):
            if market["key"] == "spreads":
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == home_raw:
                        spread_home = outcome.get("point")
                    elif outcome["name"] == away_raw:
                        spread_away = outcome.get("point")
            elif market["key"] == "totals":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        total_line = outcome.get("point")

        key = frozenset({home.lower(), away.lower()})
        result[key] = {
            "game_id":    game.get("id", ""),
            "home_team":  home,
            "away_team":  away,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "total_line":  total_line,
            "bookmaker":   bk.get("key", ""),
        }

    return result


def _fetch_espn_live_states() -> list:
    """
    Fetch in-progress NCAAB games from ESPN scoreboard.
    Returns list of dicts with score, clock, period, status.
    """
    try:
        resp = requests.get(ESPN_SCOREBOARD, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [poll_live_lines] ESPN error: {e}")
        return []

    states = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status_obj = comp.get("status", {})
        status_type = status_obj.get("type", {})
        state_name = status_type.get("name", "")  # "STATUS_IN_PROGRESS", etc.
        completed = status_type.get("completed", False)

        # Only capture in-progress games
        if completed or state_name not in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME"):
            continue

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        try:
            score_home = int(home.get("score", 0))
            score_away = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue

        home_name = _norm(home.get("team", {}).get("displayName", ""))
        away_name  = _norm(away.get("team", {}).get("displayName", ""))

        # Period and clock
        period = status_obj.get("period", 1)
        clock_str = status_obj.get("displayClock", "0:00")
        try:
            parts = clock_str.split(":")
            clock_secs = int(parts[0]) * 60 + int(parts[1])
        except Exception:
            clock_secs = 0

        half_elapsed = (20 * 60 - clock_secs) / 60.0
        time_elapsed = half_elapsed if period == 1 else 20.0 + half_elapsed
        time_remaining = max(0.0, 40.0 - time_elapsed)

        states.append({
            "home_team":     home_name,
            "away_team":     away_name,
            "score_home":    score_home,
            "score_away":    score_away,
            "time_elapsed":  round(time_elapsed, 2),
            "time_remaining": round(time_remaining, 2),
            "period":        period,
            "game_status":   state_name,
        })

    return states


def poll_and_store_live_lines() -> int:
    """
    Fetch live odds + ESPN game states, join by team, store in live_lines_history.
    Returns number of rows inserted.
    """
    now = datetime.now(timezone.utc).isoformat()

    live_odds = _fetch_live_odds()
    espn_states = _fetch_espn_live_states()

    if not espn_states:
        print(f"  [poll_live_lines] No live games at {now}")
        return 0

    inserted = 0
    with db_conn() as conn:
        for state in espn_states:
            home = state["home_team"]
            away = state["away_team"]
            key = frozenset({home.lower(), away.lower()})

            odds = live_odds.get(key, {})

            conn.execute(
                """INSERT INTO live_lines_history
                   (pull_timestamp, game_id, home_team, away_team,
                    score_home, score_away, time_elapsed, time_remaining,
                    game_status, period, spread_home, spread_away, total_line, bookmaker)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    now,
                    odds.get("game_id", ""),
                    home,
                    away,
                    state["score_home"],
                    state["score_away"],
                    state["time_elapsed"],
                    state["time_remaining"],
                    state["game_status"],
                    state["period"],
                    odds.get("spread_home"),
                    odds.get("spread_away"),
                    odds.get("total_line"),
                    odds.get("bookmaker", ""),
                ],
            )
            inserted += 1
            print(
                f"  [poll_live_lines] {home} vs {away} | "
                f"score {state['score_home']}-{state['score_away']} | "
                f"t={state['time_elapsed']:.1f}min | "
                f"spread_home={odds.get('spread_home')}"
            )

    print(f"  [poll_live_lines] Stored {inserted} rows at {now}")
    return inserted


if __name__ == "__main__":
    poll_and_store_live_lines()
