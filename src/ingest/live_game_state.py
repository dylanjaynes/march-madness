"""
Live game state polling — ESPN scoreboard + Odds API live lines.
Writes snapshots to live_game_snapshots table every 60s.
"""

import uuid
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.utils.config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT
from src.utils.db import db_conn, query_df
from src.utils.team_map import normalize_team_name, is_known_team

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/summary"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}

STATUS_HALFTIME    = "STATUS_HALFTIME"
STATUS_IN_PROGRESS = "STATUS_IN_PROGRESS"
STATUS_FINAL       = "STATUS_FINAL"

PREFERRED_BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "bovada"]


def _norm(name: str) -> str:
    return normalize_team_name(name) if is_known_team(name) else name


def compute_game_time(period: int, clock_display: str) -> Tuple[float, float]:
    """
    Parse clock string 'MM:SS' to (time_elapsed_minutes, time_remaining_minutes).
    period=1: time_elapsed = 20 - clock_mins, time_remaining = 20 + clock_mins
    period=2: time_elapsed = 20 + (20 - clock_mins), time_remaining = clock_mins
    Total always = 40 minutes.
    """
    try:
        parts = clock_display.strip().split(":")
        mins = float(parts[0])
        secs = float(parts[1]) if len(parts) > 1 else 0.0
        clock_mins = mins + secs / 60.0
    except Exception:
        clock_mins = 0.0

    if period == 1:
        time_elapsed   = 20.0 - clock_mins
        time_remaining = 20.0 + clock_mins
    elif period == 2:
        time_elapsed   = 20.0 + (20.0 - clock_mins)
        time_remaining = clock_mins
    else:
        # OT or unknown — treat as full-game elapsed
        time_elapsed   = 40.0
        time_remaining = 0.0

    # Clamp to valid range
    time_elapsed   = max(0.0, min(40.0, time_elapsed))
    time_remaining = max(0.0, min(40.0, time_remaining))
    return time_elapsed, time_remaining


def _fetch_live_odds_by_event() -> dict:
    """
    Attempt to fetch live market lines from The Odds API.
    Returns dict keyed by (norm_home, norm_away) -> spread_home float or None.
    Falls back gracefully if live lines unavailable.
    """
    if not ODDS_API_KEY:
        return {}

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american",
        "live": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=10, headers=HEADERS)
        if resp.status_code == 422:
            # live=true not supported — try without
            params.pop("live", None)
            resp = requests.get(url, params=params, timeout=10, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    result = {}
    for game in data:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home_norm = _norm(home_raw)
        away_norm = _norm(away_raw)

        bookmakers = game.get("bookmakers", [])
        bk_map = {b["key"]: b for b in bookmakers}
        bk = None
        for pref in PREFERRED_BOOKMAKERS:
            if pref in bk_map:
                bk = bk_map[pref]
                break
        if bk is None and bookmakers:
            bk = bookmakers[0]
        if bk is None:
            continue

        spread_home = None
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "spreads":
                continue
            for outcome in mkt.get("outcomes", []):
                if _norm(outcome.get("name", "")) == home_norm:
                    try:
                        spread_home = float(outcome["point"])
                    except Exception:
                        pass
                    break

        result[(home_norm, away_norm)] = spread_home

    return result


def _parse_box_score(competitors: list) -> dict:
    """
    Extract in-game box score stats from ESPN competitor objects.
    Returns dict with efg1, efg2, orb1, orb2, to1, to2 etc.
    """
    stats_out = {
        "efg_pct1": None, "efg_pct2": None,
        "orb1": None, "orb2": None,
        "to1": None, "to2": None,
        "score1": 0, "score2": 0,
        "team1": "", "team2": "",
    }
    if not competitors or len(competitors) < 2:
        return stats_out

    for idx, comp in enumerate(competitors[:2]):
        suffix = str(idx + 1)
        team_name = _norm(comp.get("team", {}).get("displayName", ""))
        stats_out[f"team{suffix}"] = team_name
        try:
            stats_out[f"score{suffix}"] = int(comp.get("score", 0))
        except Exception:
            pass

        for stat in comp.get("statistics", []):
            name = stat.get("name", "")
            try:
                val = float(stat.get("displayValue", "0").replace("%", ""))
            except Exception:
                val = None

            if name in ("effectiveFieldGoalPct", "fieldGoalPct"):
                if stats_out[f"efg_pct{suffix}"] is None:
                    stats_out[f"efg_pct{suffix}"] = val
            elif name == "offensiveRebounds":
                stats_out[f"orb{suffix}"] = int(val) if val is not None else None
            elif name == "turnovers":
                stats_out[f"to{suffix}"] = int(val) if val is not None else None

    return stats_out


def fetch_live_game_states() -> List[Dict]:
    """
    Calls ESPN scoreboard (no date param = today) + Odds API live markets.
    Joins on normalized team names.
    Returns list of game state dicts matching live_game_snapshots schema.
    If live Odds API lines unavailable, sets live_spread=None.
    """
    try:
        resp = requests.get(ESPN_SCOREBOARD, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        scoreboard = resp.json()
    except Exception as e:
        print(f"  [live_game_state] ESPN scoreboard error: {e}")
        return []

    live_odds = _fetch_live_odds_by_event()

    events = scoreboard.get("events", [])
    states = []

    for event in events:
        try:
            status_obj = event.get("status", {})
            status_type = status_obj.get("type", {})
            status_name = status_type.get("name", "")

            # Only process live or final games
            if status_name not in (STATUS_IN_PROGRESS, STATUS_HALFTIME, STATUS_FINAL):
                continue

            competitions = event.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            competitors = comp.get("competitors", [])

            if len(competitors) < 2:
                continue

            # ESPN: index 0 = home, index 1 = away (typically)
            home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            team1_raw = home_comp.get("team", {}).get("displayName", "")
            team2_raw = away_comp.get("team", {}).get("displayName", "")
            team1 = _norm(team1_raw)
            team2 = _norm(team2_raw)

            try:
                score1 = int(home_comp.get("score", 0))
                score2 = int(away_comp.get("score", 0))
            except Exception:
                score1, score2 = 0, 0

            current_margin = score1 - score2  # home perspective

            # Clock and period
            period = int(status_obj.get("period", 1))
            clock_display = status_obj.get("displayClock", "20:00")

            if status_name == STATUS_HALFTIME:
                time_elapsed   = 20.0
                time_remaining = 20.0
            elif status_name == STATUS_FINAL:
                time_elapsed   = 40.0
                time_remaining = 0.0
            else:
                time_elapsed, time_remaining = compute_game_time(period, clock_display)

            # Box score stats from competitor statistics
            box = _parse_box_score(competitors)

            # Live market spread lookup
            live_spread = live_odds.get((team1, team2))
            if live_spread is None:
                live_spread = live_odds.get((team2, team1))
                if live_spread is not None:
                    live_spread = -live_spread  # flip to home convention

            game_id = event.get("id", "")
            snap_ts = datetime.now(timezone.utc).isoformat()

            state = {
                "game_id":         game_id,
                "snapshot_ts":     snap_ts,
                "team1":           team1,
                "team2":           team2,
                "score1":          score1,
                "score2":          score2,
                "current_margin":  current_margin,
                "period":          period,
                "clock_display":   clock_display,
                "game_status":     status_name,
                "time_elapsed":    time_elapsed,
                "time_remaining":  time_remaining,
                "live_spread":     live_spread,
                # In-game box score stats
                "efg_pct1":        box.get("efg_pct1"),
                "efg_pct2":        box.get("efg_pct2"),
                "orb1":            box.get("orb1"),
                "orb2":            box.get("orb2"),
                "to1":             box.get("to1"),
                "to2":             box.get("to2"),
            }
            states.append(state)

        except Exception as e:
            print(f"  [live_game_state] Error parsing event: {e}")
            continue

    return states


def store_snapshot(game_state: dict) -> str:
    """
    Writes a single game state to live_game_snapshots.
    Returns snapshot_id (UUID).
    """
    snapshot_id = str(uuid.uuid4())
    try:
        with db_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO live_game_snapshots (
                    snapshot_id, game_id, snapshot_ts,
                    team1, team2, score1, score2, current_margin,
                    period, clock_display, game_status,
                    time_elapsed, time_remaining, live_spread,
                    efg_pct1, efg_pct2, orb1, orb2, to1, to2
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )""",
                [
                    snapshot_id,
                    game_state.get("game_id", ""),
                    game_state.get("snapshot_ts", datetime.now(timezone.utc).isoformat()),
                    game_state.get("team1", ""),
                    game_state.get("team2", ""),
                    game_state.get("score1", 0),
                    game_state.get("score2", 0),
                    game_state.get("current_margin", 0),
                    game_state.get("period", 1),
                    game_state.get("clock_display", ""),
                    game_state.get("game_status", ""),
                    game_state.get("time_elapsed", 0.0),
                    game_state.get("time_remaining", 40.0),
                    game_state.get("live_spread"),
                    game_state.get("efg_pct1"),
                    game_state.get("efg_pct2"),
                    game_state.get("orb1"),
                    game_state.get("orb2"),
                    game_state.get("to1"),
                    game_state.get("to2"),
                ],
            )
    except Exception as e:
        print(f"  [live_game_state] store_snapshot error: {e}")

    return snapshot_id


def fetch_live_game_states_with_pbp() -> List[Dict]:
    """
    Wraps fetch_live_game_states() and enriches each state with live PBP data.
    Sets state['pbp_available'] = True/False on each game.
    The existing fetch_live_game_states() is untouched.
    """
    from src.ingest.pbp_parser import parse_plays, compute_game_state_at

    states = fetch_live_game_states()

    for state in states:
        try:
            game_id = state.get("game_id")
            if not game_id:
                state["pbp_available"] = False
                continue

            resp = requests.get(
                ESPN_SUMMARY,
                params={"event": game_id},
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            # Silent skip for invalid ESPN IDs
            if "code" in data and "header" not in data:
                state["pbp_available"] = False
                continue

            plays_raw = data.get("plays", [])
            if not plays_raw:
                state["pbp_available"] = False
                continue

            # Get home team ID
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
            time_elapsed = state.get("time_elapsed", 20.0)

            pbp_state = compute_game_state_at(
                plays_parsed,
                at_time_elapsed=time_elapsed,
                home_team=state.get("team1", ""),
                away_team=state.get("team2", ""),
            )

            state.update(pbp_state)
            state["pbp_available"] = True

        except Exception as e:
            state["pbp_available"] = False
            print(f"  [pbp] {state.get('team1', '?')} warning: {e}")

    return states


def poll_live_games(pregame_spread_lookup: Optional[Dict] = None) -> List[Dict]:
    """
    Main polling function. Fetches states, stores snapshots, returns states.
    pregame_spread_lookup: {(team1, team2): float} for pre-game spread context.
    """
    pregame_spread_lookup = pregame_spread_lookup or {}
    states = fetch_live_game_states()

    for state in states:
        # Enrich with pre-game spread if available
        key1 = (state["team1"], state["team2"])
        key2 = (state["team2"], state["team1"])
        pregame = pregame_spread_lookup.get(key1) or pregame_spread_lookup.get(key2)
        if pregame is not None:
            state["pregame_spread"] = pregame

        store_snapshot(state)

    return states
