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


def fetch_and_store_scores(year: int, days_from: int = 3) -> dict:
    """
    Fetch completed NCAAB game scores from The Odds API /scores endpoint,
    including pre-game closing lines (requires paid tier).

    Upserts into two tables:
      historical_results  — game scores, teams, round info
      historical_lines    — pre-game spread and total for each game

    The /scores endpoint returns up to `days_from` days of completed games.
    Set days_from=3 to get the last 3 days; max is 3 on most plans.

    Returns dict with counts: {"results": N, "lines": N}
    """
    from src.utils.team_map import normalize_team_name, is_known_team
    from src.utils.db import query_df

    if not ODDS_API_KEY:
        print("  [odds] No ODDS_API_KEY — cannot fetch scores")
        return {"results": 0, "lines": 0}

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/scores"
    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": days_from,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  [odds] Scores fetch error: {e}")
        return {"results": 0, "lines": 0}

    # Load existing tournament bracket to get seeds
    bracket_df = query_df(
        "SELECT team, seed FROM tournament_bracket WHERE year = ?",
        params=[year],
    )
    seed_map = {}
    if not bracket_df.empty:
        for _, row in bracket_df.iterrows():
            seed_map[normalize_team_name(str(row["team"])).lower()] = int(row["seed"] or 8)

    result_rows = []
    line_rows = []

    for game in games:
        # Only completed games
        if not game.get("completed", False):
            continue

        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        commence = game.get("commence_time", "")

        home = normalize_team_name(home_raw) if is_known_team(home_raw) else home_raw
        away = normalize_team_name(away_raw) if is_known_team(away_raw) else away_raw

        # Parse game date from commence_time ISO string
        try:
            game_date = commence[:10]  # "YYYY-MM-DD"
        except Exception:
            game_date = None

        # Extract final scores
        score_home = score_away = None
        for team_score in game.get("scores", []) or []:
            name = team_score.get("name", "")
            val  = team_score.get("score")
            try:
                val = int(val)
            except (TypeError, ValueError):
                val = None
            if name == home_raw:
                score_home = val
            elif name == away_raw:
                score_away = val

        if score_home is None or score_away is None:
            continue

        winner = home if score_home > score_away else away
        home_seed = seed_map.get(home.lower(), 8)
        away_seed = seed_map.get(away.lower(), 8)

        # First Four tipoffs are March 18-19 2026; everything else is R64+
        round_num  = 0 if game_date in ("2026-03-18", "2026-03-19") else 1
        round_name = "First Four" if round_num == 0 else "R64"

        result_rows.append({
            "year":         year,
            "round_number": round_num,
            "round_name":   round_name,
            "game_date":    game_date,
            "team1":        home,
            "team2":        away,
            "score1":       score_home,
            "score2":       score_away,
            "winner":       winner,
            "margin":       abs(score_home - score_away),
            "total_points": score_home + score_away,
            "seed1":        home_seed,
            "seed2":        away_seed,
        })

        # ── Pre-game closing lines ─────────────────────────────────────────────
        # Paid tier returns bookmakers[] on /scores with daysFrom set
        spread_line = total_line = spread_favorite = None
        bookmakers = game.get("bookmakers", []) or []
        bk = _select_bookmaker(bookmakers) if bookmakers else None
        if bk:
            for market in bk.get("markets", []) or []:
                if market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        pt = outcome.get("point")
                        if pt is not None:
                            try:
                                pt_f = float(pt)
                            except (TypeError, ValueError):
                                continue
                            if pt_f < 0:
                                # Negative point = this team is the favorite
                                name = outcome.get("name", "")
                                fav = normalize_team_name(name) if is_known_team(name) else name
                                spread_favorite = fav
                                spread_line = abs(pt_f)
                elif market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Over":
                            try:
                                total_line = float(outcome.get("point"))
                            except (TypeError, ValueError):
                                pass

        if spread_line is not None or total_line is not None:
            line_rows.append({
                "year":            year,
                "game_date":       game_date,
                "team1":           home,
                "team2":           away,
                "spread_favorite": spread_favorite,
                "spread_line":     spread_line,
                "total_line":      total_line,
                "ats_result":      None,
                "ou_result":       None,
                "open_spread":     None,
                "open_total":      None,
                "source":          "odds_api",
            })

    # ── Upsert both tables ─────────────────────────────────────────────────────
    n_results = n_lines = 0

    if result_rows:
        df_results = pd.DataFrame(result_rows)
        upsert_df(df_results, "historical_results", if_exists="append")
        n_results = len(df_results)
        print(f"  [odds] Stored {n_results} game results")

    if line_rows:
        df_lines = pd.DataFrame(line_rows)
        upsert_df(df_lines, "historical_lines", if_exists="append")
        n_lines = len(df_lines)
        print(f"  [odds] Stored {n_lines} closing lines")

    return {"results": n_results, "lines": n_lines}


def fetch_and_store_scores(year: int, days_from: int = 3) -> dict:
    """
    Fetch completed tournament scores + closing lines from The Odds API
    and upsert into historical_results and historical_lines.

    Delegates to fetch_live_scores.ingest_live_scores which handles
    both the /scores and /odds endpoints cleanly.

    Returns dict: {"results": int, "lines": int}
    """
    from src.ingest.fetch_live_scores import ingest_live_scores
    summary = ingest_live_scores(year=year, days_from=days_from)
    return {
        "results": summary.get("results_inserted", 0),
        "lines":   summary.get("lines_inserted", 0),
    }