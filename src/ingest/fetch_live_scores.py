"""
Fetch live and recently completed NCAA Tournament scores + closing lines
from The Odds API for the current tournament year.

Uses two endpoints:
  - /v4/sports/{sport}/scores?daysFrom=3  → game results
  - /v4/sports/{sport}/odds               → current/closing spread + total

Results are upserted into historical_results and historical_lines so the
Results page can grade them exactly like past tournament years.

Cost: 2 credits (scores with daysFrom) + 2 credits (odds, spreads+totals) = 4
"""

import requests
from datetime import datetime, timezone

from src.utils.config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, TOURNAMENT_YEARS
from src.utils.db import db_conn, query_df
from src.utils.team_map import normalize_team_name, is_known_team

PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "bovada"]
SCORES_URL = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/scores"
ODDS_URL   = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"

# First Four is round 0 — use round_number 0 to distinguish from R64 (1)
# but map it to 1 for display/model purposes since it feeds into R64 slots.
ROUND_NAMES = {0: "First Four", 1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


def _norm(name: str) -> str:
    return normalize_team_name(name) if is_known_team(name) else name


def _parse_date(iso_str: str) -> str:
    """Convert ISO 8601 UTC timestamp → ET date string YYYY-MM-DD.

    Tournament games tip off in the afternoon/evening ET. A 9:45 PM ET game
    is past midnight UTC (next day), so we must convert to ET to get the
    correct local game date.
    ET = UTC-4 during DST (March–November).
    """
    try:
        from datetime import timedelta
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt_et = dt - timedelta(hours=4)  # UTC → ET (DST, March tournament)
        return dt_et.strftime("%Y-%m-%d")
    except Exception:
        return iso_str[:10]


def fetch_completed_scores(days_from: int = 3) -> list:
    """
    Call /scores?daysFrom={days_from} and return list of completed games.
    Cost: 2 API credits.
    """
    if not ODDS_API_KEY:
        print("[fetch_live_scores] No ODDS_API_KEY configured")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": days_from,
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(SCORES_URL, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"  [scores] quota: {remaining} remaining / {used} used")
        resp.raise_for_status()
        games = resp.json()
        completed = [g for g in games if g.get("completed") and g.get("scores")]
        print(f"  [scores] {len(games)} total, {len(completed)} completed with scores")
        return completed
    except Exception as e:
        print(f"  [scores] API error: {e}")
        return []


def fetch_current_odds() -> dict:
    """
    Call /odds for spreads+totals and return dict keyed by frozenset of
    normalized lower-case team names → {spread_home, total, home, away}.
    Cost: 2 API credits (spreads + totals, 1 region).
    """
    if not ODDS_API_KEY:
        return {}

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(ODDS_URL, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"  [odds]   quota: {remaining} remaining / {used} used")
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  [odds] API error: {e}")
        return {}

    result = {}
    for game in games:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = _norm(home_raw)
        away = _norm(away_raw)
        key = frozenset({home.lower(), away.lower()})

        # Find best bookmaker
        bk_map = {b["key"]: b for b in game.get("bookmakers", [])}
        bk = None
        for pref in PREFERRED_BOOKS:
            if pref in bk_map:
                bk = bk_map[pref]
                break
        if not bk and game.get("bookmakers"):
            bk = game["bookmakers"][0]
        if not bk:
            continue

        spread_home = total_line = None
        for market in bk.get("markets", []):
            if market["key"] == "spreads":
                for o in market.get("outcomes", []):
                    if o.get("name") == home_raw:
                        spread_home = o.get("point")
            elif market["key"] == "totals":
                for o in market.get("outcomes", []):
                    if o.get("name") == "Over":
                        total_line = o.get("point")

        result[key] = {
            "home": home,
            "away": away,
            "home_raw": home_raw,
            "away_raw": away_raw,
            "spread_home": spread_home,
            "total_line": total_line,
        }

    print(f"  [odds]   {len(result)} games with odds")
    return result


def _lookup_seeds(team1: str, team2: str, year: int) -> tuple:
    """Look up seeds from tournament_bracket table."""
    rows = query_df(
        "SELECT team, seed FROM tournament_bracket WHERE year = ? AND team IN (?, ?)",
        params=[year, team1, team2],
    )
    if rows.empty:
        return None, None
    seed_map = dict(zip(rows["team"], rows["seed"]))
    s1 = seed_map.get(team1) or seed_map.get(normalize_team_name(team1))
    s2 = seed_map.get(team2) or seed_map.get(normalize_team_name(team2))
    return s1, s2


def _infer_round(seed1, seed2) -> int:
    """
    Infer round number from seeds.
    First Four: both seeds equal (e.g. 11 vs 11, 16 vs 16).
    R64: seeds sum to 17 (1+16, 2+15, ... 8+9).
    Later rounds: any other combination.
    """
    if seed1 is None or seed2 is None:
        return 1
    s1, s2 = int(seed1), int(seed2)
    if s1 == s2:
        return 0  # First Four
    if s1 + s2 == 17:
        return 1  # R64
    return 2      # R32 or later — caller can refine


def ingest_live_scores(year: int = None, days_from: int = 3) -> dict:
    """
    Fetch completed scores + current odds, upsert into historical_results
    and historical_lines. Returns summary dict.

    This is safe to call repeatedly — upserts are idempotent.
    """
    if year is None:
        year = TOURNAMENT_YEARS[-1]

    print(f"\n=== Fetching live scores for {year} ===")

    completed = fetch_completed_scores(days_from=days_from)
    odds_map  = fetch_current_odds()

    # Build tournament team set to filter out NIT and other tournaments
    bracket_df = query_df(
        "SELECT team FROM tournament_bracket WHERE year = ?", params=[year]
    )
    tournament_teams = set()
    if not bracket_df.empty:
        for _, br in bracket_df.iterrows():
            tournament_teams.add(_norm(str(br["team"])).lower())

    results_inserted = 0
    lines_inserted   = 0
    skipped          = 0

    for game in completed:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home = _norm(home_raw)
        away = _norm(away_raw)
        game_date = _parse_date(game.get("commence_time", ""))

        # Skip non-NCAAT games (NIT, CBI, etc.) when bracket data is available
        if tournament_teams and (
            home.lower() not in tournament_teams
            and away.lower() not in tournament_teams
        ):
            continue

        # Parse scores
        scores = game.get("scores", []) or []
        score_map = {_norm(s["name"]): int(s["score"]) for s in scores if s.get("score")}
        score_home = score_map.get(home)
        score_away = score_map.get(away)
        if score_home is None or score_away is None:
            # Try raw names
            score_home = score_map.get(_norm(home_raw), score_map.get(home_raw))
            score_away = score_map.get(_norm(away_raw), score_map.get(away_raw))
        if score_home is None or score_away is None:
            print(f"    SKIP (no scores): {home_raw} vs {away_raw}")
            skipped += 1
            continue

        # Winner / margin
        winner = home if score_home > score_away else away
        margin = abs(score_home - score_away)
        total_pts = score_home + score_away

        # Seeds
        seed_home, seed_away = _lookup_seeds(home, away, year)

        # Round inference
        round_num = _infer_round(seed_home, seed_away)
        round_name = ROUND_NAMES.get(round_num, "R64")

        # For DB storage: team1 = winner convention matches historical scraper
        # Actually historical_results has no winner-first convention — just store home/away
        t1, t2 = home, away
        s1, s2 = score_home, score_away
        sd1, sd2 = seed_home, seed_away

        # ── Upsert into historical_results ────────────────────────────────────
        try:
            with db_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO historical_results
                       (year, round_number, round_name, game_date,
                        team1, team2, score1, score2,
                        winner, margin, total_points, seed1, seed2)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [year, round_num, round_name, game_date,
                     t1, t2, s1, s2,
                     winner, margin, total_pts, sd1, sd2],
                )
            results_inserted += 1
            print(f"    ✓ result: {t1} {s1} – {t2} {s2} ({game_date}, {round_name})")
        except Exception as e:
            print(f"    ✗ result insert error: {e}")

        # ── Match odds and upsert into historical_lines ────────────────────────
        odds_key = frozenset({home.lower(), away.lower()})
        odds = odds_map.get(odds_key)

        if odds:
            spread_home_val = odds["spread_home"]
            total_line_val  = odds["total_line"]

            # Convert to team1 (home) perspective in positive-favored convention
            # Odds API: negative = favored → negate to get positive = favored
            spread_t1 = -spread_home_val if spread_home_val is not None else None
            spread_fav = t1 if (spread_t1 is not None and spread_t1 > 0) else t2

            try:
                with db_conn() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO historical_lines
                           (year, game_date, team1, team2, spread_favorite,
                            spread_line, total_line, open_spread, open_total,
                            ats_result, ou_result, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)""",
                        [year, game_date, t1, t2, spread_fav,
                         spread_t1, total_line_val, "odds_api_live"],
                    )
                lines_inserted += 1
                print(f"    ✓ line:   spread {spread_t1:+.1f} · total {total_line_val}")
            except Exception as e:
                print(f"    ✗ line insert error: {e}")
        else:
            # Game may have already closed — try to get from odds_history DB snapshot
            existing_line = query_df(
                """SELECT oh.spread_home, oh.total_line, oh.home_team, oh.away_team
                   FROM odds_history oh
                   WHERE (oh.home_team = ? OR oh.away_team = ?)
                     AND (oh.home_team = ? OR oh.away_team = ?)
                   ORDER BY oh.pull_timestamp DESC LIMIT 1""",
                params=[t1, t1, t2, t2],
            )
            if not existing_line.empty:
                row = existing_line.iloc[0]
                stored_home = _norm(str(row["home_team"]))
                sh = float(row["spread_home"]) if row["spread_home"] is not None else None
                tl = float(row["total_line"]) if row["total_line"] is not None else None
                # Align perspective: if stored home != our t1, flip spread
                spread_t1 = -sh if (sh is not None and stored_home.lower() == t1.lower()) else sh
                spread_fav = t1 if (spread_t1 is not None and spread_t1 > 0) else t2
                try:
                    with db_conn() as conn:
                        conn.execute(
                            """INSERT OR REPLACE INTO historical_lines
                               (year, game_date, team1, team2, spread_favorite,
                                spread_line, total_line, open_spread, open_total,
                                ats_result, ou_result, source)
                               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)""",
                            [year, game_date, t1, t2, spread_fav,
                             spread_t1, tl, "odds_history_snapshot"],
                        )
                    lines_inserted += 1
                    print(f"    ✓ line (from snapshot): spread {spread_t1}")
                except Exception as e:
                    print(f"    ✗ snapshot line insert error: {e}")
            else:
                print(f"    — no line available: {t1} vs {t2}")

    summary = {
        "results_inserted": results_inserted,
        "lines_inserted":   lines_inserted,
        "skipped":          skipped,
        "total_completed":  len(completed),
    }
    print(f"\n  Summary: {results_inserted} results, {lines_inserted} lines, {skipped} skipped")
    return summary


ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}


def ingest_espn_results(year: int) -> int:
    """
    Fetch all completed NCAA Tournament results for `year` from ESPN scoreboard.
    Sweeps every date from First Four through Championship (Selection Sunday +3
    to +25 days) and stores completed games in historical_results.

    Returns number of new rows inserted.
    """
    import time
    from datetime import date, timedelta
    from src.utils.config import SELECTION_SUNDAY

    sel_str = SELECTION_SUNDAY.get(year)
    if not sel_str:
        print(f"  [espn] No SELECTION_SUNDAY config for {year}")
        return 0

    # Parse selection sunday: '0315' → date(year, 3, 15)
    month = int(sel_str[:2])
    day   = int(sel_str[2:])
    sel_date = date(year, month, day)

    # Tournament bracket: use to filter ESPN games to only NCAAT teams
    bracket_df = query_df(
        "SELECT team, seed FROM tournament_bracket WHERE year = ?",
        params=[year],
    )
    tournament_teams = set()
    seed_map = {}
    if not bracket_df.empty:
        for _, br in bracket_df.iterrows():
            norm = _norm(str(br["team"])).lower()
            tournament_teams.add(norm)
            seed_map[norm] = int(br["seed"])

    # First Four starts Thursday after Selection Sunday (~3 days later)
    # Championship is ~25 days after Selection Sunday
    start_date = sel_date + timedelta(days=3)
    end_date   = sel_date + timedelta(days=25)

    inserted = 0
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        try:
            resp = requests.get(
                ESPN_SCOREBOARD_URL,
                params={"dates": date_str},
                headers=ESPN_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [espn] {date_str} fetch error: {e}")
            current += timedelta(days=1)
            time.sleep(0.3)
            continue

        events = data.get("events", [])
        for event in events:
            try:
                status_name = (
                    event.get("status", {}).get("type", {}).get("name", "")
                )
                if status_name != "STATUS_FINAL":
                    continue

                competitions = event.get("competitions", [])
                if not competitions:
                    continue
                comp = competitions[0]
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home_comp = next(
                    (c for c in competitors if c.get("homeAway") == "home"),
                    competitors[0],
                )
                away_comp = next(
                    (c for c in competitors if c.get("homeAway") == "away"),
                    competitors[1],
                )

                home_raw = home_comp.get("team", {}).get("displayName", "")
                away_raw = away_comp.get("team", {}).get("displayName", "")
                home = _norm(home_raw)
                away = _norm(away_raw)

                # Only store NCAAT games
                if tournament_teams and (
                    home.lower() not in tournament_teams
                    and away.lower() not in tournament_teams
                ):
                    continue

                try:
                    score_home = int(home_comp.get("score", 0))
                    score_away = int(away_comp.get("score", 0))
                except Exception:
                    continue

                if score_home == 0 and score_away == 0:
                    continue

                winner = home if score_home > score_away else away
                margin = abs(score_home - score_away)
                total_pts = score_home + score_away

                seed_home = seed_map.get(home.lower())
                seed_away = seed_map.get(away.lower())
                round_num  = _infer_round(seed_home, seed_away)
                round_name = ROUND_NAMES.get(round_num, "R64")
                game_date  = current.isoformat()
                espn_id    = event.get("id", "")

                with db_conn() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO historical_results
                           (year, round_number, round_name, game_date,
                            team1, team2, score1, score2,
                            winner, margin, total_points, seed1, seed2, espn_game_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [year, round_num, round_name, game_date,
                         home, away, score_home, score_away,
                         winner, margin, total_pts, seed_home, seed_away, espn_id],
                    )
                inserted += 1
                print(f"  [espn] ✓ {home} {score_home}–{score_away} {away} ({game_date})")

            except Exception as e:
                print(f"  [espn] parse error on {date_str}: {e}")

        time.sleep(0.3)
        current += timedelta(days=1)

    print(f"  [espn] Inserted {inserted} results for {year}")
    return inserted
