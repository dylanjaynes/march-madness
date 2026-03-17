"""
NIT game results (ESPN API) and betting lines (Odds API historical) ingestion.
"""

import time
import requests
import pandas as pd
from datetime import date, timedelta

from src.utils.config import ODDS_API_KEY, SELECTION_SUNDAY
from src.utils.db import db_conn, query_df
from src.utils.team_map import normalize_team_name, is_known_team

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}

NIT_DATE_RANGES = {
    2019: ("2019-03-19", "2019-04-04"),
    2021: ("2021-03-18", "2021-04-01"),
    2022: ("2022-03-15", "2022-04-02"),
    2023: ("2023-03-14", "2023-03-31"),
    2024: ("2024-03-19", "2024-04-06"),
    2025: ("2025-03-19", "2025-04-04"),
}


def _norm(name: str) -> str:
    return normalize_team_name(name) if is_known_team(name) else name


def _date_range(start: str, end: str):
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def _parse_round(note: str) -> tuple:
    """Returns (round_number, round_name) from ESPN note string."""
    n = note.lower()
    if "1st round" in n or "first round" in n:
        return 1, "R32"
    if "2nd round" in n or "second round" in n:
        return 2, "R16"
    if "quarterfinal" in n:
        return 3, "QF"
    if "semifinal" in n:
        return 4, "SF"
    if "championship" in n or "final" in n:
        return 5, "Champ"
    return 1, "R32"


def fetch_nit_games_espn(date_str: str) -> list:
    """
    Fetch completed NIT games from ESPN API for a given date (YYYYMMDD format).
    Returns list of dicts with team1, team2, score1, score2, game_date,
    round_number, round_name.
    """
    try:
        resp = requests.get(
            ESPN_SCOREBOARD,
            params={"groups": "50", "dates": date_str, "limit": "100"},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [nit] ESPN error {date_str}: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        # Only NIT games
        note_text = " ".join(n.get("headline", "") for n in comp.get("notes", []))
        if "NIT" not in note_text:
            continue

        # Only completed games
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        try:
            h_score = int(home.get("score", 0))
            a_score = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue

        # Skip games that haven't actually been played (both scores 0)
        if h_score == 0 and a_score == 0:
            continue

        round_num, round_name = _parse_round(note_text)
        game_date_str = event.get("date", "")[:10]

        games.append({
            "team1":        _norm(home["team"]["displayName"]),
            "team2":        _norm(away["team"]["displayName"]),
            "score1":       h_score,
            "score2":       a_score,
            "game_date":    game_date_str,
            "round_number": round_num,
            "round_name":   round_name,
        })
    return games


def ingest_nit_results(years=None) -> dict:
    """Scrape NIT results from ESPN and store in nit_results table."""
    targets = years or list(NIT_DATE_RANGES.keys())
    totals = {}

    for year in targets:
        if year not in NIT_DATE_RANGES:
            continue
        start, end = NIT_DATE_RANGES[year]
        print(f"\n=== NIT {year}: scanning {start} to {end} ===")

        year_games = []
        for date_str in _date_range(start, end):
            games = fetch_nit_games_espn(date_str)
            if games:
                print(f"  {date_str}: {len(games)} NIT games")
                year_games.extend(games)
            time.sleep(0.3)

        # Deduplicate (same game sometimes appears across multiple dates)
        seen = set()
        deduped = []
        for g in year_games:
            key = (g["team1"], g["team2"], g["game_date"])
            if key not in seen:
                seen.add(key)
                deduped.append(g)

        print(f"  Total {year}: {len(deduped)} unique games")
        stored = 0
        with db_conn() as conn:
            for g in deduped:
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO nit_results
                           (year, game_date, team1, team2, score1, score2, round_number, round_name)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        [year, g["game_date"], g["team1"], g["team2"],
                         g["score1"], g["score2"], g["round_number"], g["round_name"]],
                    )
                    stored += 1
                except Exception as e:
                    print(f"    DB error: {e}")
        totals[year] = stored
        print(f"  Stored {stored} games for {year}")

    return totals


def ingest_nit_lines(years=None) -> dict:
    """
    Pull NIT betting lines from Odds API historical endpoint.
    fetch_odds_snapshot() returns ALL NCAAB games including NIT.
    Match against nit_results and store in nit_lines.
    """
    if not ODDS_API_KEY:
        print("[nit] ODDS_API_KEY not set — skipping lines ingest")
        return {}

    from src.ingest.odds_historical import fetch_odds_snapshot, _build_lines_map, _norm as _onorm

    def _fuzzy(t1, t2, lmap):
        for key, line in lmap.items():
            api = list(key)
            if any(t1 in a or a in t1 for a in api) and any(t2 in a or a in t2 for a in api):
                return line
        return None

    targets = years or list(NIT_DATE_RANGES.keys())
    totals = {}

    for year in targets:
        results = query_df(
            "SELECT game_date, team1, team2 FROM nit_results WHERE year = ? ORDER BY game_date",
            params=[year],
        )
        if results.empty:
            print(f"  No NIT results for {year} — run ingest_nit_results() first")
            totals[year] = 0
            continue

        unique_dates = sorted(results["game_date"].dropna().unique())
        print(f"\n=== NIT lines {year}: {len(unique_dates)} dates ===")

        stored = 0
        for game_date in unique_dates:
            date_str = f"{game_date}T14:00:00Z"
            games_api = fetch_odds_snapshot(date_str)
            if not games_api:
                time.sleep(1)
                continue

            lines_map = _build_lines_map(games_api)
            day_games = results[results["game_date"] == game_date]

            for _, row in day_games.iterrows():
                t1_raw, t2_raw = row["team1"], row["team2"]
                t1 = _onorm(t1_raw).lower()
                t2 = _onorm(t2_raw).lower()

                line = lines_map.get(frozenset({t1, t2})) or _fuzzy(t1, t2, lines_map)
                if line is None:
                    print(f"    NO MATCH: {t1_raw} vs {t2_raw}")
                    continue

                # Convert to team1-perspective model convention
                home_n = line["home"].lower()
                raw = line["spread_home"] if home_n == t1 else line.get("spread_away")
                if raw is None:
                    raw = line["spread_home"]

                # Negate: API -8 favored → stored +8 positive = team1 favored
                spread_t1 = -raw if raw is not None else None
                fav = t1_raw if (spread_t1 is not None and spread_t1 > 0) else t2_raw

                with db_conn() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO nit_lines
                           (year, game_date, team1, team2, spread_line, total_line, spread_favorite, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        [year, game_date, t1_raw, t2_raw, spread_t1,
                         line.get("total"), fav, "odds_api_historical"],
                    )
                    stored += 1
            time.sleep(0.5)

        totals[year] = stored
        print(f"  Stored {stored} lines for {year}")
    return totals


def get_nit_coverage_report():
    """Print per-year games vs lines coverage."""
    print("\n=== NIT Coverage Report ===")
    r = query_df("SELECT year, COUNT(*) as games FROM nit_results GROUP BY year ORDER BY year")
    l = query_df("SELECT year, COUNT(*) as lines FROM nit_lines GROUP BY year ORDER BY year")
    if r.empty:
        print("  No NIT data yet")
        return None
    m = r.merge(l, on="year", how="left").fillna(0)
    m["lines"] = m["lines"].astype(int)
    m["pct"] = (m["lines"] / m["games"] * 100).round(1)
    print(m.to_string(index=False))
    return m
