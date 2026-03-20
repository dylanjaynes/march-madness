"""
One-time script to match historical_results rows to ESPN game IDs.

Algorithm:
- For each (year, game_date, team1, team2) in historical_results WHERE espn_game_id IS NULL
- Fetch ESPN scoreboard for that date: ?dates={YYYYMMDD}&groups=100
- Normalize ESPN team names with normalize_team_name()
- Match on frozenset of normalized team names
- On match: UPDATE historical_results SET espn_game_id = event_id
- Sleep 1.5s between date fetches
- Log unmatched games to console
"""

import time
import requests
from collections import defaultdict

from src.utils.db import db_conn, query_df
from src.utils.team_map import normalize_team_name, is_known_team

HEADERS = {"User-Agent": "Mozilla/5.0"}
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


def _fetch_espn_scoreboard(date_str: str) -> list[dict]:
    """
    Fetch ESPN scoreboard for a given date (YYYYMMDD string).
    Returns a list of event dicts, or empty list on failure.
    """
    params = {"dates": date_str, "groups": "100", "limit": "200"}
    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])
    except Exception as exc:
        print(f"  [ESPN fetch error] date={date_str}: {exc}")
        return []


def _build_date_game_map(events: list[dict]) -> dict[frozenset, str]:
    """
    Given a list of ESPN event dicts, return a mapping of
    frozenset({normalized_team1, normalized_team2}) -> espn_game_id.
    """
    game_map: dict[frozenset, str] = {}
    for event in events:
        event_id = event.get("id", "")
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        competitors = competitions[0].get("competitors", [])
        if len(competitors) < 2:
            continue
        names = set()
        for competitor in competitors:
            raw_name = (
                competitor.get("team", {}).get("displayName")
                or competitor.get("team", {}).get("name")
                or ""
            )
            normalized = normalize_team_name(raw_name)
            names.add(normalized)
        if len(names) == 2:
            game_map[frozenset(names)] = event_id
    return game_map


def run_enrichment() -> None:
    """
    Main entry point.  Iterates over historical_results rows that have no
    espn_game_id, groups them by game_date, fetches the ESPN scoreboard once
    per unique date, and writes matched IDs back to the DB.
    """
    sql = """
        SELECT year, game_date, team1, team2
        FROM historical_results
        WHERE espn_game_id IS NULL
        ORDER BY game_date
    """
    df = query_df(sql)
    if df.empty:
        print("No unmatched games found in historical_results.")
        return

    print(f"Found {len(df)} unmatched games across {df['game_date'].nunique()} dates.")

    # Group games by date so we only hit ESPN once per date
    # Skip rows with NULL game_date (can't query ESPN without a date)
    date_groups: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        if row["game_date"] is None or (hasattr(row["game_date"], "__class__") and str(row["game_date"]) == "nan"):
            continue
        date_groups[str(row["game_date"])].append(row.to_dict())

    total_matched = 0
    total_unmatched = 0

    for game_date, games in sorted(date_groups.items()):
        # ESPN wants YYYYMMDD without dashes
        date_nodash = game_date.replace("-", "")
        events = _fetch_espn_scoreboard(date_nodash)
        game_map = _build_date_game_map(events)

        for game in games:
            team1_norm = normalize_team_name(game["team1"])
            team2_norm = normalize_team_name(game["team2"])
            key = frozenset({team1_norm, team2_norm})

            if key in game_map:
                espn_id = game_map[key]
                try:
                    with db_conn() as conn:
                        conn.execute(
                            """
                            UPDATE historical_results
                            SET espn_game_id = ?
                            WHERE year = ? AND game_date = ? AND team1 = ? AND team2 = ?
                            """,
                            [espn_id, game["year"], game["game_date"], game["team1"], game["team2"]],
                        )
                    total_matched += 1
                except Exception as exc:
                    print(
                        f"  [DB write error] {game['team1']} vs {game['team2']} "
                        f"on {game['game_date']}: {exc}"
                    )
            else:
                print(
                    f"  [UNMATCHED] {game['year']} | {game['game_date']} | "
                    f"{game['team1']} vs {game['team2']} "
                    f"(normalized: {team1_norm!r} vs {team2_norm!r})"
                )
                total_unmatched += 1

        time.sleep(1.5)

    print(
        f"\nEnrichment complete. Matched: {total_matched}, Unmatched: {total_unmatched}"
    )


if __name__ == "__main__":
    run_enrichment()
