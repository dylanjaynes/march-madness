"""
Bracket ingestion — fetches the official NCAA tournament bracket from
Sports Reference and stores it in the tournament_bracket table.

Works as soon as Sports Reference publishes the bracket (typically within
hours of Selection Sunday). Also supports fetching projected brackets
from BartTorvik before the official announcement.
"""
import time
import requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup

from src.utils.db import db_conn, query_df
from src.utils.team_map import normalize_team_name, is_known_team

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Sports Reference region IDs → display names
SR_REGION_MAP = {
    "east":    "East",
    "west":    "West",
    "south":   "South",
    "midwest": "Midwest",
}


def fetch_bracket(year: int) -> dict:
    """
    Scrape the R64 bracket (seeds + teams per region) from Sports Reference.
    Returns dict: {"East": {1: "Duke", 16: "Baylor", ...}, "West": {...}, ...}
    Returns empty dict if page not yet available (pre-bracket-release).
    """
    url = f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
    print(f"  [bracket] Fetching {year} bracket from {url}")
    time.sleep(1.5)  # be polite

    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [bracket] Failed to fetch: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    bracket = {}

    for region_id, region_name in SR_REGION_MAP.items():
        region_div = soup.find("div", id=region_id)
        if not region_div:
            continue

        # First round div = R64
        rounds = region_div.find_all("div", class_="round", recursive=False)
        if not rounds:
            bracket_div = region_div.find("div", id="bracket")
            if bracket_div:
                rounds = bracket_div.find_all("div", class_="round", recursive=False)

        if not rounds:
            continue

        r64_div = rounds[0]  # first round = R64
        seed_team: dict[int, str] = {}

        for game_div in r64_div.find_all("div", recursive=False):
            team_divs = game_div.find_all("div", recursive=False)
            for td in team_divs[:2]:
                seed, team = _parse_seed_team(td)
                if seed and team:
                    seed_team[seed] = team

        if seed_team:
            bracket[region_name] = seed_team

    n = sum(len(v) for v in bracket.values())
    print(f"  [bracket] Parsed {n} teams across {len(bracket)} regions")
    return bracket


def _parse_seed_team(td) -> tuple:
    """Extract (seed, team_name) from a team div."""
    seed = None
    seed_span = td.find("span")
    if seed_span:
        try:
            seed = int(seed_span.text.strip())
        except (ValueError, AttributeError):
            pass

    name_link = td.find("a", href=lambda h: h and "/cbb/schools/" in h)
    if not name_link:
        return None, None

    raw_name = name_link.text.strip()
    team = normalize_team_name(raw_name) if is_known_team(raw_name) else raw_name
    return seed, team


def store_bracket(year: int, bracket: dict) -> int:
    """
    Store bracket dict in tournament_bracket table.
    Returns number of rows stored.
    """
    if not bracket:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for region, seed_team in bracket.items():
        for seed, team in seed_team.items():
            rows.append((year, region, seed, team, now))

    with db_conn() as conn:
        conn.execute("DELETE FROM tournament_bracket WHERE year = ?", [year])
        conn.executemany(
            "INSERT INTO tournament_bracket (year, region, seed, team, fetched_at) VALUES (?,?,?,?,?)",
            rows,
        )

    print(f"  [bracket] Stored {len(rows)} teams for {year}")
    return len(rows)


def fetch_and_store_bracket(year: int) -> dict:
    """Convenience: fetch + store + return bracket dict."""
    bracket = fetch_bracket(year)
    if bracket:
        store_bracket(year, bracket)
    return bracket


def load_bracket_from_db(year: int) -> dict:
    """
    Load stored bracket from DB.
    Returns {"East": {1: "Duke", ...}, ...} or empty dict if not stored.
    """
    df = query_df(
        "SELECT region, seed, team FROM tournament_bracket WHERE year = ? ORDER BY region, seed",
        params=[year],
    )
    if df.empty:
        return {}

    bracket = {}
    for _, row in df.iterrows():
        region = row["region"]
        if region not in bracket:
            bracket[region] = {}
        bracket[region][int(row["seed"])] = row["team"]

    return bracket


def get_bracket_status(year: int) -> dict:
    """Return metadata about the stored bracket."""
    df = query_df(
        "SELECT COUNT(*) as n, MAX(fetched_at) as last_fetch FROM tournament_bracket WHERE year = ?",
        params=[year],
    )
    if df.empty or df.iloc[0]["n"] == 0:
        return {"stored": False, "n_teams": 0, "fetched_at": None}
    return {
        "stored": True,
        "n_teams": int(df.iloc[0]["n"]),
        "fetched_at": df.iloc[0]["last_fetch"],
    }
