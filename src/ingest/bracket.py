"""
Bracket ingestion — fetches the official NCAA tournament bracket from
Sports Reference and stores it in the tournament_bracket table.

Works as soon as Sports Reference publishes the bracket (typically within
hours of Selection Sunday). Also supports fetching projected brackets
from ESPN Bracketology (Joe Lunardi) before the official announcement.
"""
import json
import re
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

    Tries two URL formats — Sports Reference added a /men/ path in recent years.
    """
    # Try new URL format first (/men/ subdirectory), then fall back to legacy
    urls = [
        f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html",
        f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html",
    ]

    resp = None
    for url in urls:
        print(f"  [bracket] Trying {url}")
        time.sleep(1.5)  # be polite
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                resp = r
                print(f"  [bracket] Got 200 from {url}")
                break
            else:
                print(f"  [bracket] Got {r.status_code} from {url}, trying next...")
        except Exception as e:
            print(f"  [bracket] Error fetching {url}: {e}")

    if resp is None:
        print("  [bracket] All URLs failed")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    # Debug: check what region divs exist on the page
    found_ids = [d.get("id") for d in soup.find_all("div", id=True) if d.get("id") in SR_REGION_MAP]
    print(f"  [bracket] Region divs found: {found_ids}")
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


# ── Projected bracket (CBS Sports Bracketology — Jerry Palm) ──────────────────

_PROJ_REGIONS = {"East", "West", "South", "Midwest"}

# Short-name → canonical normalizations for CBS Sports team names
_CBS_NAME_FIXES = {
    "UConn": "Connecticut",
    "UCONN": "Connecticut",
    "Ole Miss": "Mississippi",
    "USC": "Southern California",
    "UNLV": "Nevada-Las Vegas",
    "UAB": "Alabama-Birmingham",
    "UTEP": "Texas-El Paso",
    "UTSA": "Texas-San Antonio",
    "SMU": "Southern Methodist",
    "LSU": "Louisiana State",
    "TCU": "Texas Christian",
    "UCF": "Central Florida",
    "BYU": "Brigham Young",
    "VCU": "Virginia Commonwealth",
    "UNC": "North Carolina",
    "UNC Wilmington": "North Carolina-Wilmington",
    "NC State": "North Carolina State",
    "UNCW": "North Carolina-Wilmington",
    "St. John's": "St. John's (NY)",
    "St. John's (N.Y.)": "St. John's (NY)",
    "Miami (FL)": "Miami (Florida)",
    "Miami": "Miami (Florida)",
    "Pitt": "Pittsburgh",
    "FGCU": "Florida Gulf Coast",
    "FIU": "Florida International",
    "FAU": "Florida Atlantic",
    "FDU": "Fairleigh Dickinson",
    "ETSU": "East Tennessee State",
    "SIUE": "SIU Edwardsville",
    "LIU": "Long Island University",
    "LMU": "Loyola Marymount",
    "UMass": "Massachusetts",
    "UIC": "Illinois-Chicago",
    "UAlbany": "Albany",
    "UMBC": "Maryland-Baltimore County",
    "UNC Asheville": "North Carolina-Asheville",
    "UNC Greensboro": "North Carolina-Greensboro",
    "UMKC": "Kansas City",
    "UT Martin": "Tennessee-Martin",
    "Michigan St.": "Michigan State",
    "Iowa St.": "Iowa State",
    "Kansas St.": "Kansas State",
    "Mississippi St.": "Mississippi State",
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Oklahoma St.": "Oklahoma State",
    "Arizona St.": "Arizona State",
    "Florida St.": "Florida State",
    "Oregon St.": "Oregon State",
    "Colorado St.": "Colorado State",
    "Fresno St.": "Fresno State",
    "Utah St.": "Utah State",
    "Boise St.": "Boise State",
    "San Diego St.": "San Diego State",
    "Sacramento St.": "Sacramento State",
    "N. Carolina": "North Carolina",
    "N. Iowa": "Northern Iowa",
    "N. Dakota St.": "North Dakota State",
    "S. Dakota St.": "South Dakota State",
    "SE Louisiana": "Southeastern Louisiana",
    "E. Washington": "Eastern Washington",
    "W. Kentucky": "Western Kentucky",
    "S. Illinois": "Southern Illinois",
    "Mid. Tennessee": "Middle Tennessee",
    "McNeese": "McNeese State",
    "Kennesaw St.": "Kennesaw State",
}


def _normalize_cbs_name(raw: str) -> str:
    """Normalize a CBS Sports short team name to the canonical DB name."""
    raw = raw.strip()
    if raw in _CBS_NAME_FIXES:
        return _CBS_NAME_FIXES[raw]
    if is_known_team(raw):
        return normalize_team_name(raw)
    return raw


def fetch_projected_bracket_cbs() -> dict:
    """
    Fetch Jerry Palm's CBS Sports Bracketology projected bracket.

    CBS Sports renders the bracket server-side with clear CSS classes:
      .bracket-table-wrapper > .bracket-table-left/.bracket-table-right
        > .region-title  (East / West / South / Midwest)
        > .bracket-row > .bracket-row-seed + .full-width (seed + team)

    Returns {"East": {1: "Duke", ...}, ...} or {} on failure.
    """
    url = "https://www.cbssports.com/college-basketball/bracketology/"
    print(f"  [cbs_bracket] Fetching {url}")
    time.sleep(1.0)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [cbs_bracket] Fetch failed: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    bracket = {}

    for wrapper in soup.find_all(class_="bracket-table-wrapper"):
        for side in ("bracket-table-left", "bracket-table-right"):
            half = wrapper.find(class_=side)
            if not half:
                continue
            region_el = half.find(class_="region-title")
            region = region_el.text.strip() if region_el else None
            if not region or region not in _PROJ_REGIONS:
                continue

            seed_team: dict[int, str] = {}
            for row in half.find_all(class_="bracket-row"):
                seed_el = row.find(class_="bracket-row-seed")
                name_el = row.find(class_="full-width")
                if not seed_el or not name_el:
                    continue
                try:
                    seed = int(seed_el.text.strip())
                    name = name_el.text.strip()
                    if 1 <= seed <= 16 and name:
                        seed_team[seed] = _normalize_cbs_name(name)
                except (ValueError, AttributeError):
                    pass
            if seed_team:
                bracket[region] = seed_team

    n = sum(len(v) for v in bracket.values())
    print(f"  [cbs_bracket] Parsed {n} teams across {len(bracket)} regions")
    return bracket


def fetch_and_store_projected_bracket_espn(year: int) -> dict:
    """Compatibility shim: now fetches CBS Sports bracketology, stores in DB, returns dict."""
    bracket = fetch_projected_bracket_cbs()
    if bracket:
        store_bracket(year, bracket)
    return bracket


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
