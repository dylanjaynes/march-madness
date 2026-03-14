"""
OddsPortal historical odds scraper for NCAA Tournament 2022–2025.

Primary: scrape oddsportal.com for March Madness game lines.
Fallback: read data/raw/manual_lines_2022_2025.csv (user-filled from covers.com).

OddsPortal structure:
  /basketball/usa/ncaa/results/#/page/{N}/
  Each result row contains: teams, date, score, odds (1x2 / handicap).
  We need handicap (spread) and O/U (totals) markets.
"""

import re
import time
import warnings
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

from src.utils.config import RAW_DIR, TOURNAMENT_YEARS
from src.utils.db import db_conn, upsert_df, query_df
from src.utils.team_map import normalize_team_name, is_known_team

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.oddsportal.com/",
}

MANUAL_CSV_PATH = RAW_DIR / "manual_lines_2022_2025.csv"

# Tournament years where SBRO has no data
ODDSPORTAL_YEARS = [2022, 2023, 2024, 2025]

# Approx date ranges for each tournament
TOURN_DATES = {
    2022: ("2022-03-15", "2022-04-05"),
    2023: ("2023-03-14", "2023-04-03"),
    2024: ("2024-03-19", "2024-04-08"),
    2025: ("2025-03-18", "2025-04-07"),
}


def _norm(name: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalize_team_name(name) if is_known_team(name) else name


# ── Manual CSV fallback ───────────────────────────────────────────────────────

def create_manual_csv_template():
    """Create a blank template CSV for the user to fill in from covers.com."""
    if MANUAL_CSV_PATH.exists():
        print(f"  [oddsportal] Manual CSV already exists: {MANUAL_CSV_PATH}")
        return
    cols = [
        "year", "game_date", "team1", "team2",
        "spread_line", "open_spread",
        "total_line", "open_total",
        "team1_score", "team2_score",
        "spread_favorite", "notes"
    ]
    template = pd.DataFrame(columns=cols)
    # Add a sample row to guide the user
    template.loc[0] = {
        "year": 2023,
        "game_date": "2023-03-16",
        "team1": "Alabama",
        "team2": "Texas A&M-CC",
        "spread_line": -19.5,
        "open_spread": -20.0,
        "total_line": 143.5,
        "open_total": 144.0,
        "team1_score": 96,
        "team2_score": 75,
        "spread_favorite": "Alabama",
        "notes": "R64"
    }
    template.to_csv(MANUAL_CSV_PATH, index=False)
    print(f"  [oddsportal] Created manual CSV template: {MANUAL_CSV_PATH}")
    print(f"  Fill this in from covers.com/ncaab/matchups for 2022–2025 tournament games.")


def load_manual_csv(year: int) -> pd.DataFrame:
    """Load lines from the manually-filled CSV for a given year (only rows with actual line data)."""
    if not MANUAL_CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(MANUAL_CSV_PATH)
    df = df[df["year"] == year].copy()
    # Only return rows that actually have lines filled in
    has_data = df["spread_line"].notna() | df["total_line"].notna()
    return df[has_data].copy()


# ── OddsPortal scraper ────────────────────────────────────────────────────────

def _get_tournament_team_names(year: int) -> set:
    """Get set of known tournament team names for cross-referencing."""
    rows = query_df(
        "SELECT team1, team2 FROM historical_results WHERE year = ?",
        params=[year]
    )
    names = set()
    for _, r in rows.iterrows():
        names.add(_norm(str(r["team1"])).lower())
        names.add(_norm(str(r["team2"])).lower())
    return names


def scrape_oddsportal_year(year: int) -> pd.DataFrame:
    """
    Scrape OddsPortal for NCAA Tournament lines for a given year.
    Returns DataFrame in historical_lines format.
    """
    start_date, end_date = TOURN_DATES.get(year, ("", ""))
    tourn_teams = _get_tournament_team_names(year)

    # OddsPortal results page — paginate until outside date range
    base_url = "https://www.oddsportal.com/basketball/usa/ncaa/results/"
    session = requests.Session()
    session.headers.update(HEADERS)

    games = []
    page = 1
    consecutive_misses = 0

    while page <= 20 and consecutive_misses < 3:
        url = f"{base_url}#/page/{page}/" if page > 1 else base_url
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [oddsportal] Page {page} error: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.find_all("tr", class_=lambda c: c and "deactivate" in c)

        if not rows:
            # Try alternate row selector
            rows = soup.find_all("tr", {"class": re.compile(r"odd|even")})

        if not rows:
            print(f"  [oddsportal] No rows found on page {page}")
            break

        found_in_range = False
        for row in rows:
            game = _parse_oddsportal_row(row, year, start_date, end_date, tourn_teams)
            if game:
                games.append(game)
                found_in_range = True

        if not found_in_range:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        page += 1
        time.sleep(2)

    if not games:
        print(f"  [oddsportal] No games scraped for {year}")
        return pd.DataFrame()

    print(f"  [oddsportal] Scraped {len(games)} tournament games for {year}")
    return pd.DataFrame(games)


def _parse_oddsportal_row(row, year: int, start_date: str, end_date: str,
                          tourn_teams: set) -> dict:
    """Parse a single OddsPortal result row."""
    try:
        cells = row.find_all("td")
        if len(cells) < 4:
            return None

        # Date cell
        date_cell = row.find("td", class_=lambda c: c and "datet" in str(c).lower())
        if not date_cell:
            return None
        date_str = date_cell.get_text(strip=True)

        # Teams cell
        teams_link = row.find("a", href=lambda h: h and "/basketball/" in str(h))
        if not teams_link:
            return None
        teams_text = teams_link.get_text(strip=True)
        if " - " not in teams_text:
            return None
        parts = teams_text.split(" - ", 1)
        team1_raw, team2_raw = parts[0].strip(), parts[1].strip()
        team1 = _norm(team1_raw)
        team2 = _norm(team2_raw)

        # Cross-reference against tournament teams
        if (team1.lower() not in tourn_teams and team2.lower() not in tourn_teams):
            return None

        # Score cell
        score_cell = row.find("td", class_=lambda c: c and "score" in str(c).lower())
        score1 = score2 = None
        if score_cell:
            score_text = score_cell.get_text(strip=True)
            score_match = re.search(r"(\d+):(\d+)", score_text)
            if score_match:
                score1 = int(score_match.group(1))
                score2 = int(score_match.group(2))

        # Odds cells (handicap/spread and total)
        # OddsPortal shows multiple odds columns depending on market
        odds_cells = [c.get_text(strip=True) for c in cells[3:]]
        spread = total = None
        for val in odds_cells:
            numeric = re.sub(r"[^\d.+-]", "", val)
            try:
                f = float(numeric)
                if 100 <= abs(f) <= 200:
                    total = f  # totals are ~130-160 for NCAAB
                elif abs(f) <= 35:
                    spread = f
            except ValueError:
                pass

        return {
            "year": year,
            "game_date": _parse_oddsportal_date(date_str, year),
            "team1": team1,
            "team2": team2,
            "spread_line": spread,
            "open_spread": None,
            "total_line": total,
            "open_total": None,
            "team1_score": score1,
            "team2_score": score2,
            "spread_favorite": team1 if (spread and spread < 0) else (team2 if (spread and spread > 0) else None),
            "source": "oddsportal",
        }
    except Exception:
        return None


def _parse_oddsportal_date(date_str: str, year: int) -> str:
    """Parse OddsPortal date string to YYYY-MM-DD."""
    # Format varies: "21 Mar 2024", "21/03/2024", etc.
    for fmt in ["%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"]:
        try:
            from datetime import datetime
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return f"{year}-03-20"  # fallback


# ── Main ingestion functions ──────────────────────────────────────────────────

def ingest_oddsportal_year(year: int) -> pd.DataFrame:
    """
    Try OddsPortal first, fall back to manual CSV, then create template.
    Covers all years (not just 2022+) — SBRO/manual CSV both feed through here.
    """
    # First check manual CSV (user may have pre-filled it)
    manual_df = load_manual_csv(year)
    if not manual_df.empty:
        print(f"  [oddsportal] Loaded {len(manual_df)} games from manual CSV for {year}")
        lines_df = _manual_to_lines_format(manual_df, year)
        _store_lines(lines_df, year, "manual_csv")
        return lines_df

    # Try scraping OddsPortal
    print(f"  [oddsportal] Attempting OddsPortal scrape for {year}...")
    df = scrape_oddsportal_year(year)
    if not df.empty:
        lines_df = _scrape_to_lines_format(df, year)
        _store_lines(lines_df, year, "oddsportal")
        return lines_df

    # Fall back: create template for user to fill in
    create_manual_csv_template()
    print(f"  [oddsportal] Could not get lines for {year}. Please fill in: {MANUAL_CSV_PATH}")
    return pd.DataFrame()


def _manual_to_lines_format(df: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for _, g in df.iterrows():
        s1 = g.get("team1_score")
        s2 = g.get("team2_score")
        spread = g.get("spread_line")
        total = g.get("total_line")

        ats = ou = None
        if pd.notna(s1) and pd.notna(s2) and pd.notna(spread):
            margin = float(s1) - float(s2)
            cover = margin - abs(float(spread)) if float(spread) < 0 else margin + float(spread)
            ats = "WIN" if cover > 0 else ("LOSS" if cover < 0 else "PUSH")
        if pd.notna(s1) and pd.notna(s2) and pd.notna(total):
            actual = float(s1) + float(s2)
            ou = "OVER" if actual > float(total) else ("UNDER" if actual < float(total) else "PUSH")

        rows.append({
            "year": year,
            "game_date": str(g.get("game_date", "")),
            "team1": _norm(str(g.get("team1", ""))),
            "team2": _norm(str(g.get("team2", ""))),
            "spread_favorite": str(g.get("spread_favorite", "")),
            "spread_line": float(spread) if pd.notna(spread) else None,
            "total_line": float(total) if pd.notna(total) else None,
            "open_spread": float(g["open_spread"]) if pd.notna(g.get("open_spread")) else None,
            "open_total": float(g["open_total"]) if pd.notna(g.get("open_total")) else None,
            "ats_result": ats,
            "ou_result": ou,
            "source": "manual_csv",
        })
    return pd.DataFrame(rows)


def _scrape_to_lines_format(df: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for _, g in df.iterrows():
        rows.append({
            "year": year,
            "game_date": str(g.get("game_date", "")),
            "team1": str(g.get("team1", "")),
            "team2": str(g.get("team2", "")),
            "spread_favorite": str(g.get("spread_favorite", "")),
            "spread_line": g.get("spread_line"),
            "total_line": g.get("total_line"),
            "open_spread": g.get("open_spread"),
            "open_total": g.get("open_total"),
            "ats_result": None,
            "ou_result": None,
            "source": "oddsportal",
        })
    return pd.DataFrame(rows)


def _store_lines(df: pd.DataFrame, year: int, source: str):
    if df.empty:
        return
    with db_conn() as conn:
        conn.execute(
            "DELETE FROM historical_lines WHERE year = ? AND source = ?",
            [year, source]
        )
    upsert_df(df, "historical_lines", if_exists="append")
    print(f"  [oddsportal] Stored {len(df)} lines for {year} (source={source})")


def ingest_all_oddsportal() -> dict:
    """Ingest lines for all tournament years from manual CSV / OddsPortal."""
    from src.utils.config import TOURNAMENT_YEARS
    results = {}
    for year in TOURNAMENT_YEARS:
        print(f"\n[oddsportal] Processing {year}...")
        df = ingest_oddsportal_year(year)
        results[year] = len(df) if not df.empty else 0
    return results
