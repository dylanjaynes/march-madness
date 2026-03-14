"""
SportsBookReviewsOnline (SBRO) historical odds parser.
Implements the paired-row format described in sbro_parser_instructions.md.

File coverage:
  2018-19 season → tournament year 2019
  2019-20 season → SKIP (cancelled)
  2020-21 season → tournament year 2021
  2021-22 season → tournament year 2022 (HTML table on SBRO site)

For years 2022–2025 where SBRO has no files, see oddsportal.py.
"""

import io
import re
import time
import warnings
import requests
import pandas as pd
from pathlib import Path

from src.utils.config import RAW_DIR
from src.utils.db import db_conn, upsert_df, query_df
from src.utils.team_map import normalize_team_name, is_known_team

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Known SBRO file locations (local path → year)
SBRO_FILES = {
    2019: [
        RAW_DIR / "ncaa-basketball-2018-19.xlsx",
        RAW_DIR / "ncaabasketball201819.xlsx",
        RAW_DIR / "sbro_2019.xlsx",
    ],
    2021: [
        RAW_DIR / "ncaa-basketball-2020-21.xlsx",
        RAW_DIR / "ncaabasketball202021.xlsx",
        RAW_DIR / "sbro_2021.xlsx",
    ],
    2022: [
        RAW_DIR / "ncaa-basketball-2021-22.xlsx",
        RAW_DIR / "ncaabasketball202122.xlsx",
        RAW_DIR / "sbro_2022.xlsx",
    ],
}

# SBRO download URL candidates (tried in order; first xlsx response wins)
SBRO_DOWNLOAD_URLS = {
    2019: [
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/ncaa-basketball-2018-19.xlsx",
        "https://www.sportsbookreviewsonline.com/wp-content/uploads/ncaa-basketball-2018-19.xlsx",
    ],
    2021: [
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/ncaa-basketball-2020-21.xlsx",
        "https://www.sportsbookreviewsonline.com/wp-content/uploads/ncaa-basketball-2020-21.xlsx",
    ],
    2022: [
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/ncaa-basketball-2021-22.xlsx",
    ],
}

# NCAA Tournament date range (MMDD): First Four through Championship
TOURN_DATE_MIN = "0313"
TOURN_DATE_MAX = "0410"


# ── Core parser (implements spec exactly) ────────────────────────────────────

def parse_sbro_file(filepath_or_bytes, year: int) -> pd.DataFrame:
    """
    Parse an SBRO Excel file using the paired-row format from the spec.
    Returns one row per game with visitor, home, scores, spread, total.
    """
    if isinstance(filepath_or_bytes, (str, Path)):
        df = pd.read_excel(str(filepath_or_bytes))
    else:
        df = pd.read_excel(io.BytesIO(filepath_or_bytes))

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Handle 'pk' spread values per spec
    for col in ["Open", "Close"]:
        if col in df.columns:
            df[col] = df[col].replace("pk", 0.0)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    games = []
    i = 0
    while i < len(df) - 1:
        row_v = df.iloc[i]
        row_h = df.iloc[i + 1]

        vh_v = str(row_v.get("VH", "")).strip().upper()
        vh_h = str(row_h.get("VH", "")).strip().upper()

        if vh_v in ("V", "N") and vh_h in ("H", "N"):
            v_open = pd.to_numeric(row_v.get("Open"), errors="coerce")
            h_open = pd.to_numeric(row_h.get("Open"), errors="coerce")

            # Disambiguation: total is always the large number (>80), spread is small (<40)
            # Use magnitude for both Open and Close independently — some SBRO rows
            # have total on V-side, others on H-side, and Close assignment varies too.
            v_close = pd.to_numeric(row_v.get("Close"), errors="coerce")
            h_close = pd.to_numeric(row_h.get("Close"), errors="coerce")

            # Determine which Open is total vs spread
            if pd.notna(v_open) and v_open > 80:
                total_open, spread_open = v_open, h_open
            else:
                total_open, spread_open = h_open, v_open

            # Determine which Close is total vs spread
            if pd.notna(v_close) and v_close > 80:
                total_close, spread_close = v_close, h_close
            elif pd.notna(h_close) and h_close > 80:
                total_close, spread_close = h_close, v_close
            else:
                # Both closes are small — fall back to matching same side as Open
                if pd.notna(v_open) and v_open > 80:
                    total_close, spread_close = v_close, h_close
                else:
                    total_close, spread_close = h_close, v_close

            date_raw = str(row_v.get("Date", "")).strip()
            # Strip any decimal (some xlsx exports add .0)
            date_raw = date_raw.split(".")[0].zfill(4)

            visitor_raw = str(row_v.get("Team", "")).strip()
            home_raw    = str(row_h.get("Team", "")).strip()

            games.append({
                "year":          year,
                "date":          date_raw,
                "visitor":       _norm(visitor_raw),
                "home":          _norm(home_raw),
                "visitor_score": pd.to_numeric(row_v.get("Final"), errors="coerce"),
                "home_score":    pd.to_numeric(row_h.get("Final"), errors="coerce"),
                "total_open":    total_open,
                "total_close":   total_close,
                "spread_open":   spread_open,   # points home team is favored by
                "spread_close":  spread_close,
                "home_ml":       pd.to_numeric(row_h.get("ML"), errors="coerce"),
                "visitor_ml":    pd.to_numeric(row_v.get("ML"), errors="coerce"),
                "neutral_site":  vh_v == "N",
            })
            i += 2
        else:
            i += 1

    return pd.DataFrame(games)


def _norm(name: str) -> str:
    """Normalize team name, suppressing warnings for unknown names."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalize_team_name(name) if is_known_team(name) else name


# ── Filter to NCAA Tournament games ──────────────────────────────────────────

def filter_to_tournament(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Two-step filter per spec:
    1. Date range 0315–0410
    2. Cross-reference against scraped tournament game list
    """
    if df.empty:
        return df

    # Step 1: date range
    date_mask = (df["date"] >= TOURN_DATE_MIN) & (df["date"] <= TOURN_DATE_MAX)
    df_dates = df[date_mask].copy()
    if df_dates.empty:
        return df_dates

    # Step 2: cross-reference vs known tournament teams for this year
    known = query_df(
        "SELECT team1, team2 FROM historical_results WHERE year = ?",
        params=[year]
    )
    if known.empty:
        print(f"  [sbro] No scraped tournament results for {year} to cross-ref against")
        return df_dates

    # Build a set of (normalized_name) that appear in tournament
    tourn_teams = set()
    for _, row in known.iterrows():
        tourn_teams.add(_norm(str(row["team1"])).lower())
        tourn_teams.add(_norm(str(row["team2"])).lower())

    mask = (
        df_dates["visitor"].str.lower().isin(tourn_teams) |
        df_dates["home"].str.lower().isin(tourn_teams)
    )
    filtered = df_dates[mask].copy()
    print(f"  [sbro] Date filter: {len(df_dates)} rows → Tournament cross-ref: {len(filtered)} games")
    return filtered


# ── Download helpers ──────────────────────────────────────────────────────────

def _try_download(year: int) :
    """Attempt to download SBRO file from known URLs. Returns raw bytes or None."""
    s = requests.Session()
    s.headers.update(HEADERS)
    for url in SBRO_DOWNLOAD_URLS.get(year, []):
        try:
            s.get(url, timeout=10)  # trigger any cookie/redirect
            r = s.get(url, timeout=20)
            if r.status_code == 200 and r.content[:4] == b"PK\x03\x04":
                print(f"  [sbro] Downloaded {year} from {url}")
                return r.content
        except Exception:
            pass
    return None


def _find_local_file(year: int):
    """Return the first existing local file path for this year, or None."""
    for path in SBRO_FILES.get(year, []):
        if path.exists():
            return path
    return None


def _save_raw(content: bytes, year: int) -> Path:
    """Save downloaded bytes to data/raw/."""
    dest = RAW_DIR / f"sbro_{year}.xlsx"
    dest.write_bytes(content)
    return dest


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_sbro_year(year: int) -> pd.DataFrame:
    """
    Full ingestion for one SBRO year:
    1. Try local file
    2. Try downloading from known URLs
    3. Parse → filter to tournament games → store in historical_lines table
    """
    # 1. Check local
    local = _find_local_file(year)
    if local:
        print(f"  [sbro] Using local file: {local.name}")
        raw_df = parse_sbro_file(local, year)
    else:
        # 2. Try download
        content = _try_download(year)
        if content:
            path = _save_raw(content, year)
            raw_df = parse_sbro_file(path, year)
        else:
            print(f"  [sbro] No file found for {year}. Place file at: {RAW_DIR}/sbro_{year}.xlsx")
            return pd.DataFrame()

    print(f"  [sbro] Parsed {len(raw_df)} total games for {year}")

    # Filter to tournament
    tourn_df = filter_to_tournament(raw_df, year)
    if tourn_df.empty:
        print(f"  [sbro] No tournament games found for {year}")
        return pd.DataFrame()

    # Store in historical_lines
    lines_rows = _to_lines_format(tourn_df, year)
    if lines_rows.empty:
        return pd.DataFrame()

    with db_conn() as conn:
        conn.execute(
            "DELETE FROM historical_lines WHERE year = ? AND source = 'sbro'",
            [year]
        )
    upsert_df(lines_rows, "historical_lines", if_exists="append")
    print(f"  [sbro] Stored {len(lines_rows)} tournament lines for {year}")
    return lines_rows


def _to_lines_format(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Convert SBRO game rows to historical_lines table format."""
    rows = []
    for _, g in df.iterrows():
        visitor = str(g["visitor"])
        home    = str(g["home"])
        v_score = g.get("visitor_score")
        h_score = g.get("home_score")

        # Spread sign: spread_close > 0 means home favored by that many points
        # Convert to favorite perspective: negative = home favored
        spread = g.get("spread_close")
        if pd.notna(spread):
            # Home-favored spread in SBRO is positive → market convention: home -spread
            spread_line = -float(spread) if float(spread) != 0 else 0.0
            spread_fav  = home if (pd.notna(spread) and float(spread) > 0) else (
                visitor if (pd.notna(spread) and float(spread) < 0) else "EVEN"
            )
        else:
            spread_line = None
            spread_fav  = None

        total = g.get("total_close")

        # ATS result (from home team perspective since spread is home-referenced)
        ats = None
        if pd.notna(v_score) and pd.notna(h_score) and pd.notna(spread):
            actual_margin = float(h_score) - float(v_score)  # positive = home won
            cover = actual_margin + float(spread)             # home covers if > 0
            if cover > 0:
                ats = "HOME_COVER"
            elif cover < 0:
                ats = "VISITOR_COVER"
            else:
                ats = "PUSH"

        ou = None
        if pd.notna(v_score) and pd.notna(h_score) and pd.notna(total):
            actual_total = float(v_score) + float(h_score)
            ou = "OVER" if actual_total > float(total) else (
                "UNDER" if actual_total < float(total) else "PUSH"
            )

        rows.append({
            "year":            year,
            "game_date":       _mmdd_to_date(str(g["date"]), year),
            "team1":           visitor,
            "team2":           home,
            "spread_favorite": spread_fav,
            "spread_line":     spread_line,
            "total_line":      float(total) if pd.notna(total) else None,
            "open_spread":     -float(g["spread_open"]) if pd.notna(g.get("spread_open")) else None,
            "open_total":      float(g["total_open"]) if pd.notna(g.get("total_open")) else None,
            "ats_result":      ats,
            "ou_result":       ou,
            "source":          "sbro",
        })

    return pd.DataFrame(rows)


def _mmdd_to_date(mmdd: str, year: int) -> str:
    """Convert MMDD string to YYYY-MM-DD. Assumes March-April = same year."""
    mmdd = str(mmdd).zfill(4)
    mm = int(mmdd[:2])
    dd = int(mmdd[2:])
    # Tournament is always in the same calendar year as the tournament year
    return f"{year}-{mm:02d}-{dd:02d}"


def ingest_all_sbro() -> dict:
    """Run SBRO ingestion for all available years."""
    results = {}
    for year in [2019, 2021, 2022]:
        print(f"\n[sbro] Processing {year}...")
        df = ingest_sbro_year(year)
        results[year] = len(df) if not df.empty else 0
    return results
