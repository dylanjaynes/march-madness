import time
import requests
import pandas as pd
from datetime import datetime

from src.utils.config import TORVIK_BASE, SELECTION_SUNDAY
from src.utils.db import db_conn, upsert_df
from src.utils.team_map import normalize_team_name

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_SESSION = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(HEADERS)
    return _SESSION


def _get(url: str, retries: int = 3) -> list:
    """
    BartTorvik uses a JS verification challenge.
    Bypass: GET to trigger cookie, then POST with js_test_submitted=1.
    """
    s = _get_session()
    for attempt in range(retries):
        try:
            s.get(url, timeout=15)
            resp = s.post(url, data={"js_test_submitted": "1"}, timeout=15)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                raise ValueError("Empty response")
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_team_ratings(year: int, end_date: str = None) -> pd.DataFrame:
    """
    Pull full team efficiency table from BartTorvik for a given season year.
    end_date: MMDD string (e.g. '0317') — if provided, stats are accumulated
              only through that date, eliminating post-Selection Sunday leakage.
    """
    url = f"{TORVIK_BASE}/trank.php?year={year}&json=1"
    if end_date:
        url += f"&edate={end_date}"  # BartTorvik uses 'edate' for end-date filter
    data = _get(url)
    time.sleep(1)

    rows = []
    for item in data:
        # BartTorvik JSON is an array of arrays or dicts depending on version
        if isinstance(item, list):
            # Array format — columns vary by year; parse positionally
            row = _parse_rating_row_array(item, year)
        elif isinstance(item, dict):
            row = _parse_rating_row_dict(item, year)
        else:
            continue
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  [torvik] No ratings data for {year}")
        return df
    df["year"] = year
    df["updated_at"] = datetime.utcnow().isoformat()
    return df


def _parse_rating_row_dict(item: dict, year: int) -> dict:
    try:
        team_raw = item.get("team") or item.get("Team") or item.get("rk", {})
        if not team_raw or not isinstance(team_raw, str):
            return None
        return {
            "team": normalize_team_name(team_raw),
            "conf": item.get("conf") or item.get("Conf", ""),
            "adj_o": float(item.get("adjoe") or item.get("adj_o") or item.get("AdjOE", 0) or 0),
            "adj_d": float(item.get("adjde") or item.get("adj_d") or item.get("AdjDE", 0) or 0),
            "adj_t": float(item.get("tempo") or item.get("adj_t") or item.get("Tempo", 0) or 0),
            "barthag": float(item.get("barthag") or item.get("Barthag", 0) or 0),
            "efg_o": float(item.get("efg_o") or item.get("EFG%", 0) or 0),
            "efg_d": float(item.get("efg_d") or item.get("EFGD%", 0) or 0),
            "to_rate_o": float(item.get("tov_o") or item.get("to_rate_o") or item.get("TOR", 0) or 0),
            "to_rate_d": float(item.get("tov_d") or item.get("to_rate_d") or item.get("TORD", 0) or 0),
            "or_rate_o": float(item.get("orb_o") or item.get("or_rate_o") or item.get("ORB", 0) or 0),
            "or_rate_d": float(item.get("orb_d") or item.get("or_rate_d") or item.get("DRB", 0) or 0),
            "ft_rate_o": float(item.get("ftr_o") or item.get("ft_rate_o") or item.get("FTR", 0) or 0),
            "ft_rate_d": float(item.get("ftr_d") or item.get("ft_rate_d") or item.get("FTRD", 0) or 0),
            "three_pt_rate_o": float(item.get("3pr_o") or item.get("three_pt_rate_o") or item.get("2P%", 0) or 0),
            "three_pt_rate_d": float(item.get("3pr_d") or item.get("three_pt_rate_d") or 0),
            "three_pt_pct_o": float(item.get("3p_o") or item.get("three_pt_pct_o") or item.get("3P%", 0) or 0),
            "three_pt_pct_d": float(item.get("3p_d") or item.get("three_pt_pct_d") or 0),
            "two_pt_pct_o": float(item.get("2p_o") or item.get("two_pt_pct_o") or 0),
            "two_pt_pct_d": float(item.get("2p_d") or item.get("two_pt_pct_d") or 0),
            "sos": float(item.get("sos") or item.get("SOS", 0) or 0),
            "seed": None,
        }
    except (TypeError, ValueError):
        return None


def _parse_rating_row_array(item: list, year: int) -> dict:
    """
    BartTorvik trank JSON confirmed positional format (37 elements):
    [0] team, [1] adj_o, [2] adj_d, [3] barthag, [4] record, [5] wins, [6] games,
    [7] efg_o, [8] efg_d, [9] to_rate_o, [10] to_rate_d,
    [11] or_rate_o, [12] or_rate_d, [13] ft_rate_o, [14] ft_rate_d,
    [15] two_pt_pct_o, [16] two_pt_pct_d, [17] three_pt_pct_o, [18] three_pt_pct_d,
    [19] three_pt_rate_o, [20] three_pt_rate_d,
    [21] ??, [22] adj_t, [23..26] ?? ,
    [27..29] empty, [30] year, [31..33] empty, [34] sos, [35] ??, [36] None
    """
    if len(item) < 10:
        return None
    try:
        team_raw = str(item[0])
        if not team_raw or team_raw.isdigit():
            return None
        return {
            "team": normalize_team_name(team_raw),
            "conf": "",  # Not in array format; will be null
            "adj_o": float(item[1] or 0),
            "adj_d": float(item[2] or 0),
            "barthag": float(item[3] or 0),
            "efg_o": float(item[7] or 0),
            "efg_d": float(item[8] or 0),
            "to_rate_o": float(item[9] or 0),
            "to_rate_d": float(item[10] or 0),
            "or_rate_o": float(item[11] or 0),
            "or_rate_d": float(item[12] or 0),
            "ft_rate_o": float(item[13] or 0),
            "ft_rate_d": float(item[14] or 0),
            "two_pt_pct_o": float(item[15] or 0),
            "two_pt_pct_d": float(item[16] or 0),
            "three_pt_pct_o": float(item[17] or 0),
            "three_pt_pct_d": float(item[18] or 0),
            "three_pt_rate_o": float(item[19] or 0),
            "three_pt_rate_d": float(item[20] or 0),
            "adj_t": float(item[22] or 0) if len(item) > 22 else 0.0,
            "sos": float(item[34] or 0) if len(item) > 34 else 0.0,
            "seed": None,
        }
    except (TypeError, ValueError, IndexError):
        return None


def fetch_team_ratings_at_date(team: str, date_mmdd: str, year: int) -> dict:
    """Get team ratings as-of a specific date (MMDD format) to avoid look-ahead bias."""
    url = f"{TORVIK_BASE}/trank.php?year={year}&begin={date_mmdd}&end={date_mmdd}&json=1"
    try:
        data = _get(url)
        time.sleep(1)
        for item in data:
            row = _parse_rating_row_array(item, year) if isinstance(item, list) else _parse_rating_row_dict(item, year)
            if row and row.get("team") == team:
                return row
    except Exception as e:
        print(f"  [torvik] Error fetching ratings for {team} at {date_mmdd}/{year}: {e}")
    return {}


def fetch_game_results(year: int) -> pd.DataFrame:
    """Pull full schedule + results for a season from BartTorvik."""
    url = f"{TORVIK_BASE}/getgames.php?year={year}&json=1"
    data = _get(url)
    time.sleep(1)

    rows = []
    for item in data:
        if isinstance(item, list):
            row = _parse_game_row_array(item, year)
        elif isinstance(item, dict):
            row = _parse_game_row_dict(item, year)
        else:
            continue
        if row:
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_game_row_dict(item: dict, year: int) -> dict:
    try:
        team_raw = item.get("team") or item.get("Team", "")
        opp_raw = item.get("opp") or item.get("Opp") or item.get("opponent", "")
        if not team_raw or not opp_raw:
            return None
        date_str = str(item.get("date") or item.get("Date", ""))
        loc = str(item.get("loc") or item.get("location") or item.get("Loc", "N"))
        t_score = item.get("team_score") or item.get("pts") or item.get("Pts")
        o_score = item.get("opp_score") or item.get("opp_pts") or item.get("OppPts")
        game_type = str(item.get("type") or item.get("Type") or "")
        is_tourn = "NCAA" in game_type.upper() or "NCAAT" in game_type.upper()
        return {
            "year": year,
            "game_date": date_str,
            "team": normalize_team_name(team_raw),
            "opponent": normalize_team_name(opp_raw),
            "location": loc,
            "team_score": int(t_score) if t_score is not None else None,
            "opp_score": int(o_score) if o_score is not None else None,
            "is_tournament": is_tourn,
            "tournament_round": game_type if is_tourn else None,
        }
    except (TypeError, ValueError):
        return None


def _parse_game_row_array(item: list, year: int) -> dict:
    """
    BartTorvik getgames array format (positional):
    [team, opp, date, ..., loc, t_score, opp_score, ..., type]
    Actual column order can shift — use best-guess positional parsing.
    """
    if len(item) < 8:
        return None
    try:
        team_raw = str(item[0])
        opp_raw = str(item[1])
        date_str = str(item[2])
        loc = str(item[3]) if len(item) > 3 else "N"
        t_score = item[4] if len(item) > 4 else None
        o_score = item[5] if len(item) > 5 else None
        game_type = str(item[-1]) if item else ""
        is_tourn = "NCAA" in game_type.upper()
        return {
            "year": year,
            "game_date": date_str,
            "team": normalize_team_name(team_raw),
            "opponent": normalize_team_name(opp_raw),
            "location": loc,
            "team_score": int(t_score) if t_score not in (None, "") else None,
            "opp_score": int(o_score) if o_score not in (None, "") else None,
            "is_tournament": is_tourn,
            "tournament_round": game_type if is_tourn else None,
        }
    except (TypeError, ValueError, IndexError):
        return None


def fetch_tournament_games(year: int) -> pd.DataFrame:
    """Filter fetch_game_results() to NCAA Tournament games only."""
    df = fetch_game_results(year)
    if df.empty:
        return df
    if "is_tournament" in df.columns:
        tourn = df[df["is_tournament"] == True].copy()
    else:
        tourn = pd.DataFrame()
    # Fallback: filter by mid-March through April date range
    if tourn.empty and "game_date" in df.columns:
        df["_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        mask = (df["_date"].dt.month == 3) & (df["_date"].dt.day >= 14) | (df["_date"].dt.month == 4)
        tourn = df[mask].drop(columns=["_date"])
    return tourn


def store_team_ratings(year: int) -> pd.DataFrame:
    """Fetch and store team ratings for a year to SQLite (current/live ratings)."""
    print(f"  [torvik] Fetching ratings for {year}...")
    df = fetch_team_ratings(year)
    if df.empty:
        print(f"  [torvik] No data for {year}")
        return df
    # Remove any existing rows for this year first
    with db_conn() as conn:
        conn.execute("DELETE FROM torvik_ratings WHERE year = ?", [year])
    upsert_df(df, "torvik_ratings", if_exists="append")
    print(f"  [torvik] Stored {len(df)} teams for {year}")
    return df


def store_ratings_snapshot(year: int, as_of_date: str) -> pd.DataFrame:
    """
    Fetch ratings as-of Selection Sunday (end_date) and store in
    torvik_ratings_snapshot. Called once per historical year during ingestion.
    as_of_date: MMDD string matching SELECTION_SUNDAY[year] (e.g. '0317').
    """
    print(f"  [torvik] Fetching snapshot ratings for {year} as-of {as_of_date}...")
    df = fetch_team_ratings(year, end_date=as_of_date)
    if df.empty:
        print(f"  [torvik] No snapshot data for {year}")
        return df
    df["as_of_date"] = as_of_date
    # Replace existing snapshot for this (year, as_of_date)
    with db_conn() as conn:
        conn.execute(
            "DELETE FROM torvik_ratings_snapshot WHERE year = ? AND as_of_date = ?",
            [year, as_of_date]
        )
    upsert_df(df, "torvik_ratings_snapshot", if_exists="append")
    print(f"  [torvik] Stored {len(df)} teams in snapshot for {year} (as_of={as_of_date})")
    return df


def store_game_results(year: int) -> pd.DataFrame:
    """Fetch and store game results for a year to SQLite."""
    print(f"  [torvik] Fetching games for {year}...")
    df = fetch_game_results(year)
    if df.empty:
        print(f"  [torvik] No games for {year}")
        return df
    with db_conn() as conn:
        conn.execute("DELETE FROM torvik_games WHERE year = ?", [year])
    upsert_df(df, "torvik_games", if_exists="append")
    print(f"  [torvik] Stored {len(df)} games for {year}")
    return df


def refresh_team_ratings():
    """Called by scheduler: refresh current season ratings."""
    from src.utils.config import TOURNAMENT_YEARS
    current_year = TOURNAMENT_YEARS[-1]
    store_team_ratings(current_year)
