import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.utils.config import TOURNAMENT_YEARS, SELECTION_SUNDAY
from src.utils.db import db_conn, upsert_df
from src.utils.team_map import normalize_team_name, is_known_team

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Sports Reference region div IDs
REGIONS = ["east", "midwest", "south", "west", "national"]

# Round number by position within bracket
# sports-reference lists rounds left to right: R64, R32, S16, E8, F4, Champ
ROUND_BY_REGION_POSITION = {0: 1, 1: 2, 2: 3, 3: 4}  # per-region round nums
NATIONAL_ROUND_POSITION = {0: 5, 1: 6}  # F4, Champ


def scrape_tournament_results(year: int) -> pd.DataFrame:
    """
    Scrape NCAA Tournament results from sports-reference.com.
    Structure: region divs (east/west/south/midwest) + national div.
    Each div has round divs, each round has game divs with team/score links.
    """
    url = f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
    print(f"  [historical] Scraping {year} from {url}")
    time.sleep(2.5)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [historical] Failed to fetch {year}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []

    # Scrape regional brackets (R64 through E8)
    for region_id in REGIONS[:4]:
        region_div = soup.find("div", id=region_id)
        if not region_div:
            continue
        rounds = region_div.find_all("div", class_="round", recursive=False)
        if not rounds:
            # Try one level deeper (bracket div)
            bracket = region_div.find("div", id="bracket")
            if bracket:
                rounds = bracket.find_all("div", class_="round", recursive=False)

        for round_idx, round_div in enumerate(rounds):
            round_num = round_idx + 1  # 1=R64, 2=R32, 3=S16, 4=E8
            _parse_round_div(round_div, year, round_num, rows)

    # Scrape Final Four + Championship (national div)
    national_div = soup.find("div", id="national")
    if national_div:
        rounds = national_div.find_all("div", class_="round", recursive=False)
        if not rounds:
            bracket = national_div.find("div", id="bracket")
            if bracket:
                rounds = bracket.find_all("div", class_="round", recursive=False)
        for round_idx, round_div in enumerate(rounds):
            round_num = 5 + round_idx  # 5=F4, 6=Champ
            _parse_round_div(round_div, year, round_num, rows)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  [historical] No data parsed for {year}")
        return df

    # Deduplicate (same game might be parsed from both sides)
    df = df.drop_duplicates(subset=["year", "team1", "team2", "round_number"])
    df["year"] = year
    print(f"  [historical] Parsed {len(df)} games for {year}")
    return df


def _parse_round_div(round_div, year: int, round_num: int, rows: list):
    """Parse all games within a round div."""
    # Each game is a direct child div (no class, or class="game")
    game_divs = round_div.find_all("div", recursive=False)

    for game_div in game_divs:
        # Skip non-game divs
        inner_text = game_div.get_text(strip=True)
        if not inner_text:
            continue

        team_divs = game_div.find_all("div", recursive=False)
        if len(team_divs) < 2:
            continue

        team_data = []
        for td in team_divs[:2]:
            td_info = _parse_team_div(td)
            if td_info:
                team_data.append(td_info)

        if len(team_data) < 2:
            continue

        t1, t2 = team_data[0], team_data[1]

        # Extract date from boxscore link
        game_date = None
        score_link = game_div.find("a", href=lambda h: h and "/cbb/boxscores/" in h)
        if score_link:
            m = re.search(r"/cbb/boxscores/(\d{4}-\d{2}-\d{2})", score_link.get("href", ""))
            if m:
                game_date = m.group(1)

        # Determine winner
        t1_is_winner = "winner" in (team_divs[0].get("class") or [])
        t2_is_winner = "winner" in (team_divs[1].get("class") or [])

        if t1_is_winner:
            winner = t1["team"]
        elif t2_is_winner:
            winner = t2["team"]
        elif t1["score"] and t2["score"] and t1["score"] != t2["score"]:
            winner = t1["team"] if t1["score"] > t2["score"] else t2["team"]
        else:
            winner = None

        s1 = t1["score"]
        s2 = t2["score"]

        rows.append({
            "year": year,
            "round_number": round_num,
            "round_name": _round_name(round_num),
            "game_date": game_date,
            "team1": t1["team"],
            "team2": t2["team"],
            "score1": s1,
            "score2": s2,
            "winner": winner,
            "margin": abs(s1 - s2) if (s1 is not None and s2 is not None) else None,
            "total_points": (s1 + s2) if (s1 is not None and s2 is not None) else None,
            "seed1": t1["seed"],
            "seed2": t2["seed"],
        })


def _parse_team_div(td) -> dict:
    """Parse a single team div: seed, name, score."""
    seed_span = td.find("span")
    seed = None
    if seed_span:
        try:
            seed = int(seed_span.text.strip())
        except (ValueError, AttributeError):
            pass

    # Team name from /cbb/schools/ link
    name_link = td.find("a", href=lambda h: h and "/cbb/schools/" in h)
    if not name_link:
        return None
    raw_name = name_link.text.strip()
    team = normalize_team_name(raw_name) if is_known_team(raw_name) else raw_name

    # Score from boxscore link (second <a> tag or numeric text link)
    score = None
    all_links = td.find_all("a")
    for link in all_links:
        if "/cbb/boxscores/" in link.get("href", ""):
            try:
                score = int(link.text.strip())
            except (ValueError, AttributeError):
                pass
            break

    return {"team": team, "seed": seed, "score": score}


def _round_name(round_num: int) -> str:
    names = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}
    return names.get(round_num, f"R{round_num}")


def scrape_historical_lines(year: int) -> pd.DataFrame:
    """
    Historical lines stub — returns empty DataFrame.
    Lines will be populated from a separate seeding step.
    """
    return pd.DataFrame(columns=[
        "year", "game_date", "team1", "team2",
        "spread_favorite", "spread_line", "total_line",
        "ats_result", "ou_result"
    ])


def build_historical_dataset() -> pd.DataFrame:
    """
    Master ingestion function:
    1. Scrape tournament results for each year from Sports Reference
    2. Fetch BartTorvik team ratings as-of Selection Sunday (no post-tournament data)
       — stored in torvik_ratings_snapshot keyed by (year, team, as_of_date)
       — also stored in torvik_ratings for backward compatibility / live use
    3. Fetch game logs for rest-day calculations
    4. Returns combined results DataFrame
    """
    from src.ingest.torvik import fetch_team_ratings, fetch_game_results, store_ratings_snapshot
    from src.utils.db import init_db

    # Ensure snapshot table exists (safe to run on existing DB)
    init_db()

    all_results = []

    for year in TOURNAMENT_YEARS:
        print(f"\n=== Processing {year} ===")
        selection_sunday = SELECTION_SUNDAY[year]  # MMDD, e.g. '0317'

        # 1. Scrape tournament results
        df_results = scrape_tournament_results(year)
        if df_results.empty:
            print(f"  Skipping {year} (no results)")
            continue

        # Store results
        with db_conn() as conn:
            conn.execute("DELETE FROM historical_results WHERE year = ?", [year])
        upsert_df(df_results, "historical_results", if_exists="append")
        print(f"  Stored {len(df_results)} games in historical_results")

        # 2. Fetch BartTorvik ratings as-of Selection Sunday → snapshot table
        try:
            snap_df = store_ratings_snapshot(year, as_of_date=selection_sunday)
            # Also keep torvik_ratings current (full-season) for live projections
            df_ratings = fetch_team_ratings(year)
            if not df_ratings.empty:
                with db_conn() as conn:
                    conn.execute("DELETE FROM torvik_ratings WHERE year = ?", [year])
                upsert_df(df_ratings, "torvik_ratings", if_exists="append")
                print(f"  Stored {len(df_ratings)} full-season team ratings")
        except Exception as e:
            print(f"  Ratings error for {year}: {e}")

        # 3. Fetch game logs (for rest-day calculations)
        try:
            df_games = fetch_game_results(year)
            if not df_games.empty:
                with db_conn() as conn:
                    conn.execute("DELETE FROM torvik_games WHERE year = ?", [year])
                upsert_df(df_games, "torvik_games", if_exists="append")
                print(f"  Stored {len(df_games)} game logs")
        except Exception as e:
            print(f"  Game logs error for {year}: {e}")

        all_results.append(df_results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n=== Total: {len(combined)} tournament games across {len(all_results)} years ===")
        return combined
    return pd.DataFrame()
