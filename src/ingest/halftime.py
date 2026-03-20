"""
Fetches first-half box score data for historical tournament games using the ESPN
summary endpoint.

ESPN endpoint:
  https://site.api.espn.com/apis/site/v2/sports/basketball/
  mens-college-basketball/summary?event={espn_game_id}

Notes on stat availability:
  - Period scores:  competitions[0].competitors[i].linescores[0].value  (index 0 = first half)
  - For COMPLETED games ESPN only returns full-game cumulative box-score stats —
    it does NOT expose half-by-half shooting splits.  eFG%, offensive rebounds,
    and turnovers below are therefore full-game proxies, not true first-half
    figures.  This limitation is flagged wherever the values are written.
  - Team stats live at: boxscore.teams[i].statistics
    Keys used: fieldGoalsMade, fieldGoalsAttempted, threePointFieldGoalsMade,
               offensiveRebounds, turnovers
  - eFG% = (FGM + 0.5 * 3PM) / FGA
"""

import time
import requests
from typing import Optional, List, Dict

from src.utils.db import db_conn, query_df

HEADERS = {"User-Agent": "Mozilla/5.0"}
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary"
)

# Batch configuration
BATCH_SIZE = 20
BATCH_SLEEP_SECS = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stat_value(statistics: List[Dict], name: str) -> Optional[float]:
    """
    Return the numeric value for a named stat from an ESPN statistics list.
    Stats use displayValue (string). Combined stats like 'fieldGoalsMade-fieldGoalsAttempted'
    have displayValue '23-64' — return only the first number (made).
    """
    for stat in statistics:
        if stat.get("name") == name:
            try:
                raw = str(stat.get("displayValue", ""))
                # Combined stats like "23-64" → take first number
                return float(raw.split("-")[0])
            except (TypeError, ValueError):
                return None
    return None


def _stat_attempted(statistics: List[Dict], name: str) -> Optional[float]:
    """
    For combined stats like 'fieldGoalsMade-fieldGoalsAttempted' (displayValue '23-64'),
    return the second number (attempted).
    """
    for stat in statistics:
        if stat.get("name") == name:
            try:
                parts = str(stat.get("displayValue", "")).split("-")
                if len(parts) >= 2:
                    return float(parts[1])
            except (TypeError, ValueError):
                return None
    return None


def _compute_efg(fgm: Optional[float], fga: Optional[float], tpm: Optional[float]) -> Optional[float]:
    """eFG% = (FGM + 0.5 * 3PM) / FGA.  Returns None if inputs are invalid."""
    if fgm is None or fga is None or tpm is None:
        return None
    if fga == 0:
        return None
    return (fgm + 0.5 * tpm) / fga


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_halftime_data(
    espn_game_id: str,
    year: int,
    game_date: str,
    team1: str,
    team2: str,
) -> Optional[Dict]:
    """
    Fetch first-half data for a single game from the ESPN summary endpoint.

    Returns a dict matching the halftime_scores schema, or None on failure.

    The returned dict uses:
      - h1_score1 / h1_score2  : true first-half scores from linescores[0]
      - h1_efg1 / h1_efg2      : FULL-GAME eFG% proxies (see module docstring)
      - h1_orb1 / h1_orb2      : FULL-GAME offensive rebound proxies
      - h1_to1  / h1_to2       : FULL-GAME turnover proxies
    """
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            headers=HEADERS,
            params={"event": espn_game_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [ESPN fetch error] game_id={espn_game_id}: {exc}")
        return None

    # Detect ESPN 404 / invalid ID responses
    if "code" in data and "header" not in data:
        # ESPN returns {"code": 404, "message": "..."} for unknown IDs
        return None

    try:
        competition = data["header"]["competitions"][0]
    except (KeyError, IndexError, TypeError):
        print(f"  [parse error] no competition data for game_id={espn_game_id}")
        return None

    competitors = competition.get("competitors", [])
    if len(competitors) < 2:
        print(f"  [parse error] fewer than 2 competitors for game_id={espn_game_id}")
        return None

    # ESPN orders competitors as [home, away] but ordering can vary; we keep
    # the ESPN order and map index 0 -> team1 position, index 1 -> team2 position.
    # The caller is responsible for consistent ordering matching historical_results.

    def _linescore_h1(competitor: Dict) -> Optional[int]:
        """Extract first-half score from linescores[0].displayValue."""
        try:
            return int(competitor["linescores"][0]["displayValue"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    h1_score1 = _linescore_h1(competitors[0])
    h1_score2 = _linescore_h1(competitors[1])

    if h1_score1 is None or h1_score2 is None:
        print(f"  [missing H1 scores] game_id={espn_game_id}")
        return None

    # Box-score stats — full-game proxies for completed games
    # NOTE: boxscore.teams ordering matches the competitors ordering above.
    box_teams = []
    try:
        box_teams = data["boxscore"]["teams"]
    except (KeyError, TypeError):
        pass  # stats will remain None

    def _team_stats(idx: int) -> list[dict]:
        try:
            return box_teams[idx]["statistics"]
        except (IndexError, KeyError, TypeError):
            return []

    stats1 = _team_stats(0)
    stats2 = _team_stats(1)

    # ESPN returns FGM/FGA as combined string "made-attempted" under one stat name
    FG_KEY  = "fieldGoalsMade-fieldGoalsAttempted"
    TPM_KEY = "threePointFieldGoalsMade-threePointFieldGoalsAttempted"

    fgm1 = _stat_value(stats1, FG_KEY)
    fga1 = _stat_attempted(stats1, FG_KEY)
    tpm1 = _stat_value(stats1, TPM_KEY)
    orb1 = _stat_value(stats1, "offensiveRebounds")
    to1  = _stat_value(stats1, "turnovers")

    fgm2 = _stat_value(stats2, FG_KEY)
    fga2 = _stat_attempted(stats2, FG_KEY)
    tpm2 = _stat_value(stats2, TPM_KEY)
    orb2 = _stat_value(stats2, "offensiveRebounds")
    to2  = _stat_value(stats2, "turnovers")

    # eFG% (full-game proxy — see module docstring)
    efg1 = _compute_efg(fgm1, fga1, tpm1)
    efg2 = _compute_efg(fgm2, fga2, tpm2)

    return {
        "year":          year,
        "game_date":     game_date,
        "team1":         team1,
        "team2":         team2,
        "espn_game_id":  espn_game_id,
        "h1_score1":     h1_score1,
        "h1_score2":     h1_score2,
        "h1_margin":     float(h1_score1 - h1_score2),
        "h1_combined":   h1_score1 + h1_score2,
        # Full-game proxies — NOT true H1 splits
        "h1_efg1":  efg1,
        "h1_efg2":  efg2,
        "h1_orb1":  int(orb1) if orb1 is not None else None,
        "h1_orb2":  int(orb2) if orb2 is not None else None,
        "h1_to1":   int(to1)  if to1  is not None else None,
        "h1_to2":   int(to2)  if to2  is not None else None,
        "source":   "espn_summary_fullgame_proxy",
    }


def fetch_all_halftime_history() -> None:
    """
    Loop over historical_results rows that have an espn_game_id but no
    corresponding halftime_scores entry, fetch data from ESPN in batches of
    BATCH_SIZE, and upsert results to halftime_scores.
    """
    sql = """
        SELECT
            hr.year,
            hr.game_date,
            hr.team1,
            hr.team2,
            hr.espn_game_id
        FROM historical_results hr
        LEFT JOIN halftime_scores hs
            ON  hr.year      = hs.year
            AND hr.game_date = hs.game_date
            AND hr.team1     = hs.team1
            AND hr.team2     = hs.team2
        WHERE hr.espn_game_id IS NOT NULL
          AND hs.year IS NULL
        ORDER BY hr.game_date
    """
    df = query_df(sql)
    if df.empty:
        print("No games needing halftime data.")
        return

    print(f"Fetching halftime data for {len(df)} games...")

    rows = df.to_dict(orient="records")
    total_ok = 0
    total_fail = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start: batch_start + BATCH_SIZE]
        print(
            f"  Batch {batch_start // BATCH_SIZE + 1}: "
            f"games {batch_start + 1}–{batch_start + len(batch)}"
        )

        for row in batch:
            result = fetch_halftime_data(
                espn_game_id=row["espn_game_id"],
                year=row["year"],
                game_date=row["game_date"],
                team1=row["team1"],
                team2=row["team2"],
            )
            if result is None:
                total_fail += 1
                continue

            # Upsert into halftime_scores
            try:
                with db_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO halftime_scores (
                            year, game_date, team1, team2, espn_game_id,
                            h1_score1, h1_score2, h1_margin, h1_combined,
                            h1_efg1, h1_efg2,
                            h1_orb1, h1_orb2,
                            h1_to1,  h1_to2,
                            source
                        ) VALUES (
                            :year, :game_date, :team1, :team2, :espn_game_id,
                            :h1_score1, :h1_score2, :h1_margin, :h1_combined,
                            :h1_efg1, :h1_efg2,
                            :h1_orb1, :h1_orb2,
                            :h1_to1,  :h1_to2,
                            :source
                        )
                        ON CONFLICT(year, game_date, team1, team2) DO UPDATE SET
                            espn_game_id = excluded.espn_game_id,
                            h1_score1    = excluded.h1_score1,
                            h1_score2    = excluded.h1_score2,
                            h1_margin    = excluded.h1_margin,
                            h1_combined  = excluded.h1_combined,
                            h1_efg1      = excluded.h1_efg1,
                            h1_efg2      = excluded.h1_efg2,
                            h1_orb1      = excluded.h1_orb1,
                            h1_orb2      = excluded.h1_orb2,
                            h1_to1       = excluded.h1_to1,
                            h1_to2       = excluded.h1_to2,
                            source       = excluded.source
                        """,
                        result,
                    )
                total_ok += 1
            except Exception as exc:
                print(
                    f"  [DB write error] {row['team1']} vs {row['team2']} "
                    f"on {row['game_date']}: {exc}"
                )
                total_fail += 1

        if batch_start + BATCH_SIZE < len(rows):
            time.sleep(BATCH_SLEEP_SECS)

    print(f"\nDone. Inserted/updated: {total_ok}, Failed: {total_fail}")


if __name__ == "__main__":
    fetch_all_halftime_history()
