import pandas as pd
from src.utils.config import NEUTRAL_SITE_ADJUSTMENT, TOURNAMENT_PACE_HAIRCUT
from src.utils.db import query_df

# Home court advantage to strip from season stats (~3.5 pts in college basketball)
HOME_COURT_ADVANTAGE = 3.5


def apply_neutral_site_correction(team_ratings: dict) -> dict:
    """
    Strip home/away bias from season-long efficiency numbers.
    BartTorvik already adjusts for location, so this is mostly a pass-through.
    We still add NEUTRAL_SITE_ADJUSTMENT = 0 as documented in the spec.
    """
    adjusted = dict(team_ratings)
    # BartTorvik AdjO/AdjD are already neutral-site corrected
    # No modification needed beyond documentation
    return adjusted


def apply_tournament_pace_adjustment(projected_total: float) -> float:
    """
    Apply TOURNAMENT_PACE_HAIRCUT to raw projected totals.
    Tournament games run ~2-3 fewer possessions than regular season.
    """
    return projected_total + TOURNAMENT_PACE_HAIRCUT


def get_rest_days(team: str, game_date: str, year: int) -> int:
    """
    Look up days since last game from torvik_games table.
    Returns 7 (default assumption) if data unavailable.
    """
    sql = """
        SELECT game_date FROM torvik_games
        WHERE year = ? AND team = ? AND game_date < ?
        ORDER BY game_date DESC
        LIMIT 1
    """
    rows = query_df(sql, params=[year, team, game_date])
    if rows.empty:
        return 7  # default assumption for first tournament game
    try:
        last_game = pd.to_datetime(rows.iloc[0]["game_date"])
        current = pd.to_datetime(game_date)
        return max(1, (current - last_game).days)
    except Exception:
        return 7
