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


ROUND_REST_DEFAULTS = {
    1: 12,  # R64: ~12 days after last regular season game
    2: 2,   # R32: 2 days after R64
    3: 4,   # S16: 4 days after R32
    4: 2,   # E8: 2 days after S16
    5: 4,   # F4: 4 days after E8
    6: 2,   # Championship: 2 days after F4
}


def get_rest_days(team: str, game_date: str, year: int, round_num: int = None) -> int:
    """
    Look up days since last game from torvik_games table.
    Returns a round-based default when round_num is provided and data is unavailable,
    otherwise falls back to 7.
    """
    sql = """
        SELECT game_date FROM torvik_games
        WHERE year = ? AND team = ? AND game_date < ?
        ORDER BY game_date DESC
        LIMIT 1
    """
    rows = query_df(sql, params=[year, team, game_date])
    if rows.empty:
        return ROUND_REST_DEFAULTS.get(round_num, 7)
    try:
        last_game = pd.to_datetime(rows.iloc[0]["game_date"])
        current = pd.to_datetime(game_date)
        return max(1, (current - last_game).days)
    except Exception:
        return ROUND_REST_DEFAULTS.get(round_num, 7)
