import pandas as pd
from src.utils.db import query_df

# Module-level ratings cache — pre-populate with load_ratings_cache() to avoid
# per-team DB queries during bulk operations like Monte Carlo precomputation.
_ratings_cache: dict = {}  # key: (team, year) -> dict


def load_ratings_cache(teams: list, year: int) -> None:
    """
    Batch-load ratings for all given teams in a single DB query and store
    in module-level cache. Call this before bulk matchup computation to
    eliminate per-team DB round-trips.
    """
    if not teams:
        return
    placeholders = ",".join("?" * len(teams))
    df = query_df(
        f"SELECT * FROM torvik_ratings WHERE year = ? AND team IN ({placeholders})",
        params=[year] + list(teams),
    )
    for _, row in df.iterrows():
        _ratings_cache[(row["team"], year)] = row.to_dict()


def clear_ratings_cache() -> None:
    _ratings_cache.clear()


def build_team_feature_vector(team: str, year: int, as_of_date: str = None) -> dict:
    """
    Pull efficiency metrics for one team.

    When as_of_date (MMDD string) is provided, queries torvik_ratings_snapshot
    so historical training uses only stats known at Selection Sunday — no leakage.
    Falls back to torvik_ratings (current-season live ratings) when as_of_date
    is None or the snapshot row is missing.

    Checks module-level _ratings_cache first (populated by load_ratings_cache)
    to avoid per-team DB round-trips during bulk operations.
    """
    # Fast path: check cache (only when no as_of_date override)
    if not as_of_date:
        cache_key = (team, year)
        if cache_key in _ratings_cache:
            return _ratings_cache[cache_key]

    if as_of_date:
        sql = (
            "SELECT * FROM torvik_ratings_snapshot "
            "WHERE year = ? AND team = ? AND as_of_date = ? LIMIT 1"
        )
        rows = query_df(sql, params=[year, team, as_of_date])
        if rows.empty:
            sql_ci = (
                "SELECT * FROM torvik_ratings_snapshot "
                "WHERE year = ? AND LOWER(team) = LOWER(?) AND as_of_date = ? LIMIT 1"
            )
            rows = query_df(sql_ci, params=[year, team, as_of_date])

    if not as_of_date or rows.empty:
        sql = "SELECT * FROM torvik_ratings WHERE year = ? AND team = ? LIMIT 1"
        rows = query_df(sql, params=[year, team])
        if rows.empty:
            sql_ci = "SELECT * FROM torvik_ratings WHERE year = ? AND LOWER(team) = LOWER(?)"
            rows = query_df(sql_ci, params=[year, team])

    # Final fallback: hyphen → space normalization
    # Handles "Nebraska-Omaha" → "Nebraska Omaha", "UC-San Diego" → "UC San Diego",
    # "SIU-Edwardsville" → "SIU Edwardsville", etc.
    if rows.empty:
        alt = team.replace("-", " ")
        if alt != team:
            rows = query_df(
                "SELECT * FROM torvik_ratings WHERE year = ? AND LOWER(team) = LOWER(?)",
                params=[year, alt],
            )

    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def apply_recency_weighting(team: str, year: int, decay_halflife_days: int = 30) -> dict:
    """
    Recompute team ratings weighting recent games more heavily.
    Uses exponential decay on game-level results from torvik_games.
    Returns an enhanced feature dict, or empty dict if insufficient data.
    """
    import numpy as np
    from datetime import datetime

    sql = """
        SELECT game_date, team_score, opp_score, location
        FROM torvik_games
        WHERE year = ? AND team = ? AND team_score IS NOT NULL AND opp_score IS NOT NULL
        ORDER BY game_date DESC
    """
    df = query_df(sql, params=[year, team])
    if df.empty:
        return {}

    # Parse dates and compute days-from-latest
    df["_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["_date"])
    if df.empty:
        return {}

    latest = df["_date"].max()
    df["days_ago"] = (latest - df["_date"]).dt.days
    df["weight"] = np.exp(-df["days_ago"] * np.log(2) / decay_halflife_days)

    total_weight = df["weight"].sum()
    if total_weight == 0:
        return {}

    # Weighted scoring averages
    df["margin"] = df["team_score"] - df["opp_score"]
    weighted_margin = (df["margin"] * df["weight"]).sum() / total_weight

    return {
        "recency_weighted_margin": weighted_margin,
        "recent_games_count": len(df),
    }


def get_national_averages(year: int) -> dict:
    """Compute national average efficiency stats for a season."""
    sql = "SELECT * FROM torvik_ratings WHERE year = ?"
    df = query_df(sql, params=[year])
    if df.empty:
        return {}
    numeric_cols = ["adj_o", "adj_d", "adj_t", "barthag", "efg_o", "efg_d",
                    "to_rate_o", "to_rate_d", "or_rate_o", "or_rate_d",
                    "ft_rate_o", "ft_rate_d"]
    avgs = {}
    for col in numeric_cols:
        if col in df.columns:
            avgs[f"avg_{col}"] = df[col].mean()
    return avgs
