import numpy as np
import pandas as pd
from scipy.stats import norm

from src.utils.config import (
    SPREAD_STD_DEV, TOURNAMENT_YEARS, ROUND_NAMES,
    COMPETITIVE_COVERAGE_STD, MISMATCH_COVERAGE_STD,
    COMPETITIVE_SPREAD_THRESHOLD,
)

# COVERAGE_STD = 12.0  # legacy single-std constant (replaced by COMPETITIVE/MISMATCH below)


def is_competitive_game(market_spread) -> bool:
    """
    A game is competitive for ATS betting when the market spread
    is within COMPETITIVE_SPREAD_THRESHOLD points.
    Returns True when no line is available yet (optimistic default).
    """
    if market_spread is None:
        return True
    try:
        return abs(float(market_spread)) <= COMPETITIVE_SPREAD_THRESHOLD
    except (TypeError, ValueError):
        return True


def coverage_probability(model_spread: float, market_spread: float,
                          residual_std: float = COMPETITIVE_COVERAGE_STD) -> float:
    """Probability the bet covers. Models actual_margin ~ N(model_spread, residual_std)."""
    if model_spread >= market_spread:
        return float(1 - norm.cdf(market_spread, loc=model_spread, scale=residual_std))
    else:
        return float(norm.cdf(market_spread, loc=model_spread, scale=residual_std))


def kelly_fraction(win_prob: float, juice: int = -110) -> float:
    """Full Kelly fraction, clamped to [0, 0.30]."""
    if win_prob <= 0.5:
        return 0.0
    b = 100 / abs(juice)
    f = (b * win_prob - (1 - win_prob)) / b
    return max(0.0, min(f, 0.30))


def half_kelly(win_prob: float, juice: int = -110) -> float:
    return kelly_fraction(win_prob, juice) / 2.0


def bet_tier(edge: float, cov_prob: float) -> tuple:
    """Return (label, emoji) tier for a bet."""
    e = abs(edge)
    if e >= 7 and cov_prob >= 0.58:
        return "Strong", "🔥"
    elif e >= 5 and cov_prob >= 0.56:
        return "Value", "✅"
    elif e >= 3 and cov_prob >= 0.54:
        return "Lean", "📊"
    return "Pass", "⚪"


def season_label(year: int) -> str:
    return f"{year - 1}–{str(year)[2:]}"


def data_as_of(year: int) -> str:
    from src.utils.config import SELECTION_SUNDAY
    import datetime
    months = {3: "Mar", 4: "Apr"}
    if year in SELECTION_SUNDAY:
        mmdd = SELECTION_SUNDAY[year]
        mm, dd = int(mmdd[:2]), int(mmdd[2:])
        return f"Selection Sunday — {months.get(mm,'Mar')} {dd}, {year}"
    return f"Live — {datetime.date.today().strftime('%b %d, %Y')}"
from src.features.matchup import MATCHUP_FEATURES, build_matchup_features
from src.features.adjustments import apply_tournament_pace_adjustment
from src.model.train import load_model
from src.utils.config import (
    MISMATCH_SEED_DIFF_THRESHOLD, MISMATCH_BARTHAG_THRESHOLD,
)


def spread_to_win_prob(spread: float, std_dev: float = SPREAD_STD_DEV) -> float:
    """
    Convert point spread to win probability using normal distribution.
    Positive spread = team_a is favored by spread pts.
    """
    return float(norm.cdf(0, loc=-spread, scale=std_dev))


def _load_hybrid_models():
    """
    Try to load all four hybrid model files.
    Returns (comp_model, mis_model, cal_comp, cal_mis) or raises FileNotFoundError.
    """
    return (
        load_model("spread_competitive"),
        load_model("spread_mismatch"),
        load_model("cal_competitive"),
        load_model("cal_mismatch"),
    )


def project_game(team_a: str, team_b: str,
                 round_num: int, year: int = None,
                 seed_a: int = None, seed_b: int = None,
                 game_date: str = None) -> dict:
    """
    Project spread and total for a single tournament game.
    Results are always returned from the perspective of the better seed (lower number).
    Input order doesn't matter — teams are reordered to match training convention.

    Uses the two-stage hybrid model when available (spread_competitive.pkl +
    spread_mismatch.pkl + calibrators), falling back to spread_model.pkl otherwise.
    The returned dict includes is_mismatch (bool) so the UI can annotate the game.
    """
    if year is None:
        year = TOURNAMENT_YEARS[-1]

    # Canonical team ordering: better seed (lower number) is team_a.
    if seed_a is None or seed_b is None:
        from src.utils.db import query_df
        rows = query_df(
            "SELECT team, seed FROM torvik_ratings WHERE year=? AND team IN (?,?) AND seed IS NOT NULL",
            params=[year, team_a, team_b],
        )
        seed_map = dict(zip(rows["team"], rows["seed"])) if not rows.empty else {}
        sa = int(seed_map.get(team_a) or 8)
        sb = int(seed_map.get(team_b) or 8)
    else:
        sa, sb = seed_a, seed_b

    if sb < sa or (sb == sa and team_b < team_a):
        team_a, team_b = team_b, team_a
        seed_a, seed_b = sb, sa
    else:
        seed_a, seed_b = sa, sb

    # Try hybrid model first; fall back to legacy spread_model.pkl.
    try:
        comp_model, mis_model, cal_comp, cal_mis = _load_hybrid_models()
        use_hybrid = True
    except FileNotFoundError:
        try:
            spread_model = load_model("spread_model")
        except FileNotFoundError as e:
            return {"error": str(e)}
        use_hybrid = False

    try:
        total_model = load_model("total_model")
    except FileNotFoundError as e:
        return {"error": str(e)}

    feats = build_matchup_features(
        team_a, team_b, year, round_num,
        seed_a=seed_a, seed_b=seed_b,
        game_date=game_date,
    )

    if feats.isna().all():
        return {
            "error": f"Could not build features for {team_a} vs {team_b}",
            "team_a": team_a,
            "team_b": team_b,
        }

    X = feats[MATCHUP_FEATURES].values.reshape(1, -1)

    if use_hybrid:
        seed_diff = seed_a - seed_b  # negative when team_a is better seed

        # Check barthag_diff if available in feats (feats is a Series)
        barthag_diff = None
        if "barthag_diff" in feats.index:
            v = feats["barthag_diff"]
            if v == v:  # NaN check (NaN != NaN)
                barthag_diff = float(v)

        is_mismatch = abs(seed_diff) >= MISMATCH_SEED_DIFF_THRESHOLD
        if barthag_diff is not None:
            is_mismatch = is_mismatch or abs(barthag_diff) >= MISMATCH_BARTHAG_THRESHOLD

        if is_mismatch:
            raw_spread = float(mis_model.predict(X)[0])
            projected_spread = float(cal_mis.predict([raw_spread])[0])
        else:
            raw_spread = float(comp_model.predict(X)[0])
            projected_spread = float(cal_comp.predict([raw_spread])[0])
    else:
        is_mismatch = False
        projected_spread = float(spread_model.predict(X)[0])

    projected_total_raw = float(total_model.predict(X)[0])

    # Apply isotonic calibration for total model if available
    try:
        import pickle
        from pathlib import Path
        _cal_path = Path("models/total_model_calibrator.pkl")
        if _cal_path.exists():
            with open(_cal_path, "rb") as _f:
                _total_cal = pickle.load(_f)
            projected_total_raw = float(_total_cal.predict([projected_total_raw])[0])
    except Exception:
        pass  # calibrator not yet trained, use raw prediction

    projected_total = apply_tournament_pace_adjustment(projected_total_raw)

    projected_score_a = (projected_total + projected_spread) / 2
    projected_score_b = (projected_total - projected_spread) / 2

    win_prob_a = spread_to_win_prob(projected_spread)

    cov_std = MISMATCH_COVERAGE_STD if is_mismatch else COMPETITIVE_COVERAGE_STD

    return {
        "team_a": team_a,
        "team_b": team_b,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "round_num": round_num,
        "round_name": ROUND_NAMES.get(round_num, f"R{round_num}"),
        "projected_spread": projected_spread,
        "projected_total": projected_total,
        "projected_score_a": projected_score_a,
        "projected_score_b": projected_score_b,
        "win_prob_a": win_prob_a,
        "win_prob_b": 1 - win_prob_a,
        "is_mismatch": is_mismatch,
        "cov_std": cov_std,
    }


def project_all_live_games(year: int = None) -> pd.DataFrame:
    """
    Project all live tournament games by joining odds + model predictions.
    Returns DataFrame with model lines, market lines, and edges.
    """
    from src.ingest.odds import get_latest_odds
    from src.utils.db import query_df

    if year is None:
        year = TOURNAMENT_YEARS[-1]

    odds_df = get_latest_odds()

    rows = []
    if odds_df.empty:
        # Fall back to 2025 tournament bracket if no live odds
        bracket_df = query_df(
            "SELECT * FROM historical_results WHERE year = ? ORDER BY round_number, game_date",
            params=[year]
        )
        for _, game in bracket_df.iterrows():
            t1, t2 = game["team1"], game["team2"]
            rn = int(game["round_number"] or 1)
            proj = project_game(t1, t2, rn, year,
                                seed_a=game.get("seed1"),
                                seed_b=game.get("seed2"),
                                game_date=str(game.get("game_date") or ""))
            if "error" not in proj:
                proj["market_spread"] = None
                proj["market_total"] = None
                proj["spread_edge"] = None
                proj["total_edge"] = None
                proj["game_date"] = game.get("game_date")
                rows.append(proj)
    else:
        for _, odds in odds_df.iterrows():
            home = odds["home_team"]
            away = odds["away_team"]
            mkt_spread = odds.get("spread_home")
            mkt_total = odds.get("total_line")

            # Determine which is team_a (better seed) — default to home for live games
            proj = project_game(home, away, round_num=1, year=year)
            if "error" not in proj:
                proj["market_spread"] = mkt_spread
                proj["market_total"] = mkt_total
                proj["spread_edge"] = (
                    proj["projected_spread"] - mkt_spread
                    if mkt_spread is not None else None
                )
                proj["total_edge"] = (
                    proj["projected_total"] - mkt_total
                    if mkt_total is not None else None
                )
                proj["game_date"] = odds.get("commence_time", "")
                rows.append(proj)

    return pd.DataFrame(rows)
