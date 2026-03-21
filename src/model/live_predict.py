"""
Live in-game prediction engine.
Called by the UI on every refresh cycle.

Uses a trained live_spread_model when available; falls back to the
formula_projected_margin() blended spread projection.
"""

from src.model.predict import coverage_probability, kelly_fraction, half_kelly, bet_tier
from src.model.train import load_model
from src.utils.db import query_df as _query_df


def get_season_averages(team: str, year: int) -> dict:
    """Return season-average efficiency stats from torvik_ratings."""
    try:
        df = _query_df(
            "SELECT efg_o, efg_d, to_rate_o, to_rate_d, or_rate_o, or_rate_d "
            "FROM torvik_ratings WHERE year = ? AND team = ? LIMIT 1",
            params=[year, team],
        )
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Model loading (module-level, cached for process lifetime)
# ---------------------------------------------------------------------------
try:
    _live_model  = load_model("live_spread_model")
    _live_cal    = load_model("live_spread_calibrator")
    _use_trained = True
except FileNotFoundError:
    _live_model  = None
    _live_cal    = None
    _use_trained = False


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def _momentum_adjustment(mom5: float, mom10: float, time_remaining: float) -> float:
    """
    Blend last-5 and last-10 possession momentum into a projected margin adjustment.
    Capped at ±3 points, scaled by time remaining.
    """
    blended = mom5 * 0.6 + mom10 * 0.4
    w = time_remaining / 40.0
    return max(-3.0, min(3.0, blended * 0.3 * w))


def formula_projected_margin(
    current_margin: float,
    pregame_spread: float,
    time_elapsed: float,
    time_remaining: float,
    efg_pct_diff: float = 0.0,
    orb_margin: float = 0.0,
    to_margin: float = 0.0,
) -> float:
    """
    Blended live projected margin.

    Weights the current score by how much game has been played,
    and the pregame model spread by how much remains, then adds
    in-game efficiency adjustments (eFG%, ORB, TO) scaled by time_remaining.

    Parameters
    ----------
    current_margin   : live score difference (team1 - team2)
    pregame_spread   : model or market spread before tip-off (team1 perspective)
    time_elapsed     : minutes elapsed (0–40)
    time_remaining   : minutes remaining (0–40); sum with elapsed should equal 40
    efg_pct_diff     : team1 eFG% minus team2 eFG% (in-game, pct points e.g. 5.2)
    orb_margin       : team1 offensive rebounds minus team2 offensive rebounds
    to_margin        : team1 turnovers minus team2 turnovers (positive = team1 worse)
    """
    w_score = time_elapsed   / 40.0
    w_model = time_remaining / 40.0

    base    = current_margin * w_score + pregame_spread * w_model
    efg_adj = efg_pct_diff * 0.15 * w_model
    orb_adj = orb_margin   * 0.25 * w_model
    to_adj  = to_margin    * 0.40 * w_model

    return base + efg_adj + orb_adj + to_adj


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def project_game_live(
    team1: str,
    team2: str,
    snapshot: dict,
    pregame_spread: float,
    projected_total: float,
    year: int,
) -> dict:
    """
    Return a live prediction dict for a single in-game snapshot.

    Parameters
    ----------
    team1            : home/first team name
    team2            : away/second team name
    snapshot         : dict from live_game_snapshots / fetch_live_game_states()
    pregame_spread   : pre-tip model or market spread (team1 perspective; + = team1 favored)
    projected_total  : pre-game projected total (for context only)
    year             : tournament year

    Returns
    -------
    dict — see module docstring for full schema.
    """
    time_elapsed   = float(snapshot.get("time_elapsed",   0.0))
    time_remaining = float(snapshot.get("time_remaining", 40.0))
    current_margin = float(snapshot.get("current_margin", 0.0))
    game_status    = snapshot.get("game_status", "")
    live_spread    = snapshot.get("live_spread")     # may be None

    score1 = snapshot.get("score1", 0)
    score2 = snapshot.get("score2", 0)

    # In-game stats
    efg_pct1 = snapshot.get("efg_pct1")
    efg_pct2 = snapshot.get("efg_pct2")
    orb1     = snapshot.get("orb1")
    orb2     = snapshot.get("orb2")
    to1      = snapshot.get("to1")
    to2      = snapshot.get("to2")

    # Season averages for context
    avg1 = get_season_averages(team1, year)
    avg2 = get_season_averages(team2, year)
    efg_season1 = avg1.get("efg_o", 0.0) or 0.0
    efg_season2 = avg2.get("efg_o", 0.0) or 0.0

    # Build efficiency differentials for the formula
    efg_diff = 0.0
    if efg_pct1 is not None and efg_pct2 is not None:
        # pct stored as 0–100 from ESPN; convert to decimal if > 1
        e1 = efg_pct1 / 100.0 if efg_pct1 > 1 else efg_pct1
        e2 = efg_pct2 / 100.0 if efg_pct2 > 1 else efg_pct2
        efg_diff = (e1 - e2) * 100.0   # back to pct-points for formula coefficient

    orb_margin = float((orb1 or 0) - (orb2 or 0))
    to_margin  = float((to1  or 0) - (to2  or 0))

    # PBP features (from fetch_live_game_states_with_pbp enrichment)
    pbp_available   = snapshot.get("pbp_available", False)

    # ESPN live scoreboard omits ORB and TO — fall back to PBP-derived counts
    if pbp_available:
        if orb1 is None:
            orb1 = snapshot.get("orb_home")
        if orb2 is None:
            orb2 = snapshot.get("orb_away")
        if to1 is None:
            to1 = snapshot.get("to_home")
        if to2 is None:
            to2 = snapshot.get("to_away")
    pace_live       = snapshot.get("pace_live", 0.0) or 0.0
    momentum_5pos   = snapshot.get("momentum_5pos", 0.0) or 0.0
    momentum_10pos  = snapshot.get("momentum_10pos", 0.0) or 0.0
    possessions     = snapshot.get("possessions_home", 0) + snapshot.get("possessions_away", 0)

    # FT and foul features from PBP parser
    ft_made_diff = snapshot.get("ft_made_diff")  # ftm_home - ftm_away (team1 perspective)
    foul_diff    = snapshot.get("foul_diff")      # fouls_away - fouls_home (team1 perspective)

    # Use PBP-computed stats if available (more accurate than ESPN box score)
    if pbp_available and snapshot.get("efg_diff") is not None:
        efg_diff   = float(snapshot["efg_diff"]) * 100.0  # convert decimal to pct-points
    if pbp_available:
        if snapshot.get("orb_margin") is not None:
            orb_margin = float(snapshot["orb_margin"])
        if snapshot.get("to_margin") is not None:
            to_margin = float(snapshot["to_margin"])

    # --- Projection ---
    # The retrained model covers all game states (trained on 19 timepoints per game).
    # Use it throughout the game when PBP data is available.
    # When no PBP, use the formula (as before).
    import numpy as np
    from src.utils.db import query_df as _qdf

    uses_trained = False
    flipped = False  # set True if we reorient to better-seed perspective below
    _pregame_spread_display = pregame_spread  # preserve original for breakdown display
    if _use_trained and _live_model is not None:
        try:
            # Torvik ratings for barthag/adj_o/adj_d differentials
            _ratings = _qdf(
                "SELECT team, barthag, adj_o, adj_d FROM torvik_ratings WHERE year=? AND team IN (?,?)",
                params=[year, team1, team2],
            )
            _r1 = _ratings[_ratings["team"] == team1].iloc[0].to_dict() if not _ratings[_ratings["team"] == team1].empty else {}
            _r2 = _ratings[_ratings["team"] == team2].iloc[0].to_dict() if not _ratings[_ratings["team"] == team2].empty else {}
            barthag_diff = (_r1.get("barthag", 0) or 0) - (_r2.get("barthag", 0) or 0)
            adj_o_diff   = (_r1.get("adj_o", 0) or 0)   - (_r2.get("adj_o", 0) or 0)
            adj_d_diff   = (_r1.get("adj_d", 0) or 0)   - (_r2.get("adj_d", 0) or 0)

            # Seed/round from bracket
            _seeds = _qdf(
                "SELECT team, seed FROM tournament_bracket WHERE year=? AND team IN (?,?)",
                params=[year, team1, team2],
            )
            _s1 = int(_seeds[_seeds["team"] == team1]["seed"].iloc[0]) if not _seeds[_seeds["team"] == team1].empty else 8
            _s2 = int(_seeds[_seeds["team"] == team2]["seed"].iloc[0]) if not _seeds[_seeds["team"] == team2].empty else 8
            seed_diff = _s1 - _s2

            # ── Orientation: flip to better-seed perspective ──────────────────────
            # The model was trained with team1 = better seed (lower seed number).
            # If team2 has the lower seed, flip all signed features so the model
            # sees the familiar orientation; we negate the output back afterward.
            flipped = _s2 < _s1
            if flipped:
                current_margin  = -current_margin
                efg_diff        = -efg_diff
                orb_margin      = -orb_margin
                to_margin       = -to_margin
                momentum_5pos   = -momentum_5pos
                momentum_10pos  = -momentum_10pos
                pregame_spread  = -pregame_spread
                if ft_made_diff is not None:
                    ft_made_diff = -ft_made_diff
                if foul_diff is not None:
                    foul_diff = -foul_diff
                barthag_diff    = -barthag_diff
                adj_o_diff      = -adj_o_diff
                adj_d_diff      = -adj_d_diff
                seed_diff       = _s2 - _s1   # better_seed# - worse_seed# (always negative)

            h1_combined    = (score1 or 0) + (score2 or 0)
            pace_surprise  = h1_combined - (projected_total / 2.0)
            margin_surprise = current_margin - (pregame_spread * (time_elapsed / 40.0))
            # margin_surprise is auto-correct since current_margin/pregame_spread are flipped
            # pace_surprise is team-agnostic (total scoring vs projected total) — do NOT flip

            # Full 21-feature vector matching LIVE_FEATURES order
            feat_vec = [
                pregame_spread,                                          # pregame_spread
                current_margin,                                          # h1_margin
                float(h1_combined),                                      # h1_combined
                time_elapsed / 40.0,                                     # time_elapsed_pct
                time_remaining / 40.0,                                   # time_remaining_pct
                efg_diff if efg_diff != 0.0 else float("nan"),                      # efg_pct_diff
                orb_margin if orb_margin != 0.0 else float("nan"),                  # orb_margin
                to_margin  if to_margin  != 0.0 else float("nan"),                  # to_margin
                float(ft_made_diff) if ft_made_diff is not None else float("nan"),  # ft_made_diff
                float(foul_diff)    if foul_diff    is not None else float("nan"),  # foul_diff
                pace_surprise,                                                       # pace_surprise
                margin_surprise,                                         # margin_surprise
                barthag_diff,                                            # barthag_diff
                adj_o_diff,                                              # adj_o_diff
                adj_d_diff,                                              # adj_d_diff
                float(seed_diff),                                        # seed_diff
                float(snapshot.get("round_number", 1) or 1),            # round_number
                float(pace_live) if pace_live else float("nan"),         # pace_live
                float(momentum_5pos),                                    # momentum_5pos
                float(momentum_10pos),                                   # momentum_10pos
                float(possessions) if possessions > 0 else float("nan"),  # possessions
            ]
            features = np.array([feat_vec])
            raw = float(_live_model.predict(features)[0])
            projected_margin = float(_live_cal.predict([raw])[0]) if _live_cal is not None else raw
            # Flip output back to team1 perspective if we reoriented above
            if flipped:
                projected_margin = -projected_margin
            uses_trained = True
        except Exception as _ex:
            projected_margin = formula_projected_margin(
                current_margin, pregame_spread, time_elapsed, time_remaining,
                efg_diff, orb_margin, to_margin,
            )
            if flipped:
                projected_margin = -projected_margin
    else:
        projected_margin = formula_projected_margin(
            current_margin, pregame_spread, time_elapsed, time_remaining,
            efg_diff, orb_margin, to_margin,
        )

    # PBP momentum layer on top of formula (only when formula is used + PBP available)
    pbp_adj = 0.0
    if not uses_trained and pbp_available:
        w_model = time_remaining / 40.0
        efg_adj_pbp = (efg_diff  * 0.15 * w_model) if efg_diff else 0.0
        orb_adj_pbp = (orb_margin * 0.25 * w_model) if orb_margin else 0.0
        to_adj_pbp  = (to_margin  * 0.40 * w_model) if to_margin else 0.0
        mom_adj     = _momentum_adjustment(momentum_5pos, momentum_10pos, time_remaining)
        pbp_adj     = mom_adj  # eFG/ORB/TO already in formula; add momentum on top
        projected_margin += pbp_adj

    # Breakdown components (for display)
    w_score  = time_elapsed   / 40.0
    w_model  = time_remaining / 40.0
    time_weight_adj = (current_margin * w_score + pregame_spread * w_model) - pregame_spread
    efg_adj_val = efg_diff  * 0.15 * w_model
    orb_adj_val = orb_margin * 0.25 * w_model
    to_adj_val  = to_margin  * 0.40 * w_model
    mom_adj_val = _momentum_adjustment(momentum_5pos, momentum_10pos, time_remaining)

    # Edge vs live market spread.
    # live_spread is in betting convention (negative = home/team1 favored).
    # projected_margin is in model convention (positive = team1 wins).
    # To compare apples-to-apples: model edge = projected_margin - (-live_spread)
    #   = projected_margin + live_spread
    # Example: projected_margin=+28, live_spread=-37.5 → edge = 28 + (-37.5) = -9.5
    #   (model less bullish than market → no bet on team1 spread)
    edge = None
    if live_spread is not None:
        edge = projected_margin + live_spread

    # --- Tier and sizing ---
    # CRITICAL: suppress Strong/Value signals when time_remaining < 5 minutes;
    # force Pass in final 3 minutes regardless of edge.
    if time_remaining < 3.0:
        tier_label, tier_emoji = "Pass", "⚪"
        cov_prob   = None
        kelly_pct  = None
        bet_team   = None
    elif time_remaining < 5.0 and edge is not None:
        # Allow Lean only — cap tier at Lean
        cov_prob = coverage_probability(projected_margin, live_spread) if live_spread is not None else None
        raw_tier, raw_emoji = bet_tier(edge, cov_prob) if (cov_prob is not None and edge is not None) else ("Pass", "⚪")
        # Demote Strong/Value to Lean
        if raw_tier in ("Strong", "Value"):
            tier_label, tier_emoji = "Lean", "📊"
        else:
            tier_label, tier_emoji = raw_tier, raw_emoji
        kelly_pct = half_kelly(cov_prob) if cov_prob is not None else None
        bet_team = team1 if (edge is not None and edge >= 0) else team2
    else:
        if live_spread is not None and edge is not None:
            cov_prob = coverage_probability(projected_margin, live_spread)
            tier_label, tier_emoji = bet_tier(edge, cov_prob)
            kelly_pct = half_kelly(cov_prob)
            bet_team = team1 if edge >= 0 else team2
        else:
            cov_prob   = None
            tier_label, tier_emoji = "Pass", "⚪"
            kelly_pct  = None
            bet_team   = None

    return {
        "team1":              team1,
        "team2":              team2,
        "score1":             score1,
        "score2":             score2,
        "projected_margin":   round(projected_margin, 2),
        "live_market_spread": live_spread,
        "edge":               round(edge, 2) if edge is not None else None,
        "tier":               tier_label,
        "tier_emoji":         tier_emoji,
        "kelly_pct":          kelly_pct,
        "bet_team":           bet_team,
        "game_status":        game_status,
        "time_elapsed":       time_elapsed,
        "time_remaining":     time_remaining,
        "breakdown": {
            "pregame_spread":   _pregame_spread_display,
            "time_weight_adj":  round(time_weight_adj, 2),
            "efg_adj":          round(efg_adj_val, 2),
            "orb_adj":          round(orb_adj_val, 2),
            "to_adj":           round(to_adj_val, 2),
            "momentum_adj":     round(mom_adj_val, 2),
        },
        "stats": {
            "efg_pct1":    efg_pct1,
            "efg_pct2":    efg_pct2,
            "efg_season1": efg_season1,
            "efg_season2": efg_season2,
            "orb1":        orb1,
            "orb2":        orb2,
            "to1":         to1,
            "to2":         to2,
        },
        "pbp": {
            "available":     pbp_available,
            "momentum_5pos": round(momentum_5pos, 1),
            "momentum_10pos": round(momentum_10pos, 1),
            "pace_live":     round(pace_live, 1) if pace_live else None,
            "run_home":      snapshot.get("run_home", 0),
            "run_away":      snapshot.get("run_away", 0),
            "possessions":   possessions,
        },
        "uses_trained_model": uses_trained,
    }
