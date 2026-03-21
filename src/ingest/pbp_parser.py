"""
pbp_parser.py
-------------
Parses ESPN plays[] array into structured rows and computes game state
at arbitrary time checkpoints.

Two public functions:
  parse_plays(plays, home_team_id)  -> list[dict]
  compute_game_state_at(plays_parsed, at_time_elapsed, home_team, away_team) -> dict
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Clock helpers
# ---------------------------------------------------------------------------

def _clock_to_secs(display: str) -> int:
    """'12:34' -> 754 seconds remaining in period."""
    try:
        parts = display.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def _time_elapsed(period: int, clock_secs: int) -> float:
    """
    Convert period + clock_secs into minutes elapsed since tip-off.
    period 1: elapsed = (20*60 - clock_secs) / 60
    period 2: elapsed = 20 + (20*60 - clock_secs) / 60
    """
    half_elapsed = (20 * 60 - clock_secs) / 60.0
    return half_elapsed if period == 1 else 20.0 + half_elapsed


# ---------------------------------------------------------------------------
# Play parser
# ---------------------------------------------------------------------------

def parse_plays(plays: list, home_team_id: str) -> List[Dict]:
    """
    Parses ESPN plays[] array into structured rows.

    Classification priority (first match wins):
      MADE_2      scoringPlay=True AND scoreValue=2 AND 'Free Throw' not in text
      MADE_3      scoringPlay=True AND scoreValue=3
      MADE_FT     scoringPlay=True AND 'Free Throw' in text
      MISSED_FG   shooting=True AND scoringPlay=False AND 'Free Throw' not in text
      MISSED_FT   shooting=True AND 'Free Throw' in text AND scoringPlay=False
      TURNOVER    'Turnover' in type.text OR 'turnover' in text.lower()
      REBOUND_OFF 'Offensive Rebound' in text
      REBOUND_DEF 'Defensive Rebound' in text
      FOUL        'Foul' in type.text
      UNKNOWN     fallback

    Unknown plays are stored but excluded from feature computation.
    """
    home_id = str(home_team_id)
    result = []

    for play in plays:
        try:
            play_id   = str(play.get("id", ""))
            period    = int(play.get("period", {}).get("number", 1))
            clock_str = play.get("clock", {}).get("displayValue", "0:00")
            clock_secs = _clock_to_secs(clock_str)
            time_el    = _time_elapsed(period, clock_secs)

            text       = play.get("text", "") or ""
            type_text  = (play.get("type") or {}).get("text", "") or ""
            scoring    = bool(play.get("scoringPlay", False))
            score_val  = int(play.get("scoreValue", 0) or 0)
            shooting   = bool(play.get("shooting", False))

            team_obj = play.get("team") or {}
            team_id  = str(team_obj.get("id", ""))
            is_home  = team_id == home_id

            home_score = int(play.get("homeScore", 0) or 0)
            away_score = int(play.get("awayScore", 0) or 0)
            margin     = home_score - away_score

            # Classification
            is_fg_attempt = False
            is_fg_made    = False
            is_3pt        = False

            # ESPN type.text for free throws is "MadeFreeThrow" (camelCase, no space).
            # shooting flag is None/False on FT plays. Use type_text as primary signal.
            is_ft = "FreeThrow" in type_text or "free throw" in text.lower()

            if scoring and score_val == 2 and not is_ft:
                event_type    = "MADE_2"
                is_fg_attempt = True
                is_fg_made    = True

            elif scoring and score_val == 3:
                event_type    = "MADE_3"
                is_fg_attempt = True
                is_fg_made    = True
                is_3pt        = True

            elif is_ft and scoring:
                event_type = "MADE_FT"

            elif is_ft and not scoring:
                # Missed FT: ESPN type.text is still "MadeFreeThrow" but scoringPlay=False
                event_type = "MISSED_FT"

            elif shooting and not scoring and not is_ft:
                event_type    = "MISSED_FG"
                is_fg_attempt = True

            elif "Turnover" in type_text or "turnover" in text.lower():
                event_type = "TURNOVER"

            elif "Offensive Rebound" in text:
                event_type = "REBOUND_OFF"

            elif "Defensive Rebound" in text:
                event_type = "REBOUND_DEF"

            elif "Foul" in type_text:
                event_type = "FOUL"

            else:
                event_type = "UNKNOWN"

            result.append({
                "play_id":      play_id,
                "period":       period,
                "clock_secs":   clock_secs,
                "time_elapsed": time_el,
                "event_type":   event_type,
                "team":         "home" if is_home else "away",
                "score_value":  score_val,
                "home_score":   home_score,
                "away_score":   away_score,
                "margin":       margin,
                "is_fg_attempt": is_fg_attempt,
                "is_fg_made":    is_fg_made,
                "is_3pt":        is_3pt,
                "raw_text":      text,
            })

        except Exception:
            continue

    return result


# ---------------------------------------------------------------------------
# Game state aggregator
# ---------------------------------------------------------------------------

def compute_game_state_at(
    plays_parsed: List[Dict],
    at_time_elapsed: float,
    home_team: str,
    away_team: str,
) -> Dict:
    """
    Filter plays to those at or before at_time_elapsed and aggregate
    into a game state dict.

    Returns a dict with all fields listed in the spec; numeric fields
    that cannot be computed (e.g. eFG% when FGA=0) are set to None.
    """
    plays = [p for p in plays_parsed if p["time_elapsed"] <= at_time_elapsed]

    time_remaining = max(0.0, 40.0 - at_time_elapsed)

    if not plays:
        return {
            "time_elapsed":     at_time_elapsed,
            "time_remaining":   time_remaining,
            "score_home":       0,
            "score_away":       0,
            "current_margin":   0.0,
            "efg_home":         None,
            "efg_away":         None,
            "efg_diff":         None,
            "orb_home":         0,
            "orb_away":         0,
            "orb_margin":       0,
            "to_home":          0,
            "to_away":          0,
            "to_margin":        0,
            "possessions_home": 0,
            "possessions_away": 0,
            "pace_live":        0.0,
            "run_home":         0,
            "run_away":         0,
            "momentum_5pos":    0.0,
            "momentum_10pos":   0.0,
        }

    # Score from the last play
    last = plays[-1]
    score_home    = last["home_score"]
    score_away    = last["away_score"]
    current_margin = float(score_home - score_away)

    # Shooting stats
    fgm_home = fga_home = tpm_home = 0
    fgm_away = fga_away = tpm_away = 0
    to_home = to_away = 0
    orb_home = orb_away = 0
    ftm_home = ftm_away = 0
    fta_home = fta_away = 0
    fouls_home = fouls_away = 0

    # Possession counts (possession ends on MADE_2, MADE_3, TURNOVER, REBOUND_DEF)
    poss_home = poss_away = 0

    for p in plays:
        et = p["event_type"]
        tm = p["team"]

        if et in ("MADE_2", "MADE_3", "MISSED_FG"):
            if tm == "home":
                fga_home += 1
                if p["is_fg_made"]:
                    fgm_home += 1
                if p["is_3pt"]:
                    tpm_home += 1
            else:
                fga_away += 1
                if p["is_fg_made"]:
                    fgm_away += 1
                if p["is_3pt"]:
                    tpm_away += 1

        if et in ("MADE_2", "MADE_3", "TURNOVER", "REBOUND_DEF"):
            if tm == "home":
                poss_home += 1
            else:
                poss_away += 1

        if et == "TURNOVER":
            if tm == "home":
                to_home += 1
            else:
                to_away += 1

        if et == "REBOUND_OFF":
            if tm == "home":
                orb_home += 1
            else:
                orb_away += 1

        if et == "MADE_FT":
            if tm == "home":
                ftm_home += 1
                fta_home += 1
            else:
                ftm_away += 1
                fta_away += 1

        if et == "MISSED_FT":
            if tm == "home":
                fta_home += 1
            else:
                fta_away += 1

        if et == "FOUL":
            if tm == "home":
                fouls_home += 1
            else:
                fouls_away += 1

    # eFG% = (FGM + 0.5 * 3PM) / FGA
    efg_home = ((fgm_home + 0.5 * tpm_home) / fga_home) if fga_home > 0 else None
    efg_away = ((fgm_away + 0.5 * tpm_away) / fga_away) if fga_away > 0 else None
    efg_diff = (efg_home - efg_away) if (efg_home is not None and efg_away is not None) else None

    orb_margin = orb_home - orb_away
    # to_margin: positive = home protecting ball (away turning it over more)
    to_margin  = to_away - to_home

    # Pace: (possessions_home + possessions_away) / time_elapsed * 40
    total_poss = poss_home + poss_away
    pace_live  = (total_poss / at_time_elapsed * 40.0) if at_time_elapsed > 0 else 0.0

    # Current unanswered run — look backward through scoring plays
    scoring_plays = [
        p for p in plays
        if p["event_type"] in ("MADE_2", "MADE_3", "MADE_FT")
    ]
    run_home = run_away = 0
    if scoring_plays:
        run_team = scoring_plays[-1]["team"]
        run_pts  = scoring_plays[-1]["score_value"]
        for sp in reversed(scoring_plays[:-1]):
            if sp["team"] == run_team:
                run_pts += sp["score_value"]
            else:
                break
        if run_team == "home":
            run_home = run_pts
        else:
            run_away = run_pts

    # Momentum: net margin change over last 5 / 10 possession-ending events
    poss_end_plays = [
        p for p in plays
        if p["event_type"] in ("MADE_2", "MADE_3", "TURNOVER", "REBOUND_DEF")
    ]

    def _momentum(n: int) -> float:
        if len(poss_end_plays) < 2:
            return 0.0
        recent = poss_end_plays[-n:]
        if len(recent) < 2:
            return float(recent[-1]["margin"] if recent else 0)
        return float(recent[-1]["margin"] - recent[0]["margin"])

    momentum_5pos  = _momentum(5)
    momentum_10pos = _momentum(10)

    # FT and foul derived features
    # ft_made_diff: home FTM - away FTM (positive = home scoring more from the line)
    ft_made_diff = ftm_home - ftm_away
    # foul_diff: away fouls - home fouls (positive = home team in better foul situation)
    foul_diff    = fouls_away - fouls_home

    return {
        "time_elapsed":     at_time_elapsed,
        "time_remaining":   time_remaining,
        "score_home":       score_home,
        "score_away":       score_away,
        "current_margin":   current_margin,
        "efg_home":         efg_home,
        "efg_away":         efg_away,
        "efg_diff":         efg_diff,
        "orb_home":         orb_home,
        "orb_away":         orb_away,
        "orb_margin":       orb_margin,
        "to_home":          to_home,
        "to_away":          to_away,
        "to_margin":        to_margin,
        "ftm_home":         ftm_home,
        "ftm_away":         ftm_away,
        "fta_home":         fta_home,
        "fta_away":         fta_away,
        "fouls_home":       fouls_home,
        "fouls_away":       fouls_away,
        "ft_made_diff":     ft_made_diff,
        "foul_diff":        foul_diff,
        "possessions_home": poss_home,
        "possessions_away": poss_away,
        "pace_live":        pace_live,
        "run_home":         run_home,
        "run_away":         run_away,
        "momentum_5pos":    momentum_5pos,
        "momentum_10pos":   momentum_10pos,
    }
