import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import streamlit as st
from datetime import datetime

from src.utils.config import TOURNAMENT_YEARS
from src.ingest.live_game_state import STATUS_FINAL, STATUS_IN_PROGRESS, STATUS_HALFTIME

st.set_page_config(page_title="Live Model", page_icon="🔴", layout="wide")

# ── Auto-refresh every 60 seconds ─────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()
    st.rerun()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 600;
    padding: 6px 16px;
    color: #555 !important;
}
.stTabs [aria-selected="true"] {
    color: #e74c3c !important;
    border-bottom: 2px solid #e74c3c !important;
}
button, .stButton > button { min-height: 44px; }
</style>
""", unsafe_allow_html=True)

TIER_COLORS = {
    "Strong": ("#1a472a", "#2ecc71"),
    "Value":  ("#1e3a5f", "#3498db"),
    "Lean":   ("#3d2b1f", "#f39c12"),
    "Pass":   ("#2a2a2a", "#7f8c8d"),
}

current_year = TOURNAMENT_YEARS[-1]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    bankroll = st.number_input(
        "Bankroll ($)", min_value=50, max_value=1_000_000, value=1000, step=100
    )
    sizing = st.radio("Kelly sizing", ["Half Kelly", "Full Kelly", "Flat ($100)"])
    min_edge = st.slider("Min |edge| (pts)", 0.0, 15.0, 0.0, 0.5)
    hide_pass = st.checkbox("Hide Pass-tier bets", value=False)
    st.divider()
    st.caption("Pre-game spreads are auto-loaded from the model. Use overrides below only if needed.")
    pregame_spread_input = st.number_input(
        "Pregame spread override (team1, + = fav)",
        value=0.0,
        step=0.5,
        help="Leave at 0 to use auto model spread. Only override if you want to force a specific value.",
    )
    pregame_total_input = st.number_input(
        "Pregame total override", value=0.0, step=1.0,
        help="Leave at 0 to use auto model total.",
    )
    st.divider()
    st.markdown("### 📖 When to Bet")
    st.markdown("""
**Most reliable: Halftime**
The retrained XGBoost model covers all game states (trained at 19 timepoints per game).
Momentum, pace, and shooting splits from live PBP enrich every projection.

**Run alerts**
⚡ Shown above card when a team has a 9+ point unanswered run.

**Tiers**
- 🔥 **Strong** (edge ≥7, cov ≥58%) — bet it
- ✅ **Value** (edge ≥5, cov ≥56%) — worth a unit
- 📊 **Lean** (edge ≥3, cov ≥54%) — smaller bet or skip
- ⚪ **Pass** — no bet

**Hard limits (built in)**
- Under 5 min: Strong/Value auto-demoted to Lean
- Under 3 min: forced Pass, no bets

**Backtest note**
Cover rates are an upper bound — historical backtest used pre-game lines as a proxy for live halftime lines. Real halftime lines are sharper.
""")



# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_live_states() -> list:
    from src.ingest.live_game_state import fetch_live_game_states_with_pbp
    return fetch_live_game_states_with_pbp()


@st.cache_data(ttl=300)
def get_pregame_projections(team_pairs: tuple, year: int) -> dict:
    """
    Fetch pre-game model spread and total for each (team1, team2) pair.
    Cached for 5 minutes — pre-game values don't change once a game starts.
    Returns {(team1, team2): {"spread": float, "total": float}}.
    """
    from src.model.predict import project_game
    result = {}
    for team1, team2 in team_pairs:
        try:
            proj = project_game(team1, team2, round_num=1, year=year)
            if "error" not in proj:
                result[(team1, team2)] = {
                    "spread": proj.get("projected_spread", 0.0),
                    "total": proj.get("projected_total", 140.0),
                }
        except Exception:
            pass
    return result


def build_live_predictions(
    states: list,
    pregame_spread_override: float,
    projected_total_override: float,
    year: int,
) -> list:
    """
    Run live predictions for all game states.
    Auto-looks up pre-game model spread/total per game.
    Override inputs (sidebar) are used only when non-zero.
    """
    from src.model.live_predict import project_game_live

    # Fetch pre-game projections for all live games (cached 5 min)
    team_pairs = tuple((s["team1"], s["team2"]) for s in states)
    pregame_proj = get_pregame_projections(team_pairs, year)

    preds = []
    for snap in states:
        team1 = snap["team1"]
        team2 = snap["team2"]

        # Use per-game model spread/total; fall back to sidebar override if lookup failed
        game_proj = pregame_proj.get((team1, team2), {})
        pregame_spread = (
            pregame_spread_override if pregame_spread_override != 0.0
            else game_proj.get("spread", 0.0)
        )
        projected_total = (
            projected_total_override if projected_total_override != 0.0
            else game_proj.get("total", 140.0)
        )

        try:
            pred = project_game_live(
                team1=team1,
                team2=team2,
                snapshot=snap,
                pregame_spread=pregame_spread,
                projected_total=projected_total,
                year=year,
            )
            pred["_snap"] = snap
            pred["_pregame_spread"] = pregame_spread
            preds.append(pred)
        except Exception as e:
            preds.append({
                "team1": team1,
                "team2": team2,
                "error": str(e),
                "_snap": snap,
                "game_status": snap.get("game_status", ""),
                "tier": "Pass",
                "edge": None,
            })
    return preds


# ── Helpers ────────────────────────────────────────────────────────────────────
def _bet_size(kelly_pct, bankroll_, sizing_):
    if kelly_pct is None:
        return 0
    if sizing_ == "Full Kelly":
        return round(bankroll_ * kelly_pct * 2)  # kelly_pct is already half-kelly
    elif sizing_ == "Half Kelly":
        return round(bankroll_ * kelly_pct)
    return 100


def _format_spread(val, team):
    if val is None:
        return "—"
    return f"{team} {val:+.1f}"


def _stat_color(val1, val2, bet_team, team1, team2):
    """Green if stat favors the bet team, red if not, grey if unclear."""
    if val1 is None or val2 is None or bet_team is None:
        return "#aaa"
    diff = val1 - val2
    if bet_team == team1:
        return "#2ecc71" if diff > 0 else "#e74c3c"
    else:
        return "#2ecc71" if diff < 0 else "#e74c3c"


def render_game_card(pred: dict, bankroll_: int, sizing_: str):
    """Render a single full-width game card using HTML/CSS."""
    if "error" in pred:
        st.warning(f"{pred.get('team1','?')} vs {pred.get('team2','?')}: {pred['error']}")
        return

    # Run alert — show above card when unanswered run >= 9
    pbp_data = pred.get("pbp", {})
    run_home = pbp_data.get("run_home", 0) or 0
    run_away = pbp_data.get("run_away", 0) or 0
    _team1 = pred.get("team1", "")
    _team2 = pred.get("team2", "")
    if run_home >= 9:
        st.warning(f"⚡ {_team1} on a {run_home}-0 run")
    elif run_away >= 9:
        st.warning(f"⚡ {_team2} on a {run_away}-0 run")

    snap           = pred.get("_snap", {})
    team1          = pred["team1"]
    team2          = pred["team2"]
    score1         = snap.get("score1", 0)
    score2         = snap.get("score2", 0)
    period         = snap.get("period", 1)
    clock          = snap.get("clock_display", "—")
    status         = pred.get("game_status", "")
    tier           = pred.get("tier", "Pass")
    tier_emoji     = pred.get("tier_emoji", "⚪")
    edge           = pred.get("edge")
    live_mkt       = pred.get("live_market_spread")
    proj_margin    = pred.get("projected_margin", 0.0)
    bet_team       = pred.get("bet_team")
    kelly_pct      = pred.get("kelly_pct")
    uses_trained   = pred.get("uses_trained_model", False)

    bg, accent = TIER_COLORS.get(tier, ("#2a2a2a", "#7f8c8d"))

    # Period label
    if status == STATUS_HALFTIME:
        period_label = "Halftime"
    elif status == STATUS_FINAL:
        period_label = "Final"
    elif period == 1:
        period_label = "1st Half"
    elif period == 2:
        period_label = "2nd Half"
    else:
        period_label = f"OT{period - 2}"

    # Spread display strings
    model_spread_display  = _format_spread(-proj_margin, team1) if proj_margin is not None else "—"
    market_spread_display = _format_spread(live_mkt, team1) if live_mkt is not None else "No line"
    edge_display          = f"{edge:+.1f} pts" if edge is not None else "—"
    bet_size              = _bet_size(kelly_pct, bankroll_, sizing_)
    bet_str               = f"${bet_size:,} on {bet_team}" if bet_team and bet_size > 0 else "—"

    # Stats
    stats    = pred.get("stats", {})
    efg1     = stats.get("efg_pct1")
    efg2     = stats.get("efg_pct2")
    efg_s1   = stats.get("efg_season1")
    efg_s2   = stats.get("efg_season2")
    orb1     = stats.get("orb1")
    orb2     = stats.get("orb2")
    to1      = stats.get("to1")
    to2      = stats.get("to2")

    def _fmt(v, fmt=".0f"):
        return f"{v:{fmt}}" if v is not None else "—"

    efg_color = _stat_color(efg1, efg2, bet_team, team1, team2)
    orb_color = _stat_color(orb1, orb2, bet_team, team1, team2)
    # Fewer turnovers is better — flip the sign for color
    to_color  = _stat_color(-(to1 or 0), -(to2 or 0), bet_team, team1, team2)

    def _stat_row(label, v1, v2, color, fmt=".0f"):
        f1 = f"{v1:{fmt}}" if v1 is not None else "—"
        f2 = f"{v2:{fmt}}" if v2 is not None else "—"
        pct = "%" if fmt == ".1f" else ""
        return (
            f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
            f"border-bottom:1px solid #333;font-size:0.8rem'>"
            f"<span style='color:#aaa;flex:1'>{label}</span>"
            f"<span style='color:{color};flex:1;text-align:center'>{f1}{pct}</span>"
            f"<span style='color:{color};flex:1;text-align:center'>{f2}{pct}</span>"
            f"</div>"
        )

    # PBP momentum/pace
    mom5       = pbp_data.get("momentum_5pos", 0.0) or 0.0
    pace_live  = pbp_data.get("pace_live")

    momentum_html = ""
    if pbp_data.get("available") and mom5 != 0.0:
        mom_team = team1 if mom5 > 0 else team2
        momentum_html = (
            f"<div style='font-size:0.75rem;color:#f39c12;margin-top:4px'>"
            f"Last 5 poss: {mom_team} +{abs(mom5):.0f} pts"
            f"</div>"
        )

    stats_html = (
        f"<div style='margin-top:10px'>"
        f"<div style='display:flex;justify-content:space-between;padding:2px 0;font-size:0.75rem'>"
        f"<span style='color:#aaa;flex:1'>Stat</span>"
        f"<span style='color:#aaa;flex:1;text-align:center'>{team1}</span>"
        f"<span style='color:#aaa;flex:1;text-align:center'>{team2}</span>"
        f"</div>"
        + _stat_row("eFG% (live)", efg1, efg2, efg_color, ".1f")
        + _stat_row("eFG% (season)", efg_s1, efg_s2, "#ccc", ".1f")
        + _stat_row("Off Reb", orb1, orb2, orb_color)
        + _stat_row("Turnovers", to1, to2, to_color)
        + f"</div>"
        + momentum_html
    )

    pace_html = ""
    if pace_live and proj_margin is not None:
        proj_total = abs(proj_margin) + (score1 + score2)  # rough projection
        pace_html = f"<div style='font-size:0.75rem;color:#aaa;margin-top:4px'>Pace: {pace_live:.0f} proj pts/game</div>"

    card_html = f"""
<div style='background:{bg};border-left:4px solid {accent};border-radius:8px;padding:16px;margin-bottom:12px'>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div>
      <span style='font-size:1.1rem;font-weight:bold;color:#fff'>{team1} vs {team2}</span><br>
      <span style='color:#aaa;font-size:0.8rem'>{period_label} &mdash; {clock}</span>
    </div>
    <div style='font-size:1.4rem;font-weight:bold;color:#fff'>{score1} &ndash; {score2}</div>
    <div style='background:{accent};color:#fff;padding:4px 10px;border-radius:20px;font-size:0.85rem;font-weight:bold'>{tier_emoji} {tier}</div>
  </div>
  <div style='display:flex;gap:16px;margin-top:12px;flex-wrap:wrap'>
    <div>
      <div style='color:#aaa;font-size:0.7rem'>MODEL</div>
      <div style='font-size:1rem;font-weight:bold;color:#fff'>{model_spread_display}</div>
    </div>
    <div>
      <div style='color:#aaa;font-size:0.7rem'>MARKET</div>
      <div style='font-size:1rem;color:#ddd'>{market_spread_display}</div>
    </div>
    <div>
      <div style='color:#aaa;font-size:0.7rem'>EDGE</div>
      <div style='font-size:1rem;font-weight:bold;color:#2ecc71'>{edge_display}</div>
    </div>
    <div>
      <div style='color:#aaa;font-size:0.7rem'>BET</div>
      <div style='font-size:1rem;font-weight:bold;color:#fff'>{bet_str}</div>
    </div>
  </div>
  {stats_html}
  {pace_html}
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)

    # Breakdown expander
    breakdown = pred.get("breakdown", {})
    pg_sp   = breakdown.get("pregame_spread", 0.0)
    tw_adj  = breakdown.get("time_weight_adj", 0.0)
    eg_adj  = breakdown.get("efg_adj", 0.0)
    ob_adj  = breakdown.get("orb_adj", 0.0)
    ta_adj  = breakdown.get("to_adj", 0.0)
    mom_adj = breakdown.get("momentum_adj", 0.0)
    mkt_display = _format_spread(live_mkt, team1) if live_mkt is not None else "No line"
    edge_line   = f"{edge:+.1f} pts" if edge is not None else "—"

    pregame_spread_used = pred.get("_pregame_spread", pg_sp)
    breakdown_text = (
        f"Pre-game model:    {team1} {pregame_spread_used:+.1f}  (auto)\n"
        f"Time-weight adj:   {tw_adj:+.1f}\n"
        f"eFG% adj:          {eg_adj:+.1f}\n"
        f"Rebound adj:       {ob_adj:+.1f}\n"
        f"TO adj:            {ta_adj:+.1f}\n"
        f"Momentum (5 poss): {mom_adj:+.1f}\n"
        f"────────────────\n"
        f"Live model:        {team1} {proj_margin:+.1f}\n"
        f"Market:            {mkt_display}\n"
        f"Edge:              {edge_line}"
    )

    with st.expander("📊 Spread Breakdown"):
        st.code(breakdown_text, language=None)
        if not uses_trained:
            st.caption("📡 Using formula + PBP momentum — retrained model pending backfill")

    st.divider()


# ── Page header ────────────────────────────────────────────────────────────────
st.title("🔴 Live Model")

hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    next_refresh = max(0, 60 - int(time.time() - st.session_state.last_refresh))
    st.caption(
        f"Auto-refreshes every 60s · Next refresh in ~{next_refresh}s · "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )
with hcol2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()

# ── Load and predict ───────────────────────────────────────────────────────────
with st.spinner("Fetching live games from ESPN..."):
    raw_states = get_live_states()

if not raw_states:
    st.info("No live NCAAB games found right now. Check back during tournament game windows.")
    st.stop()

with st.spinner("Running live model..."):
    all_preds = build_live_predictions(
        states=raw_states,
        pregame_spread_override=pregame_spread_input,
        projected_total_override=pregame_total_input,
        year=current_year,
    )

# Bucket by status
in_progress = [
    p for p in all_preds
    if p.get("game_status") in (STATUS_IN_PROGRESS, STATUS_HALFTIME)
]
completed = [
    p for p in all_preds
    if p.get("game_status") == STATUS_FINAL
]

# Filter helper
def _passes_filter(pred):
    if "error" in pred:
        return True
    tier  = pred.get("tier", "Pass")
    edge  = pred.get("edge")
    abs_e = abs(edge) if edge is not None else 0.0
    if hide_pass and tier == "Pass":
        return False
    if min_edge > 0 and abs_e < min_edge:
        return False
    return True

in_progress_filtered = sorted(
    [p for p in in_progress if _passes_filter(p)],
    key=lambda p: abs(p.get("edge") or 0),
    reverse=True,
)

# Active alerts: non-Pass tiers with |edge| >= 3
alerts = [
    p for p in in_progress
    if p.get("tier") in ("Strong", "Value", "Lean")
    and abs(p.get("edge") or 0) >= 3.0
]

# ── Section 1: Active Alerts ───────────────────────────────────────────────────
if alerts:
    st.subheader("🚨 Active Alerts")
    st.caption(f"{len(alerts)} game(s) with |edge| ≥ 3 — actionable now")
    for pred in sorted(alerts, key=lambda p: abs(p.get("edge") or 0), reverse=True):
        render_game_card(pred, bankroll, sizing)

# ── Section 2: In Progress ─────────────────────────────────────────────────────
st.subheader("📺 In Progress")
if not in_progress_filtered:
    if in_progress:
        st.info("Live games found but none match current filters.")
    else:
        st.info("No live games in progress right now.")
else:
    for pred in in_progress_filtered:
        render_game_card(pred, bankroll, sizing)

# ── Section 3: Completed Today ─────────────────────────────────────────────────
if completed:
    st.subheader("✅ Completed Today")
    for pred in completed:
        snap        = pred.get("_snap", {})
        team1       = pred.get("team1", "?")
        team2       = pred.get("team2", "?")
        score1      = snap.get("score1", 0)
        score2      = snap.get("score2", 0)
        final_margin = score1 - score2
        proj        = pred.get("projected_margin", 0.0)
        st.markdown(
            f"<div style='background:#1a1a1a;border-left:4px solid #555;"
            f"border-radius:6px;padding:12px;margin-bottom:8px'>"
            f"<span style='font-weight:bold;color:#fff'>{team1} vs {team2}</span> "
            f"<span style='color:#aaa;font-size:0.9rem'>Final: {score1}–{score2} "
            f"(margin: {final_margin:+.0f}, model proj: {proj:+.1f})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
