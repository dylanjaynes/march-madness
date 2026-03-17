import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import datetime as _dt
from datetime import datetime, timezone, timedelta
import pytz
from pathlib import Path

_TOTAL_CAL_PATH = Path("models/total_model_calibrator.pkl")
_total_calibrated = _TOTAL_CAL_PATH.exists()

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.model.predict import (
    project_game, coverage_probability, kelly_fraction, half_kelly,
    bet_tier, season_label, data_as_of,
)
from src.utils.db import db_conn, query_df


st.set_page_config(page_title="Live Games", page_icon="📡", layout="wide")

# Custom CSS: fix tab text colors and general readability
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
</style>
""", unsafe_allow_html=True)

current_year = TOURNAMENT_YEARS[-1]

EST = pytz.timezone("US/Eastern")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    bankroll = st.number_input("Bankroll ($)", min_value=50, max_value=1_000_000,
                                value=1000, step=100)
    sizing = st.radio("Kelly sizing", ["Half Kelly", "Full Kelly", "Flat ($100)"])
    min_edge = st.slider("Min |edge| (pts)", 0.0, 15.0, 0.0, 0.5)
    hide_pass = st.checkbox("Hide Pass-tier bets", value=False)
    st.divider()
    round_context = st.selectbox(
        "Round context",
        [1, 2, 3, 4, 5, 6],
        format_func=lambda r: ROUND_NAMES.get(r, f"R{r}"),
        help="R64 works best for most games.",
    )
    st.divider()
    today = _dt.date.today()
    date_filter = st.sidebar.radio(
        "Show games",
        ["Today", "Tomorrow", "All upcoming"],
        index=2,
    )


# ── Bracket seed lookup (seeds may be NULL in torvik_ratings for current year) ──
@st.cache_data(ttl=300)
def load_bracket_seed_map(year: int) -> dict:
    """Return {team_name: seed} from tournament_bracket table."""
    df = query_df(
        "SELECT team, seed FROM tournament_bracket WHERE year=? ORDER BY seed",
        params=[year],
    )
    if df.empty:
        return {}
    return dict(zip(df["team"], df["seed"].astype(int)))


# ── Odds: fetch, store pre-game snapshots, load pregame reference ──────────────
@st.cache_data(ttl=120)
def fetch_live_odds():
    from src.ingest.odds import fetch_current_games
    return fetch_current_games()


def _store_pregame_snapshot(odds_df: pd.DataFrame) -> None:
    """
    Persist current lines to odds_history for games that haven't tipped yet.
    This builds a pre-game line archive that survives page refreshes.
    Uses INSERT OR IGNORE so the *first* snapshot for each (game_id, timestamp)
    is preserved, giving us a true opening-line reference.
    """
    if odds_df.empty:
        return
    now_utc = datetime.now(timezone.utc)
    ts = now_utc.isoformat()
    try:
        with db_conn() as conn:
            for _, g in odds_df.iterrows():
                commence = str(g.get("commence_time", ""))
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    if dt <= now_utc:
                        continue  # game already started — don't overwrite pre-game line
                except Exception:
                    pass
                conn.execute(
                    """INSERT OR IGNORE INTO odds_history
                       (game_id, pull_timestamp, home_team, away_team, commence_time,
                        spread_home, spread_away, total_line, bookmaker, is_opening)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        g.get("game_id", ""),
                        ts,
                        g.get("home_team", ""),
                        g.get("away_team", ""),
                        commence,
                        g.get("spread_home"),
                        g.get("spread_away"),
                        g.get("total_line"),
                        g.get("bookmaker", ""),
                        False,
                    ],
                )
    except Exception:
        pass  # don't crash the page if storage fails


@st.cache_data(ttl=300)
def load_pregame_reference(game_ids: tuple) -> dict:
    """
    Return the earliest stored line for each game_id from odds_history.
    This is the pre-game snapshot that was locked in before tipoff.
    Returns dict: game_id → {spread_home, spread_away, total_line}
    """
    if not game_ids:
        return {}
    try:
        ph = ",".join("?" * len(game_ids))
        df = query_df(
            f"""SELECT oh.game_id,
                       oh.spread_home AS pg_spread_home,
                       oh.spread_away AS pg_spread_away,
                       oh.total_line  AS pg_total
                FROM odds_history oh
                INNER JOIN (
                    SELECT game_id, MIN(pull_timestamp) AS min_ts
                    FROM odds_history
                    WHERE game_id IN ({ph})
                    GROUP BY game_id
                ) earliest
                  ON oh.game_id = earliest.game_id
                 AND oh.pull_timestamp = earliest.min_ts""",
            params=list(game_ids),
        )
        return {row["game_id"]: row.to_dict() for _, row in df.iterrows()} if not df.empty else {}
    except Exception:
        return {}


# ── Project all games ─────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def build_projections(round_ctx: int, yr: int, bankroll_: int, sizing_: str,
                      pregame_ref: dict = None):
    odds_df = fetch_live_odds()
    if odds_df.empty:
        return pd.DataFrame(), []

    pregame_ref = pregame_ref or {}
    rows, errors = [], []
    now_utc = datetime.now(timezone.utc)
    # Seed map: used when torvik_ratings.seed is NULL (current year before data refresh)
    bracket_seeds = load_bracket_seed_map(yr)

    for _, game in odds_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        mkt_spread_home = game.get("spread_home")
        mkt_total = game.get("total_line")
        commence = str(game.get("commence_time", ""))
        game_id = game.get("game_id", "")

        # Parse tip-off time
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            local_dt = dt.astimezone(EST)
            date_key   = local_dt.strftime("%Y-%m-%d")
            date_label = local_dt.strftime("%A, %b %d")
            time_label = local_dt.strftime("%I:%M %p ET").lstrip("0")
            days_out   = (dt.date() - now_utc.date()).days
            is_live    = dt <= now_utc   # tip-off time has passed
        except Exception:
            date_key = commence[:10]
            date_label = commence[:10]
            time_label = ""
            days_out = 0
            is_live  = False

        seed_home = bracket_seeds.get(home)
        seed_away = bracket_seeds.get(away)
        proj = project_game(home, away, round_num=round_ctx, year=yr,
                            seed_a=seed_home, seed_b=seed_away)
        if "error" in proj:
            errors.append(f"{home} vs {away}: {proj['error']}")
            continue

        ta = proj["team_a"]
        tb = proj["team_b"]
        model_spread = proj["projected_spread"]   # model convention: + = team_a wins
        model_total  = proj["projected_total"]
        wpa = proj["win_prob_a"]
        wpb = proj["win_prob_b"]
        score_a = proj["projected_score_a"]
        score_b = proj["projected_score_b"]

        # ── Pre-game reference line (earliest stored snapshot before tipoff) ──
        pg = pregame_ref.get(game_id, {})
        pg_spread_home = pg.get("pg_spread_home")
        pg_total       = pg.get("pg_total")

        def _spread_to_ta(spread_h):
            """Convert home-team spread to team_a convention (+ = ta favored)."""
            if spread_h is None:
                return None
            sh = float(spread_h)
            return -sh if (ta.lower() == home.lower()) else sh

        # Live market spread (current refresh)
        live_spread_ta  = _spread_to_ta(mkt_spread_home)
        live_total      = float(mkt_total) if mkt_total is not None else None

        # Pre-game line (falls back to live if no history yet)
        pregame_spread_ta = _spread_to_ta(pg_spread_home) if pg_spread_home is not None else live_spread_ta
        pregame_total     = float(pg_total) if pg_total is not None else live_total

        # Edge is always vs the PRE-GAME line (what you actually bet at)
        mkt_spread_ta = pregame_spread_ta   # used for edge / sizing calcs
        spread_edge = (model_spread - pregame_spread_ta) if pregame_spread_ta is not None else None
        total_edge  = (model_total - pregame_total)      if pregame_total    is not None else None

        # Did the live line move vs pre-game? (show alert if ≥ 0.5 pt)
        line_moved = (
            live_spread_ta is not None
            and pregame_spread_ta is not None
            and abs(live_spread_ta - pregame_spread_ta) >= 0.5
        )
        total_moved = (
            live_total is not None
            and pregame_total is not None
            and abs(live_total - pregame_total) >= 0.5
        )

        # ── Determine model's pick & flip display perspective ─────────────────
        # coverage_probability() already returns P(model's preferred side covers)
        # regardless of which team that is
        if spread_edge is not None:
            cov_prob = coverage_probability(model_spread, mkt_spread_ta)
            hk = half_kelly(cov_prob)
            fk = kelly_fraction(cov_prob)
            tier_label, tier_emoji = bet_tier(spread_edge, cov_prob)
            if sizing_ == "Full Kelly":
                bet_size = round(bankroll_ * fk)
            elif sizing_ == "Half Kelly":
                bet_size = round(bankroll_ * hk)
            else:
                bet_size = 100
        else:
            cov_prob = hk = fk = None
            tier_label, tier_emoji = "Pass", "⚪"
            bet_size = 0

        # ── Determine VALUE pick ─────────────────────────────────────────────
        # The bet is on whichever team the MARKET undervalues, not the model's
        # projected winner. With market data: edge direction decides; without
        # market data: fall back to model's projected winner.
        #
        # spread_edge = model_spread − mkt_spread_ta (model convention)
        #   > 0 → model more bullish on team_a than market  → bet team_a
        #   < 0 → model less bullish on team_a than market  → bet team_b (underdog value)
        if mkt_spread_ta is not None:
            team_a_has_edge = (spread_edge >= 0)
        else:
            team_a_has_edge = (model_spread >= 0)

        if team_a_has_edge:
            pick, opp = ta, tb
            # Betting convention for team_a: negate model_spread
            #   model_spread > 0 (team_a wins) → -model_spread < 0 = favorite display
            #   model_spread < 0 (team_a loses) → -model_spread > 0 = underdog display
            pick_model_display = -model_spread
            pick_mkt_display   = -mkt_spread_ta if mkt_spread_ta is not None else None
            pick_win_prob, opp_win_prob = wpa, wpb
            pick_score, opp_score = score_a, score_b
        else:
            pick, opp = tb, ta
            # Betting convention for team_b: use model_spread as-is
            #   model_spread > 0 (team_a wins, team_b is underdog) → +X = underdog display
            #   model_spread < 0 (team_b wins, team_b is favorite) → -X = favorite display
            pick_model_display = model_spread
            pick_mkt_display   = mkt_spread_ta if mkt_spread_ta is not None else None
            pick_win_prob, opp_win_prob = wpb, wpa
            pick_score, opp_score = score_b, score_a

        abs_edge = abs(spread_edge) if spread_edge is not None else None

        tier_order = {"Strong": 0, "Value": 1, "Lean": 2, "Pass": 3}.get(tier_label, 4)

        rows.append({
            "date_key":    date_key,
            "date_label":  date_label,
            "days_out":    days_out,
            "time":        time_label,
            "is_live":     is_live,
            # canonical teams (for reference)
            "team_a": ta, "team_b": tb,
            # pick-perspective display fields
            "pick":        pick,
            "opp":         opp,
            "matchup":     f"{ta} vs {tb}",
            "pick_matchup": f"{pick} vs {opp}",
            "pick_model_display": pick_model_display,
            "pick_mkt_display":   pick_mkt_display,
            "pick_win_prob":  pick_win_prob,
            "opp_win_prob":   opp_win_prob,
            "pick_score":  pick_score,
            "opp_score":   opp_score,
            # model
            "model_spread":  model_spread,
            "mkt_spread_ta": mkt_spread_ta,        # = pregame line
            "spread_edge":   spread_edge,
            "abs_edge":      abs_edge,
            "cov_prob":      cov_prob,
            "hk_pct":        hk,
            "bet_size":      bet_size,
            "tier_label":    tier_label,
            "tier_emoji":    tier_emoji,
            "tier_order":    tier_order,
            # live vs pregame line tracking
            "pregame_spread_ta": pregame_spread_ta,
            "live_spread_ta":    live_spread_ta,
            "line_moved":        line_moved,
            # total
            "model_total":    round(model_total, 1),
            "pregame_total":  pregame_total,
            "live_total":     live_total,
            "mkt_total":      pregame_total,  # keep backward compat
            "total_edge":     total_edge,
            "total_moved":    total_moved,
        })

    return pd.DataFrame(rows), errors


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Upcoming CBB Games")

hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.caption(
        f"{season_label(current_year)} · {data_as_of(current_year)} · "
        f"Auto-refreshes every 2 min · {datetime.now(timezone.utc).astimezone(EST).strftime('%I:%M %p ET').lstrip('0')}"
    )
with hcol2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Loading games..."):
    # 1. Fetch live odds (cached 2 min)
    _live_df = fetch_live_odds()
    # 2. Store pre-game snapshots for upcoming games (runs every refresh, outside cache)
    _store_pregame_snapshot(_live_df)
    # 3. Load earliest stored line per game as pre-game reference
    _game_ids = tuple(_live_df["game_id"].tolist()) if not _live_df.empty else ()
    _pregame_ref = load_pregame_reference(_game_ids)
    # 4. Build full projections (cached 2 min)
    df, errors = build_projections(round_context, current_year, bankroll, sizing,
                                   pregame_ref=_pregame_ref)

if df.empty:
    st.warning("No upcoming NCAAB games found.")
    st.info("Make sure your ODDS_API_KEY is set and 2026 ratings are ingested.")
    if errors:
        with st.expander("Projection errors"):
            st.write("\n".join(errors))
    st.stop()

# ── Apply filters ─────────────────────────────────────────────────────────────
view = df.copy()
if hide_pass:
    view = view[view["tier_order"] < 3]
if min_edge > 0:
    view = view[view["abs_edge"].notna() & (view["abs_edge"] >= min_edge)]
# Date filter
_today_str    = _dt.date.today().isoformat()
_tomorrow_str = (_dt.date.today() + _dt.timedelta(days=1)).isoformat()
if date_filter == "Today":
    view = view[view["date_key"] == _today_str]
elif date_filter == "Tomorrow":
    view = view[view["date_key"] == _tomorrow_str]
# "All upcoming" — no filter

# ── Summary KPIs ──────────────────────────────────────────────────────────────
strong = (df["tier_order"] == 0).sum()
value  = (df["tier_order"] == 1).sum()
lean   = (df["tier_order"] == 2).sum()
total_alloc = df[df["tier_order"] < 3]["bet_size"].sum()
n_days = df["date_key"].nunique()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Games", len(df))
k2.metric("Days", n_days)
k3.metric("🔥 Strong", int(strong))
k4.metric("✅ Value",  int(value))
k5.metric("📊 Lean",   int(lean))
k6.metric("Allocated", f"${total_alloc:,.0f}")

# ── Best bets banner ──────────────────────────────────────────────────────────
best = df[df["tier_order"] <= 1].sort_values("abs_edge", ascending=False)
if not best.empty:
    st.divider()
    st.subheader("🏆 Best Bets")
    cols = st.columns(min(len(best), 3))
    for idx, (_, b) in enumerate(best.iterrows()):
        with cols[idx % len(cols)]:
            color = "#1a472a" if b["tier_label"] == "Strong" else "#1e3a5f"
            model_str = f"{b['pick']} {b['pick_model_display']:+.1f}"
            mkt_str   = (f"{b['pick']} {b['pick_mkt_display']:+.1f}"
                         if b["pick_mkt_display"] is not None else "—")
            st.markdown(
                f"""<div style='background:{color};border-radius:10px;padding:14px;margin-bottom:8px'>
                <div style='font-size:1.1rem;font-weight:bold; color: rgb(255, 255, 255);'>{b['tier_emoji']} {b['matchup']}</div>
                <div style='color:#ccc;font-size:0.85rem'>{b['date_label']} · {b['time']}</div>
                <div style='color:#aaa;font-size:0.8rem;margin-top:4px'>Model picks: <b style='color:#fff'>{b['pick']}</b></div>
                <hr style='border-color:#ffffff22;margin:8px 0'>
                <div style='display:flex;justify-content:space-between'>
                  <div><div style='color:#aaa;font-size:0.75rem'>MODEL</div>
                       <div style='font-size:1rem;font-weight:bold; color: rgb(255, 255, 255);'>{model_str}</div></div>
                  <div><div style='color:#aaa;font-size:0.75rem'>PRE-GAME LINE</div>
                       <div style='font-size:1rem; color: rgb(255, 255, 255);'>{mkt_str}</div></div>
                  <div><div style='color:#aaa;font-size:0.75rem'>EDGE</div>
                       <div style='font-size:1rem;font-weight:bold;color:#2ecc71'>+{b['abs_edge']:.1f} pts</div></div>
                </div>
                <div style='display:flex;justify-content:space-between;margin-top:8px'>
                  <div><div style='color:#aaa;font-size:0.75rem'>COV PROB</div>
                       <div>{b['cov_prob']:.1%}</div></div>
                  <div><div style='color:#aaa;font-size:0.75rem'>HALF KELLY</div>
                       <div>{b['hk_pct']*100:.1f}%</div></div>
                  <div><div style='color:#aaa;font-size:0.75rem'>BET</div>
                       <div style='font-weight:bold'>${b['bet_size']:,}</div></div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

# ── Games by date ─────────────────────────────────────────────────────────────
st.divider()

dates = sorted(view["date_key"].unique())
if not dates:
    st.info("No games match current filters.")
    st.stop()

date_tabs = st.tabs([view.loc[view["date_key"] == d, "date_label"].iloc[0] for d in dates])

for date, tab in zip(dates, date_tabs):
    day_df = view[view["date_key"] == date].sort_values(
        ["tier_order", "abs_edge"], ascending=[True, False]
    )

    with tab:
        if day_df.empty:
            st.caption("No games match current filters for this day.")
            continue

        # Split live vs upcoming games
        live_df     = day_df[day_df["is_live"] == True]
        upcoming_df = day_df[day_df["is_live"] == False]

        if not live_df.empty:
            st.markdown("### 🔴 In Progress")

        for _, row in live_df.iterrows():
            tier_colors = {
                "Strong": ("#1a472a", "#2ecc71"),
                "Value":  ("#1e3a5f", "#3498db"),
                "Lean":   ("#3d2b1f", "#f39c12"),
                "Pass":   ("#2a2a2a", "#7f8c8d"),
            }
            bg, accent = tier_colors.get(row["tier_label"], ("#2a2a2a", "#aaa"))

            # Pick-perspective spread strings
            model_str = f"{row['pick']} {row['pick_model_display']:+.1f}"
            mkt_str   = (f"{row['pick']} {row['pick_mkt_display']:+.1f}"
                         if row["pick_mkt_display"] is not None else "—")
            edge_str  = f"+{row['abs_edge']:.1f}" if row["abs_edge"] is not None else "—"

            ou_str = ""
            if not _total_calibrated:
                ou_str = None  # suppress — will render caption instead
            elif row["total_edge"] is not None:
                direction = "Over" if row["total_edge"] > 0 else "Under"
                ou_str = (f"O/U: model {row['model_total']} vs {row['mkt_total']} "
                          f"({direction} {abs(row['total_edge']):.1f})")

            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])

            with c1:
                st.markdown(
                    f"<div style='padding:4px 0'>"
                    f"<span style='font-weight:bold;font-size:1rem'>{row['matchup']}</span>"
                    f"<span style='margin-left:8px;background:#c0392b;color:#fff;"
                    f"font-size:0.65rem;font-weight:bold;padding:2px 6px;border-radius:4px'>🔴 LIVE</span><br>"
                    f"<span style='color:#aaa;font-size:0.8rem'>Started {row['time']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                score_line = f"{row['pick_score']:.0f} – {row['opp_score']:.0f}"
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>PROJECTED</div>"
                    f"<div style='font-size:0.95rem;font-weight:600'>{model_str}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>{score_line}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                # Build market/line display: always anchor to pre-game line
                # Show live line separately if it has moved ≥ 0.5 pts
                pg_display = (
                    f"{row['pick']} {row['pick_mkt_display']:+.1f}"
                    if row["pick_mkt_display"] is not None else "—"
                )
                if row.get("line_moved") and row.get("live_spread_ta") is not None:
                    # Live line has moved — show both
                    live_ta = row["live_spread_ta"]
                    if row["pick"] == row["team_a"]:
                        live_display = f"{row['pick']} {-live_ta:+.1f}"
                    else:
                        live_display = f"{row['pick']} {live_ta:+.1f}"
                    line_move_dir = "▲" if (live_ta > (row.get("pregame_spread_ta") or live_ta)) else "▼"
                    mkt_html = (
                        f"<div style='font-size:0.75rem;color:#aaa'>OPEN</div>"
                        f"<div style='font-size:0.95rem;font-weight:600'>{pg_display}</div>"
                        f"<div style='font-size:0.7rem;color:#f39c12'>LIVE {live_display} {line_move_dir}</div>"
                    )
                else:
                    # No movement — label clearly as pre-game or open line
                    pg_label = "OPEN LINE" if row.get("is_live") else "PRE-GAME"
                    mkt_html = (
                        f"<div style='font-size:0.75rem;color:#aaa'>{pg_label}</div>"
                        f"<div style='font-size:0.95rem'>{pg_display}</div>"
                        f"<div style='font-size:0.75rem;color:#aaa'>&nbsp;</div>"
                    )
                # Total line movement note
                if row.get("total_moved") and row.get("live_total") is not None:
                    total_note = f"<div style='font-size:0.7rem;color:#f39c12'>O/U open {row['pregame_total']} → live {row['live_total']}</div>"
                else:
                    total_note = ""
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"{mkt_html}"
                    f"<div style='color:{accent};font-weight:bold;margin-top:2px'>{edge_str} pts edge</div>"
                    f"{total_note}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c4:
                win_pct = f"{row['pick_win_prob']:.0%} / {row['opp_win_prob']:.0%}"
                cov_str = f"{row['cov_prob']:.1%}" if row["cov_prob"] else "—"
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>WIN PROB ({row['pick']})</div>"
                    f"<div style='font-size:0.9rem'>{win_pct}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>cov {cov_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c5:
                tier_badge = f"{row['tier_emoji']} {row['tier_label']}"
                bet_str = f"${row['bet_size']:,}" if row["bet_size"] > 0 else "—"
                hk_str  = f"{row['hk_pct']*100:.1f}%" if row["hk_pct"] else "—"
                st.markdown(
                    f"<div style='text-align:center;padding:6px 8px;background:{bg};"
                    f"border-left:3px solid {accent};border-radius:6px'>"
                    f"<div style='font-weight:bold;color:{accent}'>{tier_badge}</div>"
                    f"<div style='font-size:0.85rem;color:#ddd'>Bet {row['pick']}</div>"
                    f"<div style='font-size:0.9rem'>{bet_str}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>{hk_str} Kelly</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if ou_str is None:
                st.caption("O/U: model uncalibrated — retrain to enable")
            elif ou_str:
                st.caption(f"&nbsp;&nbsp;&nbsp;{ou_str}")
            st.divider()

        # ── Upcoming (pre-game) ────────────────────────────────────────────────
        if not upcoming_df.empty:
            if not live_df.empty:
                st.markdown("### 📋 Upcoming")
            for _, row in upcoming_df.iterrows():
                tier_colors = {
                    "Strong": ("#1a472a", "#2ecc71"),
                    "Value":  ("#1e3a5f", "#3498db"),
                    "Lean":   ("#3d2b1f", "#f39c12"),
                    "Pass":   ("#2a2a2a", "#7f8c8d"),
                }
                bg, accent = tier_colors.get(row["tier_label"], ("#2a2a2a", "#aaa"))

                model_str = f"{row['pick']} {row['pick_model_display']:+.1f}"
                mkt_str   = (f"{row['pick']} {row['pick_mkt_display']:+.1f}"
                             if row["pick_mkt_display"] is not None else "—")
                edge_str  = f"+{row['abs_edge']:.1f}" if row["abs_edge"] is not None else "—"

                ou_str = ""
                if not _total_calibrated:
                    ou_str = None  # suppress — will render caption instead
                elif row["total_edge"] is not None:
                    direction = "Over" if row["total_edge"] > 0 else "Under"
                    ou_str = (f"O/U: model {row['model_total']} vs {row['mkt_total']} "
                              f"({direction} {abs(row['total_edge']):.1f})")

                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])

                with c1:
                    st.markdown(
                        f"<div style='padding:4px 0'>"
                        f"<span style='font-weight:bold;font-size:1rem'>{row['matchup']}</span><br>"
                        f"<span style='color:#aaa;font-size:0.8rem'>{row['date_label']} · {row['time']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    score_line = f"{row['pick_score']:.0f} – {row['opp_score']:.0f}"
                    st.markdown(
                        f"<div style='text-align:center;padding:4px'>"
                        f"<div style='color:#aaa;font-size:0.75rem'>PROJECTED</div>"
                        f"<div style='font-size:0.95rem;font-weight:600'>{model_str}</div>"
                        f"<div style='color:#aaa;font-size:0.75rem'>{score_line}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    pg_display = (
                        f"{row['pick']} {row['pick_mkt_display']:+.1f}"
                        if row["pick_mkt_display"] is not None else "—"
                    )
                    if row.get("line_moved") and row.get("live_spread_ta") is not None:
                        live_ta = row["live_spread_ta"]
                        if row["pick"] == row["team_a"]:
                            live_display = f"{row['pick']} {-live_ta:+.1f}"
                        else:
                            live_display = f"{row['pick']} {live_ta:+.1f}"
                        line_move_dir = "▲" if (live_ta > (row.get("pregame_spread_ta") or live_ta)) else "▼"
                        mkt_html = (
                            f"<div style='font-size:0.75rem;color:#aaa'>OPEN</div>"
                            f"<div style='font-size:0.95rem;font-weight:600'>{pg_display}</div>"
                            f"<div style='font-size:0.7rem;color:#f39c12'>LIVE {live_display} {line_move_dir}</div>"
                        )
                    else:
                        mkt_html = (
                            f"<div style='font-size:0.75rem;color:#aaa'>PRE-GAME</div>"
                            f"<div style='font-size:0.95rem'>{pg_display}</div>"
                            f"<div style='font-size:0.75rem;color:#aaa'>&nbsp;</div>"
                        )
                    if row.get("total_moved") and row.get("live_total") is not None:
                        total_note = f"<div style='font-size:0.7rem;color:#f39c12'>O/U open {row['pregame_total']} → live {row['live_total']}</div>"
                    else:
                        total_note = ""
                    st.markdown(
                        f"<div style='text-align:center;padding:4px'>"
                        f"{mkt_html}"
                        f"<div style='color:{accent};font-weight:bold;margin-top:2px'>{edge_str} pts edge</div>"
                        f"{total_note}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c4:
                    win_pct = f"{row['pick_win_prob']:.0%} / {row['opp_win_prob']:.0%}"
                    cov_str = f"{row['cov_prob']:.1%}" if row["cov_prob"] else "—"
                    st.markdown(
                        f"<div style='text-align:center;padding:4px'>"
                        f"<div style='color:#aaa;font-size:0.75rem'>WIN PROB ({row['pick']})</div>"
                        f"<div style='font-size:0.9rem'>{win_pct}</div>"
                        f"<div style='color:#aaa;font-size:0.75rem'>cov {cov_str}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c5:
                    tier_badge = f"{row['tier_emoji']} {row['tier_label']}"
                    bet_str = f"${row['bet_size']:,}" if row["bet_size"] > 0 else "—"
                    hk_str  = f"{row['hk_pct']*100:.1f}%" if row["hk_pct"] else "—"
                    st.markdown(
                        f"<div style='text-align:center;padding:6px 8px;background:{bg};"
                        f"border-left:3px solid {accent};border-radius:6px'>"
                        f"<div style='font-weight:bold;color:{accent}'>{tier_badge}</div>"
                        f"<div style='font-size:0.85rem;color:#ddd'>Bet {row['pick']}</div>"
                        f"<div style='font-size:0.9rem'>{bet_str}</div>"
                        f"<div style='color:#aaa;font-size:0.75rem'>{hk_str} Kelly</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if ou_str is None:
                    st.caption("O/U: model uncalibrated — retrain to enable")
                elif ou_str:
                    st.caption(f"&nbsp;&nbsp;&nbsp;{ou_str}")
                st.divider()

# ── Errors ────────────────────────────────────────────────────────────────────
if errors:
    with st.expander(f"⚠️ {len(errors)} games skipped (team name not in DB)"):
        for e in errors:
            st.caption(e)
        st.caption("Run data ingestion or check team names on the Teams page.")
