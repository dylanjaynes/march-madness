import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES, COMPETITIVE_SPREAD_THRESHOLD
from src.model.predict import (
    project_game, coverage_probability, kelly_fraction, half_kelly, bet_tier,
    season_label, data_as_of,
)

st.set_page_config(page_title="Bet Board", page_icon="🎯", layout="wide")
st.title("🎯 Bet Board")

current_year = TOURNAMENT_YEARS[-1]
season = season_label(current_year)
data_note = data_as_of(current_year)

st.caption(f"**{season} season** · {data_note} · Ranked by edge vs. market spread")

# ── Bankroll sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Bet Sizing")
    bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=1_000_000,
                                value=1000, step=100)
    sizing = st.radio("Kelly sizing", ["Half Kelly", "Full Kelly", "Flat ($100)"])
    min_edge = st.slider("Min edge threshold (pts)", 0.0, 10.0, 3.0, 0.5)
    show_passes = st.checkbox("Show Pass-tier bets", value=False)
    show_nit = st.checkbox("Show NIT games", value=True)
    st.divider()
    if st.button("🔄 Refresh odds & projections"):
        load_bet_board.clear()
        st.rerun()
    st.caption("Half Kelly is recommended to reduce variance.")


# ── Helpers ───────────────────────────────────────────────────────────────────
_EDT = timezone(timedelta(hours=-4))  # Eastern Daylight Time (UTC-4, in effect during March tournament)

def _parse_utc(t) -> datetime:
    """Parse ISO-8601 string to UTC-aware datetime. Returns epoch if invalid."""
    if not t:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        s = str(t).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _fmt_et(t) -> str:
    """Format UTC ISO timestamp as Eastern time string like 'Mar 20 7:10p'."""
    dt = _parse_utc(t)
    if dt == datetime.min.replace(tzinfo=timezone.utc):
        return ""
    et = dt.astimezone(_EDT)
    hour = et.hour % 12 or 12
    ampm = "p" if et.hour >= 12 else "a"
    return et.strftime(f"%b %-d {hour}:%M{ampm} ET")


# ── Load live odds + project ──────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_bet_board(year: int):
    from src.ingest.odds import fetch_current_games, poll_and_store_odds
    from src.utils.db import query_df

    # Always fetch live odds first; fall back to DB cache if API is unavailable
    odds_df = fetch_current_games()
    if odds_df.empty:
        from src.ingest.odds import get_latest_odds
        odds_df = get_latest_odds()
    rows = []

    if odds_df.empty:
        # Fall back to current year bracket
        bracket_df = query_df(
            "SELECT * FROM historical_results WHERE year = ? ORDER BY round_number, game_date",
            params=[year],
        )
        for _, game in bracket_df.iterrows():
            t1, t2 = game["team1"], game["team2"]
            rn = int(game.get("round_number") or 1)
            proj = project_game(t1, t2, rn, year,
                                seed_a=game.get("seed1"),
                                seed_b=game.get("seed2"),
                                game_date=str(game.get("game_date") or ""))
            if "error" not in proj:
                # Try to pull market lines from historical_lines
                lines = query_df(
                    """SELECT spread_line, total_line, spread_favorite FROM historical_lines
                       WHERE year=? AND ((team1=? AND team2=?) OR (team1=? AND team2=?))
                       LIMIT 1""",
                    params=[year, t1, t2, t2, t1],
                )
                mkt_spread_raw = float(lines.iloc[0]["spread_line"]) if not lines.empty and pd.notna(lines.iloc[0]["spread_line"]) else None
                mkt_total = float(lines.iloc[0]["total_line"]) if not lines.empty and pd.notna(lines.iloc[0]["total_line"]) else None

                if mkt_spread_raw is not None and not lines.empty:
                    fav = str(lines.iloc[0].get("spread_favorite") or "").strip().lower()
                    team_a_name = str(proj.get("team_a") or "").strip().lower()
                    mkt_spread = mkt_spread_raw if fav == team_a_name else -mkt_spread_raw
                else:
                    mkt_spread = None

                proj["market_spread"] = mkt_spread
                proj["market_total"] = mkt_total
                proj["spread_edge"] = (proj["projected_spread"] - mkt_spread) if mkt_spread is not None else None
                proj["total_edge"] = (proj["projected_total"] - mkt_total) if mkt_total is not None else None
                proj["game_date"] = str(game.get("game_date") or "")
                proj["is_nit"] = False
                rows.append(proj)
    else:
        # Seeds: try torvik_ratings first, supplement with tournament_bracket
        seed_df = query_df(
            "SELECT team, seed FROM torvik_ratings WHERE year = ? AND seed IS NOT NULL",
            params=[year],
        )
        seed_lookup = dict(zip(seed_df["team"], seed_df["seed"])) if not seed_df.empty else {}
        tb_df = query_df(
            "SELECT team, seed FROM tournament_bracket WHERE year = ?",
            params=[year],
        )
        if not tb_df.empty:
            for _, row in tb_df.iterrows():
                if row["team"] not in seed_lookup:
                    seed_lookup[row["team"]] = row["seed"]

        # Build set of NCAA tournament teams for NIT detection.
        # Include both the raw bracket name and its Odds-API-normalized form
        # (e.g. "Virginia Commonwealth" → "VCU", "Central Florida" → "UCF").
        # Also include First Four teams (not yet in bracket table since they
        # compete for 4 open seed slots).
        from src.utils.team_map import normalize_team_name, is_known_team as _is_known
        from src.ingest.bracket import fetch_first_four_teams
        ncaa_teams = set()
        if not tb_df.empty:
            for team in tb_df["team"]:
                ncaa_teams.add(team.strip().lower())
                normed = normalize_team_name(team) if _is_known(team) else team
                ncaa_teams.add(normed.strip().lower())
        # Add First Four participants (fetched from ESPN, cached via st.cache_data)
        try:
            ff_teams = fetch_first_four_teams(year)
            ncaa_teams.update(ff_teams)
        except Exception:
            pass

        # ── Filter to upcoming games only ─────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        if "commence_time" in odds_df.columns:
            odds_df = odds_df[
                odds_df["commence_time"].apply(lambda t: _parse_utc(t) > now_utc)
            ].copy()

        for _, odds in odds_df.iterrows():
            home = odds["home_team"]
            away = odds["away_team"]
            mkt_spread_raw = odds.get("spread_home")
            mkt_total = odds.get("total_line")
            seed_home = seed_lookup.get(home)
            seed_away = seed_lookup.get(away)

            # Detect NIT: neither team is in the NCAA bracket
            is_nit = (
                home.strip().lower() not in ncaa_teams
                and away.strip().lower() not in ncaa_teams
            )

            proj = project_game(home, away, round_num=1, year=year,
                                seed_a=seed_home, seed_b=seed_away)
            if "error" not in proj:
                if mkt_spread_raw is not None:
                    try:
                        sh = float(mkt_spread_raw)
                        team_a = str(proj.get("team_a", "")).strip()
                        home_clean = str(home).strip()
                        mkt_spread = -sh if team_a.lower() == home_clean.lower() else sh
                    except (TypeError, ValueError):
                        mkt_spread = None
                else:
                    mkt_spread = None

                proj["market_spread"] = mkt_spread
                proj["market_total"] = mkt_total
                proj["spread_edge"] = (proj["projected_spread"] - mkt_spread) if mkt_spread is not None else None
                proj["total_edge"] = (proj["projected_total"] - mkt_total) if mkt_total is not None else None
                proj["game_date"] = odds.get("commence_time", "")
                proj["game_date_et"] = _fmt_et(odds.get("commence_time", ""))
                proj["is_nit"] = is_nit
                rows.append(proj)

    return pd.DataFrame(rows)


with st.spinner("Loading projections..."):
    df = load_bet_board(current_year)

if df.empty:
    st.info("No upcoming games found. Check back when odds are posted.")
    st.stop()

# ── Enrich with bet metrics ───────────────────────────────────────────────────
rows_enriched = []
for _, row in df.iterrows():
    ps = row.get("projected_spread")
    mkt = row.get("market_spread")
    pt = row.get("projected_total")
    mkt_t = row.get("market_total")

    spread_edge = row.get("spread_edge")
    total_edge = row.get("total_edge")
    is_nit = bool(row.get("is_nit", False))

    # Spread bet
    if ps is not None and mkt is not None:
        cov_prob = coverage_probability(float(ps), float(mkt))
        wk = kelly_fraction(cov_prob)
        hk = half_kelly(cov_prob)
        tier_label, tier_emoji = bet_tier(float(spread_edge), cov_prob)
        if sizing == "Full Kelly":
            bet_size = round(bankroll * wk, 0)
        elif sizing == "Half Kelly":
            bet_size = round(bankroll * hk, 0)
        else:
            bet_size = 100.0
    else:
        cov_prob = wk = hk = None
        tier_label, tier_emoji = "Pass", "⚪"
        bet_size = 0.0

    # Total bet
    if pt is not None and mkt_t is not None:
        direction = "Over" if float(pt) > float(mkt_t) else "Under"
    else:
        direction = None

    # Recommended spread bet
    if spread_edge is not None and mkt is not None:
        if float(spread_edge) >= 0:
            pick_team = row["team_a"]
            pick_spread = -float(mkt)
        else:
            pick_team = row["team_b"]
            pick_spread = float(mkt)
        pick_str = f"{pick_team} {pick_spread:+.1f}"
    else:
        pick_str = "—"

    is_mismatch = row.get("is_mismatch", False)
    rows_enriched.append({
        "Tourney": "🏀 NIT" if is_nit else "🏆 NCAA",
        "Game": f"{row['team_a']} vs {row['team_b']}",
        "Pick": pick_str,
        "Round": row.get("round_name", ROUND_NAMES.get(row.get("round_num"), "?")),
        "Date": row.get("game_date_et") or str(row.get("game_date", ""))[:10],
        "Model Spread": round(float(ps), 1) if ps is not None else None,
        "Market Spread": round(float(mkt), 1) if mkt is not None else None,
        "Edge (pts)": round(float(spread_edge), 1) if spread_edge is not None else None,
        "Cov Prob": min(round(float(cov_prob) * 100, 1), 95.0) if cov_prob is not None else None,
        "Half Kelly %": round(hk * 100, 1) if hk is not None else None,
        "Bet ($)": int(bet_size),
        "Tier": f"{tier_emoji} {tier_label}",
        "_tier_order": {"Strong": 0, "Value": 1, "Lean": 2, "Pass": 3}.get(tier_label, 4),
        "Model Total": round(float(pt), 1) if pt is not None else None,
        "Market Total": mkt_t,
        "Total Edge": round(float(total_edge), 1) if total_edge is not None else None,
        "O/U Pick": direction,
        "Win Prob A": f"{row.get('win_prob_a', 0):.1%}" if row.get("win_prob_a") is not None else None,
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "is_mismatch": is_mismatch,
        "is_nit": is_nit,
    })

edf = pd.DataFrame(rows_enriched)

# ── NIT filter ────────────────────────────────────────────────────────────────
if not show_nit:
    edf = edf[~edf["is_nit"]].copy()

if edf.empty:
    st.info("No upcoming games match your filters.")
    st.stop()

# ── Split competitive vs large-spread by market spread threshold ───────────────
def _is_competitive_row(row):
    mkt = row.get("Market Spread")
    if mkt is None or pd.isna(mkt):
        return True
    try:
        return abs(float(mkt)) <= COMPETITIVE_SPREAD_THRESHOLD
    except (TypeError, ValueError):
        return True

edf["_competitive"] = edf.apply(_is_competitive_row, axis=1)
competitive_df = edf[edf["_competitive"]].copy()
mismatch_df    = edf[~edf["_competitive"]].copy()

# ── Filter (apply only to competitive) ───────────────────────────────────────
if not show_passes:
    competitive_df = competitive_df[competitive_df["_tier_order"] < 3]

if min_edge > 0:
    competitive_df = competitive_df[competitive_df["Edge (pts)"].abs() >= min_edge]

competitive_df = competitive_df.sort_values(
    ["_tier_order", "Edge (pts)"],
    key=lambda s: s if s.name != "Edge (pts)" else s.abs(),
    ascending=[True, False],
)

# Keep edf alias pointing at competitive for legacy references below
edf = competitive_df

# ── Summary row ───────────────────────────────────────────────────────────────
strong = (competitive_df["_tier_order"] == 0).sum()
value = (competitive_df["_tier_order"] == 1).sum()
lean = (competitive_df["_tier_order"] == 2).sum()
total_allocated = competitive_df["Bet ($)"].sum()
nit_count = competitive_df["is_nit"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔥 Strong Bets", strong)
k2.metric("✅ Value Bets", value)
k3.metric("📊 Lean Bets", lean)
k4.metric("Total Bets", strong + value + lean)
k5.metric("Total Allocated", f"${total_allocated:,.0f}")

if nit_count > 0:
    st.info(
        f"🏀 **{nit_count} NIT game{'s' if nit_count != 1 else ''} included** — "
        "NIT backtest shows 68.5% ATS at Edge≥3 (37-17 across 2021–2025), "
        "but sample is smaller and model has less seeding signal for these games. "
        "Treat NIT picks with extra caution.",
        icon="ℹ️",
    )

st.divider()

# ── Main table ────────────────────────────────────────────────────────────────
display_cols = [
    "Tier", "Tourney", "Game", "Pick", "Round", "Date",
    "Model Spread", "Market Spread", "Edge (pts)",
    "Cov Prob", "Half Kelly %", "Bet ($)",
    "Win Prob A",
]
show_df = edf[[c for c in display_cols if c in edf.columns]].copy()


def tier_color(val):
    if "Strong" in str(val):
        return "background-color: #1a472a; color: white"
    if "Value" in str(val):
        return "background-color: #1e3a5f; color: white"
    if "Lean" in str(val):
        return "background-color: #3d2b1f; color: white"
    return ""


def tourney_color(val):
    if "NIT" in str(val):
        return "color: #888; font-style: italic"
    return ""


st.dataframe(
    show_df.style
        .applymap(tier_color, subset=["Tier"])
        .applymap(tourney_color, subset=["Tourney"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Tourney": st.column_config.TextColumn(label="", width="small"),
        "Pick": st.column_config.TextColumn(label="Model Pick", width="medium"),
        "Model Spread": st.column_config.NumberColumn(format="%.1f"),
        "Market Spread": st.column_config.NumberColumn(format="%.1f"),
        "Edge (pts)": st.column_config.NumberColumn(format="%.1f"),
        "Cov Prob": st.column_config.NumberColumn(label="Cov Prob", format="%.1f%%"),
        "Half Kelly %": st.column_config.NumberColumn(format="%.1f%%"),
        "Bet ($)": st.column_config.NumberColumn(format="$%d"),
        "Win Prob A": st.column_config.TextColumn(label="Win Prob"),
    },
)

st.divider()

# ── Edge distribution chart ───────────────────────────────────────────────────
if not competitive_df.empty and competitive_df["Edge (pts)"].notna().any():
    st.subheader("Edge Distribution")
    chart_df = competitive_df[competitive_df["Edge (pts)"].notna()].sort_values("Edge (pts)", ascending=False)
    # Add NIT indicator to game label in chart
    chart_df = chart_df.copy()
    chart_df["Game Label"] = chart_df.apply(
        lambda r: f"{r['Game']} (NIT)" if r["is_nit"] else r["Game"], axis=1
    )
    fig = px.bar(
        chart_df,
        x="Game Label",
        y="Edge (pts)",
        color="Tier",
        color_discrete_map={"🔥 Strong": "#2ecc71", "✅ Value": "#3498db",
                             "📊 Lean": "#f39c12", "⚪ Pass": "#7f8c8d"},
        title="Model Edge vs. Market Spread",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ── Totals section ────────────────────────────────────────────────────────────
totals_df = edf[edf["Total Edge"].notna()].copy()
if not totals_df.empty:
    st.divider()
    st.subheader("O/U Edges")
    ou_cols = ["Tourney", "Game", "Round", "Model Total", "Market Total", "Total Edge", "O/U Pick"]
    st.dataframe(
        totals_df[[c for c in ou_cols if c in totals_df.columns]]
        .sort_values("Total Edge", key=lambda s: s.abs(), ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with st.expander(
    f"⚠️ Large-spread games — {len(mismatch_df)} games (|spread| > {COMPETITIVE_SPREAD_THRESHOLD:.0f} pts)",
    expanded=False,
):
    st.caption(
        f"Games where the market spread exceeds {COMPETITIVE_SPREAD_THRESHOLD:.0f} pts. "
        "Backtest shows 59.6% ATS on 52 such games — not enough signal to bet confidently. "
        "The model's edge on these is noise, not skill."
    )
    if not mismatch_df.empty:
        disp_cols = ["Tourney", "Game", "Round", "Date", "Model Spread", "Market Spread", "Edge (pts)"]
        st.dataframe(
            mismatch_df[[c for c in disp_cols if c in mismatch_df.columns]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No large-spread games today.")

st.divider()
st.caption(
    "**How to read this:** Edge = Model Spread – Market Spread (positive = model likes Team A). "
    "Coverage probability uses a Normal distribution with σ=12 pts (backtest RMSE). "
    "Kelly% is Half Kelly applied to bankroll. "
    "Tiers: 🔥 Strong (|edge|≥7, cov≥58%) · ✅ Value (≥5, ≥56%) · 📊 Lean (≥3, ≥54%) · "
    "🏀 NIT = National Invitation Tournament (model backtest: 68.5% ATS at edge≥3, smaller sample)"
)
