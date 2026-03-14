import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
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
    st.divider()
    st.caption("Half Kelly is recommended to reduce variance.")


# ── Load live odds + project ──────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_bet_board(year: int):
    from src.ingest.odds import get_latest_odds
    from src.utils.db import query_df

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
                    """SELECT spread_line, total_line FROM historical_lines
                       WHERE year=? AND ((team1=? AND team2=?) OR (team1=? AND team2=?))
                       LIMIT 1""",
                    params=[year, t1, t2, t2, t1],
                )
                mkt_spread = float(lines.iloc[0]["spread_line"]) if not lines.empty and pd.notna(lines.iloc[0]["spread_line"]) else None
                mkt_total = float(lines.iloc[0]["total_line"]) if not lines.empty and pd.notna(lines.iloc[0]["total_line"]) else None

                proj["market_spread"] = mkt_spread
                proj["market_total"] = mkt_total
                proj["spread_edge"] = (proj["projected_spread"] - mkt_spread) if mkt_spread is not None else None
                proj["total_edge"] = (proj["projected_total"] - mkt_total) if mkt_total is not None else None
                proj["game_date"] = str(game.get("game_date") or "")
                rows.append(proj)
    else:
        for _, odds in odds_df.iterrows():
            home = odds["home_team"]
            away = odds["away_team"]
            mkt_spread = odds.get("spread_home")
            mkt_total = odds.get("total_line")
            proj = project_game(home, away, round_num=1, year=year)
            if "error" not in proj:
                proj["market_spread"] = mkt_spread
                proj["market_total"] = mkt_total
                proj["spread_edge"] = (proj["projected_spread"] - mkt_spread) if mkt_spread is not None else None
                proj["total_edge"] = (proj["projected_total"] - mkt_total) if mkt_total is not None else None
                proj["game_date"] = odds.get("commence_time", "")
                rows.append(proj)

    return pd.DataFrame(rows)


with st.spinner("Loading projections..."):
    df = load_bet_board(current_year)

if df.empty:
    st.info("No games found. Make sure historical data or live odds are loaded.")
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

    rows_enriched.append({
        "Game": f"{row['team_a']} vs {row['team_b']}",
        "Round": row.get("round_name", ROUND_NAMES.get(row.get("round_num"), "?")),
        "Date": str(row.get("game_date", ""))[:10],
        "Model Spread": ps,
        "Market Spread": mkt,
        "Edge (pts)": round(float(spread_edge), 1) if spread_edge is not None else None,
        "Cov Prob": round(cov_prob * 100, 1) if cov_prob is not None else None,
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
    })

edf = pd.DataFrame(rows_enriched)

# ── Filter ────────────────────────────────────────────────────────────────────
if not show_passes:
    edf = edf[edf["_tier_order"] < 3]

if min_edge > 0:
    edf = edf[edf["Edge (pts)"].abs() >= min_edge]

edf = edf.sort_values(["_tier_order", "Edge (pts)"],
                       key=lambda s: s if s.name != "Edge (pts)" else s.abs(),
                       ascending=[True, False])

# ── Summary row ───────────────────────────────────────────────────────────────
strong = (edf["_tier_order"] == 0).sum()
value = (edf["_tier_order"] == 1).sum()
lean = (edf["_tier_order"] == 2).sum()
total_allocated = edf["Bet ($)"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔥 Strong Bets", strong)
k2.metric("✅ Value Bets", value)
k3.metric("📊 Lean Bets", lean)
k4.metric("Total Bets", strong + value + lean)
k5.metric("Total Allocated", f"${total_allocated:,.0f}")

st.divider()

# ── Main table ────────────────────────────────────────────────────────────────
display_cols = [
    "Tier", "Game", "Round", "Date",
    "Model Spread", "Market Spread", "Edge (pts)",
    "Cov Prob", "Half Kelly %", "Bet ($)",
    "Win Prob A",
]
show_df = edf[[c for c in display_cols if c in edf.columns]].copy()

# Color-code by tier
def tier_color(val):
    if "Strong" in str(val):
        return "background-color: #1a472a; color: white"
    if "Value" in str(val):
        return "background-color: #1e3a5f; color: white"
    if "Lean" in str(val):
        return "background-color: #3d2b1f; color: white"
    return ""

st.dataframe(
    show_df.style.applymap(tier_color, subset=["Tier"]),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Edge distribution chart ───────────────────────────────────────────────────
if not edf.empty and edf["Edge (pts)"].notna().any():
    st.subheader("Edge Distribution")
    fig = px.bar(
        edf[edf["Edge (pts)"].notna()].sort_values("Edge (pts)", ascending=False),
        x="Game",
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
    ou_cols = ["Game", "Round", "Model Total", "Market Total", "Total Edge", "O/U Pick"]
    st.dataframe(
        totals_df[[c for c in ou_cols if c in totals_df.columns]]
        .sort_values("Total Edge", key=lambda s: s.abs(), ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "**How to read this:** Edge = Model Spread – Market Spread (positive = model likes Team A). "
    "Coverage probability uses a Normal distribution with σ=12 pts (backtest RMSE). "
    "Kelly% is Half Kelly applied to bankroll. "
    "Tiers: 🔥 Strong (|edge|≥7, cov≥58%) · ✅ Value (≥5, ≥56%) · 📊 Lean (≥3, ≥54%)"
)
