import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.model.predict import (
    project_game, coverage_probability, kelly_fraction, half_kelly,
    bet_tier, season_label, data_as_of,
)

st.set_page_config(page_title="Live Games", page_icon="📡", layout="wide")

current_year = TOURNAMENT_YEARS[-1]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=1_000_000,
                                value=1000, step=100)
    sizing = st.radio("Kelly sizing", ["Half Kelly", "Full Kelly", "Flat ($100)"])
    min_edge = st.slider("Min |edge| (pts)", 0.0, 15.0, 0.0, 0.5)
    hide_pass = st.checkbox("Hide Pass-tier bets", value=True)
    st.divider()
    round_context = st.selectbox(
        "Round context",
        [1, 2, 3, 4, 5, 6],
        format_func=lambda r: ROUND_NAMES.get(r, f"R{r}"),
        help="R64 works best for most games.",
    )


# ── Fetch & cache odds ────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_live_odds():
    from src.ingest.odds import fetch_current_games
    return fetch_current_games()


# ── Project all games ─────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def build_projections(round_ctx: int, yr: int):
    odds_df = fetch_live_odds()
    if odds_df.empty:
        return pd.DataFrame(), []

    rows, errors = [], []
    now_utc = datetime.now(timezone.utc)

    for _, game in odds_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        mkt_spread_home = game.get("spread_home")
        mkt_total = game.get("total_line")
        commence = str(game.get("commence_time", ""))

        # Parse tip-off time
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            local_dt = dt.astimezone()
            date_key = local_dt.strftime("%Y-%m-%d")
            date_label = local_dt.strftime("%A, %b %d")
            time_label = local_dt.strftime("%I:%M %p").lstrip("0")
            days_out = (dt.date() - now_utc.date()).days
        except Exception:
            date_key = commence[:10]
            date_label = commence[:10]
            time_label = ""
            days_out = 0

        proj = project_game(home, away, round_num=round_ctx, year=yr)
        if "error" in proj:
            errors.append(f"{home} vs {away}: {proj['error']}")
            continue

        ta = proj["team_a"]
        tb = proj["team_b"]
        model_spread = proj["projected_spread"]
        model_total  = proj["projected_total"]
        wpa = proj["win_prob_a"]
        wpb = proj["win_prob_b"]
        score_a = proj["projected_score_a"]
        score_b = proj["projected_score_b"]

        # Market spread → model convention (positive = team_a favored)
        if mkt_spread_home is not None:
            msh = float(mkt_spread_home)
            mkt_spread_ta = -msh if (ta.lower() == home.lower()) else msh
            spread_edge = model_spread - mkt_spread_ta
        else:
            mkt_spread_ta = None
            spread_edge = None

        total_edge = (model_total - float(mkt_total)) if mkt_total is not None else None

        if spread_edge is not None:
            cov_prob = coverage_probability(model_spread, mkt_spread_ta)
            hk = half_kelly(cov_prob)
            fk = kelly_fraction(cov_prob)
            tier_label, tier_emoji = bet_tier(spread_edge, cov_prob)
            if sizing == "Full Kelly":
                bet_size = round(bankroll * fk)
            elif sizing == "Half Kelly":
                bet_size = round(bankroll * hk)
            else:
                bet_size = 100
        else:
            cov_prob = hk = fk = None
            tier_label, tier_emoji = "Pass", "⚪"
            bet_size = 0

        tier_order = {"Strong": 0, "Value": 1, "Lean": 2, "Pass": 3}.get(tier_label, 4)

        rows.append({
            # display
            "date_key":   date_key,
            "date_label": date_label,
            "days_out":   days_out,
            "time":       time_label,
            "team_a":     ta,
            "team_b":     tb,
            "matchup":    f"{ta} vs {tb}",
            "score_line": f"{score_a:.0f} – {score_b:.0f}",
            "win_prob_a": wpa,
            "win_prob_b": wpb,
            # spread
            "model_spread":  model_spread,
            "mkt_spread_ta": mkt_spread_ta,
            "spread_edge":   spread_edge,
            "cov_prob":      cov_prob,
            "hk_pct":        hk,
            "bet_size":      bet_size,
            "tier_label":    tier_label,
            "tier_emoji":    tier_emoji,
            "tier_order":    tier_order,
            # total
            "model_total":   round(model_total, 1),
            "mkt_total":     mkt_total,
            "total_edge":    total_edge,
        })

    return pd.DataFrame(rows), errors


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Upcoming CBB Games")

hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.caption(
        f"{season_label(current_year)} · {data_as_of(current_year)} · "
        f"Auto-refreshes every 2 min · {datetime.now().strftime('%H:%M:%S')}"
    )
with hcol2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Loading games..."):
    df, errors = build_projections(round_context, current_year)

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
    view = view[view["spread_edge"].abs() >= min_edge]

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
best = df[df["tier_order"] <= 1].sort_values("spread_edge", key=lambda s: s.abs(), ascending=False)
if not best.empty:
    st.divider()
    st.subheader("🏆 Best Bets")
    cols = st.columns(min(len(best), 3))
    for idx, (_, b) in enumerate(best.iterrows()):
        with cols[idx % len(cols)]:
            edge = b["spread_edge"]
            direction = b["team_a"] if edge > 0 else b["team_b"]
            spread_str = f"{b['team_a']} {-b['model_spread']:+.1f}"
            mkt_str    = f"{b['team_a']} {-b['mkt_spread_ta']:+.1f}" if b["mkt_spread_ta"] is not None else "—"
            color = "#1a472a" if b["tier_label"] == "Strong" else "#1e3a5f"
            st.markdown(
                f"""<div style='background:{color};border-radius:10px;padding:14px;margin-bottom:8px'>
                <div style='font-size:1.1rem;font-weight:bold'>{b['tier_emoji']} {b['matchup']}</div>
                <div style='color:#ccc;font-size:0.85rem'>{b['date_label']} · {b['time']}</div>
                <hr style='border-color:#ffffff22;margin:8px 0'>
                <div style='display:flex;justify-content:space-between'>
                  <div><div style='color:#aaa;font-size:0.75rem'>MODEL</div>
                       <div style='font-size:1rem;font-weight:bold'>{spread_str}</div></div>
                  <div><div style='color:#aaa;font-size:0.75rem'>MARKET</div>
                       <div style='font-size:1rem'>{mkt_str}</div></div>
                  <div><div style='color:#aaa;font-size:0.75rem'>EDGE</div>
                       <div style='font-size:1rem;font-weight:bold;color:#2ecc71'>{edge:+.1f} pts</div></div>
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
    day_df = view[view["date_key"] == date].sort_values(["tier_order", "spread_edge"],
        key=lambda s: s if s.name != "spread_edge" else s.abs(), ascending=[True, False])

    with tab:
        if day_df.empty:
            st.caption("No games match current filters for this day.")
            continue

        for _, row in day_df.iterrows():
            tier_colors = {
                "Strong": ("#1a472a", "#2ecc71"),
                "Value":  ("#1e3a5f", "#3498db"),
                "Lean":   ("#3d2b1f", "#f39c12"),
                "Pass":   ("#1e1e1e", "#7f8c8d"),
            }
            bg, accent = tier_colors.get(row["tier_label"], ("#1e1e1e", "#aaa"))

            edge_str = f"{row['spread_edge']:+.1f}" if row["spread_edge"] is not None else "—"
            mkt_str  = f"{row['team_a']} {-row['mkt_spread_ta']:+.1f}" if row["mkt_spread_ta"] is not None else "—"
            ou_str   = (f"O/U: model {row['model_total']} vs {row['mkt_total']} "
                        f"({'Over' if row['total_edge'] and row['total_edge'] > 0 else 'Under'} "
                        f"{abs(row['total_edge']):.1f})" if row["total_edge"] is not None else "")

            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])

            with c1:
                st.markdown(
                    f"<div style='padding:4px 0'>"
                    f"<span style='font-weight:bold;font-size:1rem'>{row['matchup']}</span><br>"
                    f"<span style='color:#aaa;font-size:0.8rem'>{row['time']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>PROJECTED</div>"
                    f"<div style='font-size:0.95rem'>{row['team_a']} {-row['model_spread']:+.1f}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>{row['score_line']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>MARKET / EDGE</div>"
                    f"<div style='font-size:0.95rem'>{mkt_str}</div>"
                    f"<div style='color:{accent};font-weight:bold'>{edge_str} pts</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c4:
                win_pct = f"{row['win_prob_a']:.0%} / {row['win_prob_b']:.0%}"
                cov_str = f"{row['cov_prob']:.1%}" if row["cov_prob"] else "—"
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>WIN PROB</div>"
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
                    f"<div style='font-size:0.9rem'>{bet_str}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>{hk_str} Kelly</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if ou_str:
                st.caption(f"&nbsp;&nbsp;&nbsp;{ou_str}")
            st.divider()

# ── Errors ────────────────────────────────────────────────────────────────────
if errors:
    with st.expander(f"⚠️ {len(errors)} games skipped (team name not in DB)"):
        for e in errors:
            st.caption(e)
        st.caption("Run data ingestion or check team names on the Teams page.")
