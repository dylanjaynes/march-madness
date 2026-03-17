import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.model.predict import (
    project_game, coverage_probability, kelly_fraction, half_kelly,
    bet_tier, season_label, data_as_of,
)
from src.utils.db import query_df

st.set_page_config(page_title="Matchup Builder", page_icon="⚔️", layout="wide")
st.title("⚔️ Matchup Builder")

current_year = TOURNAMENT_YEARS[-1]
season = season_label(current_year)
data_note = data_as_of(current_year)
st.caption(f"**{season} season** · {data_note}")


@st.cache_data(ttl=3600)
def load_teams(year):
    df = query_df(
        "SELECT DISTINCT team FROM torvik_ratings WHERE year=? ORDER BY team",
        params=[year],
    )
    return df["team"].tolist() if not df.empty else []


# ── Team input ────────────────────────────────────────────────────────────────
with st.form("matchup_form"):
    st.subheader("Build a Matchup")
    col1, col2 = st.columns(2)

    # Determine teams list using current_year as default; will be overridden
    # by proj_year after the form is submitted, but for display we use current_year
    teams_list = load_teams(current_year)
    default_a = "Duke" if "Duke" in teams_list else (teams_list[0] if teams_list else "Duke")
    default_b = "Houston" if "Houston" in teams_list else (teams_list[1] if len(teams_list) > 1 else "Houston")

    with col1:
        team_a = st.selectbox(
            "Team A",
            teams_list,
            index=teams_list.index(default_a) if default_a in teams_list else 0,
            key="matchup_team_a",
        )
    with col2:
        team_b = st.selectbox(
            "Team B",
            teams_list,
            index=teams_list.index(default_b) if default_b in teams_list else 0,
            key="matchup_team_b",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        round_num = st.selectbox(
            "Tournament Round",
            [1, 2, 3, 4, 5, 6],
            format_func=lambda r: ROUND_NAMES.get(r, f"R{r}"),
        )
    with col4:
        market_spread = st.number_input(
            "Team A spread (optional)", value=0.0, step=0.5,
            help="Betting convention: -7 = Team A is 7pt favorite, +7 = 7pt underdog. Leave 0 to skip.",
        )
    with col5:
        market_total = st.number_input(
            "Market O/U Total (optional)", value=0.0, step=0.5,
            help="The over/under line. Leave 0 to skip.",
        )

    col6, col7 = st.columns(2)
    with col6:
        bankroll = st.number_input("Bankroll ($) for Kelly sizing", value=1000, step=100)
    with col7:
        proj_year = st.selectbox(
            "Season data",
            TOURNAMENT_YEARS[::-1],
            format_func=lambda y: f"{season_label(y)} ({y})",
        )

    submitted = st.form_submit_button("Project Matchup", type="primary", use_container_width=True)

# ── Projection ────────────────────────────────────────────────────────────────
if submitted:
    # Seeds are auto-looked up from DB — never user-controlled,
    # because seed_diff is a model feature and arbitrary seeds distort predictions.
    with st.spinner("Running projection..."):
        proj = project_game(
            team_a, team_b, round_num,
            year=proj_year,
            seed_a=None, seed_b=None,
        )

    if "error" in proj:
        st.error(f"Projection error: {proj['error']}")
        st.caption("Make sure both team names match those in the Torvik ratings database (see Teams page).")
        st.stop()

    # Use the canonical team order returned by project_game (may differ from form input)
    team_a = proj["team_a"]
    team_b = proj["team_b"]
    seed_a = proj["seed_a"] if proj.get("seed_a") is not None else seed_a
    seed_b = proj["seed_b"] if proj.get("seed_b") is not None else seed_b

    model_spread = proj["projected_spread"]   # model convention: positive = team_a wins
    pt  = proj["projected_total"]
    wpa = proj["win_prob_a"]
    wpb = proj["win_prob_b"]
    score_a = proj["projected_score_a"]
    score_b = proj["projected_score_b"]

    # Convert market input from betting convention (−7 = team_a favored)
    # to model convention (+7 = team_a favored by 7) for all calculations
    mkt_model = -market_spread if market_spread != 0.0 else None
    mkt_t = market_total if market_total != 0.0 else None

    spread_edge = model_spread - mkt_model if mkt_model is not None else None
    total_edge  = pt - mkt_t if mkt_t is not None else None
    cov_prob    = coverage_probability(model_spread, mkt_model) if mkt_model is not None else None
    tier_label, tier_emoji = bet_tier(spread_edge, cov_prob) if spread_edge is not None else ("—", "")

    # Display values in betting convention (negate model spread so favorite = negative)
    ps = -model_spread

    hk = half_kelly(cov_prob) if cov_prob is not None else None
    fk = kelly_fraction(cov_prob) if cov_prob is not None else None

    st.divider()

    # ── Score card ────────────────────────────────────────────────────────────
    st.subheader(f"{team_a} vs {team_b} — {ROUND_NAMES.get(round_num, f'R{round_num}')}")
    st.caption(f"Projection using {season_label(proj_year)} ratings · {data_as_of(proj_year)}")

    proj_col, odds_col = st.columns([3, 2])

    with proj_col:
        st.markdown("#### Model Projection")
        c1, c2, c3 = st.columns(3)
        with c1:
            sign = "+" if ps >= 0 else ""
            st.metric("Model Spread", f"{team_a} {sign}{ps:.1f}")
        with c2:
            st.metric("Model Total", f"{pt:.1f}")
        with c3:
            st.metric(f"{team_a} Win Prob", f"{wpa:.1%}")

        # Projected score box
        st.markdown(
            f"""
            <div style='background:#1e1e2e;padding:16px;border-radius:8px;text-align:center;margin-top:8px'>
                <span style='font-size:2rem;font-weight:bold'>{team_a}</span>
                <span style='font-size:1.5rem;margin:0 12px;color:#aaa'>#{seed_a}</span>
                <span style='font-size:2.5rem;font-weight:bold;color:#3498db'>{score_a:.0f}</span>
                <span style='font-size:1.5rem;margin:0 16px;color:#aaa'>–</span>
                <span style='font-size:2.5rem;font-weight:bold;color:#e74c3c'>{score_b:.0f}</span>
                <span style='font-size:1.5rem;margin:0 12px;color:#aaa'>#{seed_b}</span>
                <span style='font-size:2rem;font-weight:bold'>{team_b}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Win probability bar
        st.markdown("#### Win Probability")
        fig_prob = go.Figure(go.Bar(
            x=[wpa * 100, wpb * 100],
            y=[team_a, team_b],
            orientation="h",
            marker_color=["#3498db", "#e74c3c"],
            text=[f"{wpa:.1%}", f"{wpb:.1%}"],
            textposition="inside",
        ))
        fig_prob.update_layout(
            height=120, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with odds_col:
        st.markdown("#### Betting Analysis")
        if mkt_model is not None:
            # Display market spread in betting convention (same as user typed)
            st.metric("Market Spread", f"{team_a} {market_spread:+.1f}")
            # Positive edge = team_a covers; negative = team_b value
            bet_on = team_a if spread_edge and spread_edge > 0 else team_b
            st.metric("Edge vs. Market", f"{spread_edge:+.1f} pts",
                      delta=f"{tier_emoji} {tier_label} — lean {bet_on}")
            st.metric("Coverage Probability", f"{cov_prob:.1%}")
            st.markdown("---")
            st.metric("Half Kelly", f"{hk*100:.1f}%", f"${bankroll * hk:,.0f}")
            st.metric("Full Kelly", f"{fk*100:.1f}%", f"${bankroll * fk:,.0f}")
        else:
            st.info("Enter a market spread to see edge and Kelly sizing.")

        if mkt_t is not None:
            st.markdown("---")
            direction = "Over" if pt > mkt_t else "Under"
            st.metric("Total Edge", f"{direction} by {abs(total_edge):.1f}",
                      f"Model {pt:.1f} vs {mkt_t:.1f}")

    st.divider()

    # ── Team ratings comparison ───────────────────────────────────────────────
    st.subheader("Team Ratings Comparison")

    @st.cache_data(ttl=600)
    def get_team_ratings(team: str, year: int):
        from src.utils.db import query_df
        df = query_df(
            "SELECT * FROM torvik_ratings WHERE year = ? AND team = ? LIMIT 1",
            params=[year, team],
        )
        return df.iloc[0].to_dict() if not df.empty else {}

    ra = get_team_ratings(team_a, proj_year)
    rb = get_team_ratings(team_b, proj_year)

    if ra and rb:
        metrics = [
            ("Adj. Offense", "adj_o", True),
            ("Adj. Defense", "adj_d", False),  # lower is better
            ("Tempo", "adj_t", True),
            ("Barthag", "barthag", True),
            ("eFG% Off", "efg_o", True),
            ("eFG% Def", "efg_d", False),
            ("TO Rate Off", "to_rate_o", False),
            ("TO Rate Def", "to_rate_d", True),
            ("OR Rate", "or_rate_o", True),
            ("3P%", "three_pt_pct_o", True),
            ("2P%", "two_pt_pct_o", True),
            ("SOS", "sos", True),
        ]

        comp_rows = []
        for label, key, higher_is_better in metrics:
            va = ra.get(key)
            vb = rb.get(key)
            if va is None or vb is None:
                continue
            if higher_is_better:
                edge_a = va > vb
            else:
                edge_a = va < vb
            comp_rows.append({
                "Metric": label,
                team_a: round(float(va), 2),
                team_b: round(float(vb), 2),
                "Edge": team_a if edge_a else team_b,
            })

        comp_df = pd.DataFrame(comp_rows)
        team_a_edge_count = (comp_df["Edge"] == team_a).sum()
        team_b_edge_count = (comp_df["Edge"] == team_b).sum()

        e1, e2 = st.columns(2)
        e1.metric(f"{team_a} Statistical Edges", team_a_edge_count,
                  f"of {len(comp_df)} metrics")
        e2.metric(f"{team_b} Statistical Edges", team_b_edge_count,
                  f"of {len(comp_df)} metrics")

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Radar chart
        radar_metrics = ["adj_o", "barthag", "efg_o", "or_rate_o", "three_pt_pct_o"]
        radar_labels = ["Off. Efficiency", "Barthag", "eFG%", "Reb Rate", "3P%"]

        def normalize(val, mn, mx):
            if mx == mn:
                return 0.5
            return (val - mn) / (mx - mn)

        fig_radar = go.Figure()
        for team, ratings, color in [(team_a, ra, "#3498db"), (team_b, rb, "#e74c3c")]:
            vals = []
            for key in radar_metrics:
                va = ratings.get(key, 0) or 0
                vb_other = (rb if team == team_a else ra).get(key, 0) or 0
                mn, mx = min(va, vb_other) - 1, max(va, vb_other) + 1
                vals.append(normalize(va, mn, mx))
            vals.append(vals[0])  # close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_labels + [radar_labels[0]],
                name=team,
                line_color=color,
                fill="toself",
                opacity=0.6,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False)),
            title="Team Profile Comparison",
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        missing = []
        if not ra:
            missing.append(team_a)
        if not rb:
            missing.append(team_b)
        st.warning(f"No ratings found for: {', '.join(missing)}. Check team names on the Teams page.")

    # ── Head-to-head history ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Tournament History (Head-to-Head)")

    @st.cache_data(ttl=600)
    def get_h2h(ta: str, tb: str):
        from src.utils.db import query_df
        df = query_df(
            """SELECT year, round_name, game_date, team1, score1, score2, team2, winner, margin
               FROM historical_results
               WHERE (team1=? AND team2=?) OR (team1=? AND team2=?)
               ORDER BY year DESC""",
            params=[ta, tb, tb, ta],
        )
        return df

    h2h = get_h2h(team_a, team_b)
    if h2h.empty:
        st.info("No head-to-head tournament matchups found in historical data.")
    else:
        st.dataframe(h2h, use_container_width=True, hide_index=True)

else:
    st.info("Fill in team names above and click **Project Matchup** to get started.")
    st.caption("Team names must match the BartTorvik ratings database. Use the Teams page to look up names.")
