import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import TOURNAMENT_YEARS
from src.model.predict import season_label, data_as_of
from src.utils.db import query_df

st.set_page_config(page_title="Teams", page_icon="📋", layout="wide")
st.title("📋 Teams")

# ── Season selector ───────────────────────────────────────────────────────────
col_season, col_filter, col_conf = st.columns([1, 2, 2])
with col_season:
    selected_year = st.selectbox(
        "Season",
        TOURNAMENT_YEARS[::-1],
        format_func=lambda y: f"{season_label(y)} ({y})",
    )
    st.caption(data_as_of(selected_year))

# ── Load ratings ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_ratings(year: int) -> pd.DataFrame:
    df = query_df(
        "SELECT * FROM torvik_ratings WHERE year = ? ORDER BY barthag DESC",
        params=[year],
    )
    return df


df = load_ratings(selected_year)

if df.empty:
    st.warning(f"No ratings found for {selected_year}. Run data ingestion first.")
    st.stop()

# Load tournament teams for this year
@st.cache_data(ttl=300)
def get_tournament_teams(year: int) -> set:
    tdf = query_df(
        "SELECT DISTINCT team1, team2 FROM historical_results WHERE year = ?",
        params=[year],
    )
    teams = set()
    if not tdf.empty:
        teams.update(tdf["team1"].dropna())
        teams.update(tdf["team2"].dropna())
    return teams


tournament_teams = get_tournament_teams(selected_year)

# Add tournament flag
df["In Tournament"] = df["team"].isin(tournament_teams)

# ── Filters ───────────────────────────────────────────────────────────────────
with col_filter:
    show_filter = st.radio("Show", ["All Teams", "Tournament Teams Only"], horizontal=True)

with col_conf:
    conferences = sorted(df["conf"].dropna().unique().tolist())
    selected_conf = st.multiselect("Conference filter", conferences, placeholder="All conferences")

# Apply filters
display_df = df.copy()
if show_filter == "Tournament Teams Only":
    display_df = display_df[display_df["In Tournament"]]
if selected_conf:
    display_df = display_df[display_df["conf"].isin(selected_conf)]

# ── Team search ───────────────────────────────────────────────────────────────
search = st.text_input("Search team", placeholder="Type a team name...")
if search:
    display_df = display_df[display_df["team"].str.contains(search, case=False, na=False)]

st.divider()

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Teams", len(display_df))
k2.metric("Tournament Teams", display_df["In Tournament"].sum())
k3.metric("Conferences", display_df["conf"].nunique())
k4.metric("Season", season_label(selected_year))

st.divider()

# ── Main ratings table ────────────────────────────────────────────────────────
st.subheader(f"Ratings — {season_label(selected_year)} Season")

display_cols = {
    "team": "Team",
    "conf": "Conf",
    "seed": "Seed",
    "barthag": "Barthag",
    "adj_o": "Adj O",
    "adj_d": "Adj D",
    "adj_t": "Tempo",
    "efg_o": "eFG% O",
    "efg_d": "eFG% D",
    "to_rate_o": "TO% O",
    "to_rate_d": "TO% D",
    "or_rate_o": "OR%",
    "three_pt_pct_o": "3P% O",
    "two_pt_pct_o": "2P% O",
    "sos": "SOS",
    "In Tournament": "Tourney",
}

show_df = display_df[[c for c in display_cols if c in display_df.columns]].copy()
show_df = show_df.rename(columns=display_cols)

# Round numeric columns to appropriate precision
if "Barthag" in show_df.columns:
    show_df["Barthag"] = show_df["Barthag"].round(3)
for col in ["Adj O", "Adj D", "Tempo", "eFG% O", "eFG% D", "TO% O", "TO% D", "OR%", "3P% O", "2P% O"]:
    if col in show_df.columns:
        show_df[col] = show_df[col].round(1)
if "SOS" in show_df.columns:
    show_df["SOS"] = show_df["SOS"].round(2)
if "Seed" in show_df.columns:
    show_df["Seed"] = show_df["Seed"].where(show_df["Seed"].isna(), show_df["Seed"].astype("Int64"))

# Highlight tournament teams
def highlight_tourney(row):
    if row.get("Tourney", False):
        return ["background-color: #1a3a1a"] * len(row)
    return [""] * len(row)

st.dataframe(
    show_df.style.apply(highlight_tourney, axis=1),
    use_container_width=True,
    hide_index=True,
    height=500,
)
st.caption("Green highlight = tournament team. Sorted by Barthag (overall efficiency) descending.")
st.caption("Tournament Teams count reflects seeds stored in DB. Use Bracket Projector → Save to DB to populate seeds.")

st.divider()

# ── Top 20 chart ──────────────────────────────────────────────────────────────
st.subheader("Top 25 by Barthag")

top25 = display_df.nlargest(25, "barthag").copy()
top25["label"] = top25.apply(
    lambda r: f"#{int(r['seed'])} {r['team']}" if pd.notna(r.get("seed")) and r["seed"] > 0
    else r["team"], axis=1,
)

fig_bar = px.bar(
    top25,
    x="barthag",
    y="label",
    orientation="h",
    color="In Tournament",
    color_discrete_map={True: "#2ecc71", False: "#7f8c8d"},
    title=f"Top 25 Teams by Barthag — {season_label(selected_year)}",
    labels={"barthag": "Barthag (Overall Efficiency)", "label": "Team"},
)
fig_bar.update_layout(
    yaxis={"categoryorder": "total ascending"},
    legend_title="In Tournament",
    height=600,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Offense vs Defense scatter ────────────────────────────────────────────────
st.subheader("Offense vs. Defense Efficiency")

scatter_df = display_df[display_df["adj_o"].notna() & display_df["adj_d"].notna()].copy()
scatter_df["label"] = scatter_df.apply(
    lambda r: f"#{int(r['seed'])} {r['team']}" if pd.notna(r.get("seed")) and r["seed"] > 0
    else r["team"], axis=1,
)

if not scatter_df.empty:
    fig_scatter = px.scatter(
        scatter_df,
        x="adj_d",
        y="adj_o",
        color="In Tournament",
        color_discrete_map={True: "#2ecc71", False: "#95a5a6"},
        hover_name="label",
        hover_data={"adj_o": ":.1f", "adj_d": ":.1f", "barthag": ":.3f"},
        title="Offensive Efficiency vs. Defensive Efficiency",
        labels={"adj_d": "Adj. Defense (lower = better)", "adj_o": "Adj. Offense (higher = better)"},
        size="barthag",
        size_max=15,
    )
    # Quadrant lines at medians
    med_o = scatter_df["adj_o"].median()
    med_d = scatter_df["adj_d"].median()
    fig_scatter.add_hline(y=med_o, line_dash="dash", line_color="white", opacity=0.3)
    fig_scatter.add_vline(x=med_d, line_dash="dash", line_color="white", opacity=0.3)
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Best teams = top-left (great offense, great defense). Size = Barthag.")

st.divider()

# ── Conference comparison ─────────────────────────────────────────────────────
st.subheader("Conference Averages")

conf_df = (
    display_df.groupby("conf")
    .agg(
        Teams=("team", "count"),
        Barthag=("barthag", "mean"),
        Adj_O=("adj_o", "mean"),
        Adj_D=("adj_d", "mean"),
        Tourney_Teams=("In Tournament", "sum"),
    )
    .round(2)
    .sort_values("Barthag", ascending=False)
    .reset_index()
    .rename(columns={"conf": "Conference", "Adj_O": "Avg Adj O", "Adj_D": "Avg Adj D"})
)

if not conf_df.empty:
    st.dataframe(conf_df, use_container_width=True, hide_index=True)

    # Bar chart of conference Barthag
    top_confs = conf_df[conf_df["Teams"] >= 3].head(20)
    if not top_confs.empty:
        fig_conf = px.bar(
            top_confs,
            x="Barthag",
            y="Conference",
            orientation="h",
            title="Average Barthag by Conference (min. 3 teams)",
            color="Barthag",
            color_continuous_scale="Blues",
        )
        fig_conf.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_conf, use_container_width=True)
