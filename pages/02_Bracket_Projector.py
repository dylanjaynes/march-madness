import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict

from src.utils.config import TOURNAMENT_YEARS
from src.model.predict import project_game, season_label, data_as_of

st.set_page_config(page_title="Bracket Projector", page_icon="🔢", layout="wide")
st.title("🔢 Bracket Projector")

current_year = TOURNAMENT_YEARS[-1]
season = season_label(current_year)
data_note = data_as_of(current_year)
st.caption(f"**{season} season** · {data_note}")

REGIONS = ["East", "West", "South", "Midwest"]

# Standard NCAA bracket first-round matchup pairs, in bracket order.
# Each tuple = (top_seed, bottom_seed). Pairs play each other in R64.
# Bracket order determines R32 matchups: (1/16 winner) vs (8/9 winner), etc.
R64_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# ── Team list for autocomplete ─────────────────────────────────────────────────
@st.cache_data(ttl=600)
def get_team_list(year: int):
    from src.utils.db import query_df
    df = query_df(
        "SELECT DISTINCT team FROM torvik_ratings WHERE year = ? ORDER BY team",
        params=[year],
    )
    return df["team"].tolist() if not df.empty else []

team_list = get_team_list(current_year)

# ── Default 2025-26 bracket ────────────────────────────────────────────────────
DEFAULT_TEAMS = {
    "East": {
        1: "Duke", 16: "Baylor", 8: "Alabama", 9: "St. John's",
        5: "Michigan St.", 12: "New Mexico", 4: "Mississippi St.", 13: "BYU",
        6: "Wisconsin", 11: "Montana", 3: "Arizona", 14: "Akron",
        7: "Troy", 10: "Ole Miss", 2: "Maryland", 15: "Grand Canyon",
    },
    "West": {
        1: "Auburn", 16: "Alabama St.", 8: "Louisville", 9: "Creighton",
        5: "Michigan", 12: "UC San Diego", 4: "Texas A&M", 13: "Yale",
        6: "St. Mary's", 11: "VCU", 3: "Iowa St.", 14: "Lipscomb",
        7: "Marquette", 10: "New Mexico St.", 2: "Michigan St.", 15: "Bryant",
    },
    "South": {
        1: "Florida", 16: "Norfolk St.", 8: "UConn", 9: "Oklahoma",
        5: "Memphis", 12: "Colorado St.", 4: "Maryland", 13: "Grand Canyon",
        6: "Missouri", 11: "Drake", 3: "Texas Tech", 14: "UNCW",
        7: "Kansas", 10: "Arkansas", 2: "Tennessee", 15: "Wofford",
    },
    "Midwest": {
        1: "Houston", 16: "SIU Edwardsville", 8: "Gonzaga", 9: "Georgia",
        5: "Clemson", 12: "McNeese", 4: "Purdue", 13: "High Point",
        6: "Illinois", 11: "TCU", 3: "Kentucky", 14: "Troy",
        7: "UCLA", 10: "Utah St.", 2: "Connecticut", 15: "Tennessee St.",
    },
}


def _team_selectbox(region: str, seed: int, defaults: dict) -> str:
    """Render a searchable selectbox for a team slot. Returns team name string."""
    default_name = defaults.get(seed, "")
    # Find index of default in list (selectbox needs index)
    try:
        idx = team_list.index(default_name) + 1  # +1 because index 0 = ""
    except ValueError:
        idx = 0
    options = [""] + team_list
    chosen = st.selectbox(
        label=f"#{seed}",
        options=options,
        index=idx,
        key=f"bp_{region}_{seed}",
        label_visibility="collapsed",
    )
    return chosen.strip() if chosen else ""


# ── Team entry form ────────────────────────────────────────────────────────────
st.subheader("Enter Bracket Teams")
st.caption("Each row is a first-round matchup. Type to search for teams.")

region_seed_team: dict[str, dict[int, str]] = {}
tabs = st.tabs(REGIONS)

for region, tab in zip(REGIONS, tabs):
    with tab:
        defaults = DEFAULT_TEAMS.get(region, {})
        seed_team: dict[int, str] = {}

        # Header row
        hc = st.columns([1, 5, 1, 1, 5, 1])
        hc[0].markdown("**Seed**")
        hc[1].markdown("**Team**")
        hc[3].markdown("**Seed**")
        hc[4].markdown("**Team**")

        st.divider()

        for top_seed, bot_seed in R64_PAIRS:
            c = st.columns([1, 5, 1, 1, 5, 1])
            # Left team (top seed)
            c[0].markdown(f"**#{top_seed}**")
            with c[1]:
                top_team = _team_selectbox(region, top_seed, defaults)
            c[2].markdown("<div style='text-align:center;color:#888;padding-top:8px'>vs</div>",
                          unsafe_allow_html=True)
            # Right team (bottom seed)
            c[3].markdown(f"**#{bot_seed}**")
            with c[4]:
                bot_team = _team_selectbox(region, bot_seed, defaults)

            if top_team:
                seed_team[top_seed] = top_team
            if bot_team:
                seed_team[bot_seed] = bot_team

        region_seed_team[region] = seed_team

st.divider()

# ── Simulation settings ────────────────────────────────────────────────────────
col_sim1, col_sim2 = st.columns([1, 3])
with col_sim1:
    n_sims = st.selectbox("Monte Carlo simulations", [100, 500, 1000, 5000], index=2)
with col_sim2:
    st.caption("Higher simulations = more accurate probabilities but slower. 1000 is a good balance.")

run_btn = st.button("🔢 Project Bracket", type="primary", use_container_width=True)


# ── Simulation helpers ─────────────────────────────────────────────────────────
def _get_win_prob(ta: str, sa: int, tb: str, sb: int, round_num: int) -> float:
    """Return win prob for team_a vs team_b. team_a should be better seed."""
    proj = project_game(ta, tb, round_num=round_num, year=current_year)
    if "error" in proj:
        return 0.5
    return proj.get("win_prob_a", 0.5)


def simulate_region(seed_team: dict, deterministic: bool = True) -> list:
    """
    Simulate a 16-team region using standard NCAA bracket structure.
    seed_team: {seed: team_name}
    Returns list of rounds, each a list of (team, seed) survivors.
    """
    # Build initial bracket in correct order: pairs play consecutively
    ordered = []
    for top_seed, bot_seed in R64_PAIRS:
        top = (seed_team.get(top_seed, f"TBD{top_seed}"), top_seed)
        bot = (seed_team.get(bot_seed, f"TBD{bot_seed}"), bot_seed)
        ordered.append(top)
        ordered.append(bot)

    current_round = ordered[:]
    all_rounds = [current_round[:]]

    round_num = 1
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])
                continue
            ta, sa = current_round[i]
            tb, sb = current_round[i + 1]
            # Put better seed as team_a
            if sa <= sb:
                fav, fs, dog, ds = ta, sa, tb, sb
            else:
                fav, fs, dog, ds = tb, sb, ta, sa

            win_prob_fav = _get_win_prob(fav, fs, dog, ds, round_num)

            if deterministic:
                winner = (fav, fs) if win_prob_fav >= 0.5 else (dog, ds)
            else:
                winner = (fav, fs) if np.random.random() < win_prob_fav else (dog, ds)
            next_round.append(winner)
        current_round = next_round
        all_rounds.append(current_round[:])
        round_num += 1

    return all_rounds


def simulate_final_four(region_winners: dict, deterministic: bool = True):
    """East vs West, South vs Midwest in F4."""
    matchups = [
        (region_winners["East"], region_winners["West"], "East vs West"),
        (region_winners["South"], region_winners["Midwest"], "South vs Midwest"),
    ]
    f4_winners = []
    games = []
    for (ta, sa), (tb, sb), label in matchups:
        fav, fs, dog, ds = (ta, sa, tb, sb) if sa <= sb else (tb, sb, ta, sa)
        proj = project_game(fav, dog, round_num=5, year=current_year)
        if "error" in proj:
            wp, sp = 0.5, 0.0
        else:
            wp = proj.get("win_prob_a", 0.5)
            sp = proj.get("projected_spread", 0.0)

        if deterministic:
            winner = (fav, fs) if wp >= 0.5 else (dog, ds)
        else:
            winner = (fav, fs) if np.random.random() < wp else (dog, ds)
        f4_winners.append(winner)
        games.append({
            "Matchup": label, "Round": "Final Four",
            "Favorite": f"#{fs} {fav}", "Underdog": f"#{ds} {dog}",
            "Win Prob": f"{wp:.1%}", "Model Spread": f"{fav} {-sp:+.1f}",
            "Winner": f"#{winner[1]} {winner[0]}",
        })

    # Championship
    (ca, csa), (cb, csb) = f4_winners[0], f4_winners[1]
    fav, fs, dog, ds = (ca, csa, cb, csb) if csa <= csb else (cb, csb, ca, csa)
    proj = project_game(fav, dog, round_num=6, year=current_year)
    if "error" in proj:
        wp, sp = 0.5, 0.0
    else:
        wp = proj.get("win_prob_a", 0.5)
        sp = proj.get("projected_spread", 0.0)

    if deterministic:
        champ = (fav, fs) if wp >= 0.5 else (dog, ds)
    else:
        champ = (fav, fs) if np.random.random() < wp else (dog, ds)

    games.append({
        "Matchup": "Championship", "Round": "Championship",
        "Favorite": f"#{fs} {fav}", "Underdog": f"#{ds} {dog}",
        "Win Prob": f"{wp:.1%}", "Model Spread": f"{fav} {-sp:+.1f}",
        "Winner": f"#{champ[1]} {champ[0]}",
    })
    return games, champ


# ── Run projections ────────────────────────────────────────────────────────────
if run_btn:
    valid = True
    for region in REGIONS:
        filled = sum(1 for s, t in region_seed_team[region].items() if t)
        if filled < 4:
            st.error(f"{region}: Need at least 4 teams entered.")
            valid = False
    if not valid:
        st.stop()

    st.subheader("Projected Bracket")

    region_all_rounds: dict = {}
    region_winners_det: dict = {}

    for region in REGIONS:
        with st.spinner(f"Projecting {region}..."):
            region_all_rounds[region] = simulate_region(
                region_seed_team[region], deterministic=True
            )
            region_winners_det[region] = region_all_rounds[region][-1][0]

    # ── Display bracket results ────────────────────────────────────────────────
    result_tabs = st.tabs(REGIONS + ["🏆 Final Four"])

    round_labels = {
        1: "R64 → R32", 2: "R32 → S16", 3: "S16 → E8", 4: "Elite Eight"
    }

    for region, rtab in zip(REGIONS, result_tabs[:4]):
        with rtab:
            all_rounds = region_all_rounds[region]

            for ri in range(1, len(all_rounds)):
                label = round_labels.get(ri, f"Round {ri}")
                st.markdown(f"#### {label}")
                survivors = all_rounds[ri]
                prev = all_rounds[ri - 1]

                # Show game cards: each survivor beat one of two teams from prev round
                prev_per_winner = len(prev) // len(survivors)
                cols = st.columns(len(survivors))
                for ci, (team, seed) in enumerate(survivors):
                    with cols[ci]:
                        # Find the opponent they beat
                        start = ci * prev_per_winner
                        prev_slot = prev[start: start + prev_per_winner]
                        opp = next((t for t, s in prev_slot if t != team), "?")
                        opp_seed = next((s for t, s in prev_slot if t != team), "?")
                        st.success(f"**#{seed} {team}**\ndef. #{opp_seed} {opp}")

                st.markdown("")  # spacer

            winner, wseed = region_winners_det[region]
            st.markdown(f"### 🏆 {region} Winner: **#{wseed} {winner}**")

    with result_tabs[4]:
        st.subheader("Final Four & Championship")
        with st.spinner("Projecting Final Four..."):
            f4_games, champion = simulate_final_four(region_winners_det, deterministic=True)

        f4_df = pd.DataFrame(f4_games)[["Round", "Matchup", "Favorite", "Underdog", "Win Prob", "Model Spread", "Winner"]]
        st.dataframe(f4_df, use_container_width=True, hide_index=True)
        st.markdown(f"## 🏆 National Champion: **{champion[0]}** (#{champion[1]} seed)")

    st.divider()

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    st.subheader(f"Monte Carlo Probabilities ({n_sims:,} simulations)")

    champion_counts: dict = defaultdict(int)
    round_adv: dict = defaultdict(lambda: defaultdict(int))  # team -> round -> count

    progress = st.progress(0.0, text="Running simulations...")
    for sim in range(n_sims):
        sim_winners = {}
        for region in REGIONS:
            sim_rounds = simulate_region(region_seed_team[region], deterministic=False)
            sim_winners[region] = sim_rounds[-1][0]
            for ri, survivors in enumerate(sim_rounds[1:], 2):
                for team, seed in survivors:
                    round_adv[team][ri] += 1

        _, sim_champ = simulate_final_four(sim_winners, deterministic=False)
        champion_counts[sim_champ[0]] += 1
        if (sim + 1) % max(1, n_sims // 20) == 0:
            progress.progress((sim + 1) / n_sims, text=f"Simulation {sim+1}/{n_sims}...")

    progress.empty()

    # Champion probability chart
    champ_df = pd.DataFrame([
        {"Team": t, "Champion %": round(c / n_sims * 100, 1)}
        for t, c in sorted(champion_counts.items(), key=lambda x: -x[1]) if c > 0
    ]).head(20)

    if not champ_df.empty:
        fig = px.bar(
            champ_df, x="Champion %", y="Team", orientation="h",
            title=f"Championship Probability — top 20 ({n_sims:,} sims)",
            color="Champion %", color_continuous_scale="Blues",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Round advancement table
    all_teams = {t for r in REGIONS for t in region_seed_team[r].values() if t}
    adv_rows = []
    for team in sorted(all_teams):
        adv_rows.append({
            "Team": team,
            "R32 %":  round(round_adv[team].get(2, 0) / n_sims * 100, 1),
            "S16 %":  round(round_adv[team].get(3, 0) / n_sims * 100, 1),
            "E8 %":   round(round_adv[team].get(4, 0) / n_sims * 100, 1),
            "F4 %":   round(round_adv[team].get(5, 0) / n_sims * 100, 1),
            "Champ %": round(champion_counts.get(team, 0) / n_sims * 100, 1),
        })

    if adv_rows:
        adv_df = pd.DataFrame(adv_rows).sort_values("Champ %", ascending=False)
        st.subheader("Round Advancement Probabilities")
        st.dataframe(adv_df, use_container_width=True, hide_index=True)

else:
    st.info("Select teams above and click **Project Bracket** to run projections.")
    st.caption(f"Data as of: {data_note}")
