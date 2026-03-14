import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.model.predict import project_game, spread_to_win_prob, season_label, data_as_of

st.set_page_config(page_title="Bracket Projector", page_icon="🔢", layout="wide")
st.title("🔢 Bracket Projector")

current_year = TOURNAMENT_YEARS[-1]
season = season_label(current_year)
data_note = data_as_of(current_year)
st.caption(f"**{season} season** · {data_note}")

ROUNDS = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}

# ── Instructions ──────────────────────────────────────────────────────────────
with st.expander("How to use", expanded=False):
    st.markdown("""
    1. Enter teams and seeds for each region (up to 16 teams per region, but only first 4 per slot matter for R64)
    2. Click **Project Bracket** to simulate each round using the model
    3. The model projects the most likely winner of each matchup
    4. Win probability is shown for every matchup
    5. Use **Monte Carlo** to see round-advancement probabilities based on thousands of simulations
    """)

# ── Team input ────────────────────────────────────────────────────────────────
st.subheader("Enter Bracket Teams")

REGIONS = ["East", "West", "South", "Midwest"]

# Try to load teams from DB for autocomplete
@st.cache_data(ttl=600)
def get_team_list(year: int):
    from src.utils.db import query_df
    df = query_df("SELECT DISTINCT team FROM torvik_ratings WHERE year = ? ORDER BY team", params=[year])
    return df["team"].tolist() if not df.empty else []

team_list = get_team_list(current_year)

# Default bracket (common 1-seeds for illustration)
DEFAULT_TEAMS = {
    "East": [
        ("Duke", 1), ("Baylor", 16), ("Alabama", 8), ("St. John's", 9),
        ("Michigan St.", 5), ("New Mexico", 12), ("Mississippi St.", 4), ("BYU", 13),
        ("Wisconsin", 6), ("Montana", 11), ("Arizona", 3), ("Akron", 14),
        ("Troy", 7), ("Ole Miss", 10), ("Maryland", 2), ("Grand Canyon", 15),
    ],
    "West": [
        ("Auburn", 1), ("Alabama St.", 16), ("Louisville", 8), ("Creighton", 9),
        ("Michigan", 5), ("UC San Diego", 12), ("Texas A&M", 4), ("Yale", 13),
        ("St. Mary's", 6), ("VCU", 11), ("Iowa St.", 3), ("Lipscomb", 14),
        ("Marquette", 7), ("New Mexico St.", 10), ("Michigan St.", 2), ("Bryant", 15),
    ],
    "South": [
        ("Florida", 1), ("Norfolk St.", 16), ("UConn", 8), ("Oklahoma", 9),
        ("Memphis", 5), ("Colorado St.", 12), ("Maryland", 4), ("Grand Canyon", 13),
        ("Missouri", 6), ("Drake", 11), ("Texas Tech", 3), ("UNCW", 14),
        ("Kansas", 7), ("Arkansas", 10), ("Tennessee", 2), ("Wofford", 15),
    ],
    "Midwest": [
        ("Houston", 1), ("SIU Edwardsville", 16), ("Gonzaga", 8), ("Georgia", 9),
        ("Clemson", 5), ("McNeese", 12), ("Purdue", 4), ("High Point", 13),
        ("Illinois", 6), ("TCU", 11), ("Kentucky", 3), ("Troy", 14),
        ("UCLA", 7), ("Utah St.", 10), ("Tennessee", 2), ("Wofford", 15),
    ],
}

region_teams = {}
tabs = st.tabs(REGIONS)

for region, tab in zip(REGIONS, tabs):
    with tab:
        st.caption(f"Enter 16 teams seeded 1–16 for the **{region}** region")
        defaults = DEFAULT_TEAMS.get(region, [("", i + 1) for i in range(16)])
        pairs = []
        cols_per_row = 2
        for i in range(0, 16, cols_per_row):
            row_cols = st.columns(cols_per_row * 2)
            for j in range(cols_per_row):
                idx = i + j
                if idx >= 16:
                    break
                default_team, default_seed = defaults[idx] if idx < len(defaults) else ("", idx + 1)
                with row_cols[j * 2]:
                    seed = st.number_input(f"Seed", min_value=1, max_value=16,
                                           value=default_seed,
                                           key=f"{region}_{idx}_seed",
                                           label_visibility="collapsed")
                with row_cols[j * 2 + 1]:
                    team = st.text_input(f"Team", value=default_team,
                                         key=f"{region}_{idx}_team",
                                         label_visibility="collapsed",
                                         placeholder=f"#{default_seed} seed")
                if team:
                    pairs.append((team.strip(), int(seed)))
        region_teams[region] = sorted(pairs, key=lambda x: x[1])

st.divider()

# ── Simulation settings ───────────────────────────────────────────────────────
col_sim1, col_sim2 = st.columns([1, 3])
with col_sim1:
    n_sims = st.selectbox("Monte Carlo simulations", [100, 500, 1000, 5000], index=2)
with col_sim2:
    st.caption("Higher simulations = more accurate probabilities but slower. 1000 is a good balance.")

run_btn = st.button("🔢 Project Bracket", type="primary", use_container_width=True)


def simulate_region(teams_seeds: list, year: int, deterministic: bool = True) -> list:
    """
    Simulate a 16-team region bracket. Returns list of (team, seed) survivors per round.
    If deterministic=True, always takes the model's most-likely winner.
    If False, samples probabilistically.
    """
    current_round = list(teams_seeds)  # list of (team, seed)
    all_rounds = [current_round[:]]

    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])
                continue
            ta, sa = current_round[i]
            tb, sb = current_round[i + 1]
            # Assign favorite as team with lower seed number
            if sa <= sb:
                fav, fav_seed, dog, dog_seed = ta, sa, tb, sb
            else:
                fav, fav_seed, dog, dog_seed = tb, sb, ta, sa

            proj = project_game(fav, dog,
                                round_num=max(1, 7 - int(np.log2(len(current_round)))),
                                year=year, seed_a=fav_seed, seed_b=dog_seed)

            if "error" in proj:
                win_prob_fav = 0.5
            else:
                win_prob_fav = proj.get("win_prob_a", 0.5)

            if deterministic:
                winner = (fav, fav_seed) if win_prob_fav >= 0.5 else (dog, dog_seed)
            else:
                winner = (fav, fav_seed) if np.random.random() < win_prob_fav else (dog, dog_seed)
            next_round.append(winner)
        current_round = next_round
        all_rounds.append(current_round[:])

    return all_rounds


def simulate_final_four(region_winners: dict, year: int, deterministic: bool = True):
    """
    Simulate the Final Four + Championship. Region winners: {region: (team, seed)}
    Returns game results list and champion.
    """
    # Standard bracket: East vs West, South vs Midwest in F4
    matchups_f4 = [
        (region_winners["East"], region_winners["West"]),
        (region_winners["South"], region_winners["Midwest"]),
    ]
    f4_winners = []
    f4_games = []
    for (ta, sa), (tb, sb) in matchups_f4:
        if sa <= sb:
            fav, fav_seed, dog, dog_seed = ta, sa, tb, sb
        else:
            fav, fav_seed, dog, dog_seed = tb, sb, ta, sa
        proj = project_game(fav, dog, round_num=5, year=year, seed_a=fav_seed, seed_b=dog_seed)
        if "error" in proj:
            win_prob = 0.5
            spread = 0.0
        else:
            win_prob = proj.get("win_prob_a", 0.5)
            spread = proj.get("projected_spread", 0.0)

        if deterministic:
            winner = (fav, fav_seed) if win_prob >= 0.5 else (dog, dog_seed)
        else:
            winner = (fav, fav_seed) if np.random.random() < win_prob else (dog, dog_seed)
        f4_winners.append(winner)
        f4_games.append({
            "Round": "F4",
            "Fav": fav, "Dog": dog,
            "Win Prob": f"{win_prob:.1%}",
            "Model Spread": f"{fav} {spread:+.1f}",
            "Winner": winner[0],
        })

    # Championship
    (ca, csa), (cb, csb) = f4_winners[0], f4_winners[1]
    if csa <= csb:
        fav, fav_seed, dog, dog_seed = ca, csa, cb, csb
    else:
        fav, fav_seed, dog, dog_seed = cb, csb, ca, csa
    proj = project_game(fav, dog, round_num=6, year=year, seed_a=fav_seed, seed_b=dog_seed)
    if "error" in proj:
        win_prob = 0.5
        spread = 0.0
    else:
        win_prob = proj.get("win_prob_a", 0.5)
        spread = proj.get("projected_spread", 0.0)

    if deterministic:
        champ = (fav, fav_seed) if win_prob >= 0.5 else (dog, dog_seed)
    else:
        champ = (fav, fav_seed) if np.random.random() < win_prob else (dog, dog_seed)

    f4_games.append({
        "Round": "Champ",
        "Fav": fav, "Dog": dog,
        "Win Prob": f"{win_prob:.1%}",
        "Model Spread": f"{fav} {spread:+.1f}",
        "Winner": champ[0],
    })
    return f4_games, champ


if run_btn:
    # Validate input
    valid = True
    for region in REGIONS:
        if len(region_teams[region]) < 2:
            st.error(f"{region}: Need at least 2 teams")
            valid = False
    if not valid:
        st.stop()

    # ── Deterministic projection ──────────────────────────────────────────────
    st.subheader("Projected Bracket")

    region_all_rounds = {}
    region_winners_det = {}

    for region in REGIONS:
        with st.spinner(f"Projecting {region} region..."):
            teams = region_teams[region]
            # Pad to 16 if needed
            while len(teams) < 16:
                teams.append((f"TBD#{len(teams)+1}", len(teams) + 1))
            all_rounds = simulate_region(teams[:16], current_year, deterministic=True)
            region_all_rounds[region] = all_rounds
            region_winners_det[region] = all_rounds[-1][0]

    # Show region results
    region_tabs = st.tabs(REGIONS + ["Final Four"])
    for region, rtab in zip(REGIONS, region_tabs[:4]):
        with rtab:
            all_rounds = region_all_rounds[region]
            round_labels = ["R64 → R32", "R32 → S16", "S16 → E8", "E8 (Winner)"]

            for ri, survivors in enumerate(all_rounds[1:], 1):
                label = round_labels[min(ri - 1, len(round_labels) - 1)]
                prev = all_rounds[ri - 1]
                st.markdown(f"**{label}**")
                cols = st.columns(min(len(survivors), 4))
                for ci, (team, seed) in enumerate(survivors):
                    with cols[ci % len(cols)]:
                        # Find opponent from previous round
                        prev_idx = ci * (len(prev) // len(survivors))
                        opp_pairs = prev[prev_idx:prev_idx + (len(prev) // len(survivors))]
                        opponents = [t for t, s in opp_pairs if t != team]
                        opp_str = opponents[0] if opponents else "?"
                        st.success(f"#{seed} **{team}**\ndef. {opp_str}")

            winner, winner_seed = region_winners_det[region]
            st.markdown(f"### 🏆 {region} Winner: **{winner}** (#{winner_seed} seed)")

    with region_tabs[4]:  # Final Four tab
        st.subheader("Final Four & Championship")
        with st.spinner("Projecting Final Four..."):
            f4_games, champion = simulate_final_four(region_winners_det, current_year, deterministic=True)

        f4_df = pd.DataFrame(f4_games)
        st.dataframe(f4_df, use_container_width=True, hide_index=True)
        st.markdown(f"## 🏆 National Champion: **{champion[0]}** (#{champion[1]} seed)")

    st.divider()

    # ── Monte Carlo simulation ────────────────────────────────────────────────
    st.subheader(f"Monte Carlo Probabilities ({n_sims:,} simulations)")

    champion_counts = defaultdict(int)
    round_counts = defaultdict(lambda: defaultdict(int))  # team -> round -> count

    progress = st.progress(0.0, text="Running simulations...")

    for sim in range(n_sims):
        sim_region_winners = {}
        for region in REGIONS:
            teams = region_teams[region][:16]
            while len(teams) < 16:
                teams.append((f"TBD#{len(teams)+1}", len(teams) + 1))
            sim_rounds = simulate_region(teams, current_year, deterministic=False)
            sim_region_winners[region] = sim_rounds[-1][0]
            # Track round advancement
            for ri, survivors in enumerate(sim_rounds[1:], 2):
                for team, seed in survivors:
                    round_counts[team][ri] += 1

        _, sim_champ = simulate_final_four(sim_region_winners, current_year, deterministic=False)
        champion_counts[sim_champ[0]] += 1
        if (sim + 1) % 50 == 0:
            progress.progress((sim + 1) / n_sims, text=f"Simulation {sim+1}/{n_sims}...")

    progress.empty()

    # Champion probability chart
    champ_df = pd.DataFrame([
        {"Team": team, "Champion %": round(count / n_sims * 100, 1)}
        for team, count in sorted(champion_counts.items(), key=lambda x: -x[1])
        if count > 0
    ]).head(20)

    if not champ_df.empty:
        fig = px.bar(champ_df, x="Champion %", y="Team", orientation="h",
                     title=f"Championship Probability (top 20, {n_sims:,} sims)",
                     color="Champion %", color_continuous_scale="Blues")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Round advancement table
    all_teams = set()
    for region in REGIONS:
        all_teams.update(t for t, s in region_teams[region])

    adv_rows = []
    for team in sorted(all_teams):
        if not any(team in str(r) for r in round_counts):
            continue
        adv_rows.append({
            "Team": team,
            "R32 %": round(round_counts[team].get(2, 0) / n_sims * 100, 1),
            "S16 %": round(round_counts[team].get(3, 0) / n_sims * 100, 1),
            "E8 %": round(round_counts[team].get(4, 0) / n_sims * 100, 1),
            "F4 %": round(round_counts[team].get(5, 0) / n_sims * 100, 1),
            "Champ %": round(champion_counts.get(team, 0) / n_sims * 100, 1),
        })

    if adv_rows:
        adv_df = pd.DataFrame(adv_rows).sort_values("Champ %", ascending=False)
        st.subheader("Round Advancement Probabilities")
        st.dataframe(adv_df, use_container_width=True, hide_index=True)

else:
    st.info("Enter teams above and click **Project Bracket** to run projections.")
    st.caption(f"Data as of: {data_note}")
