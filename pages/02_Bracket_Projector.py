import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict

from src.utils.config import TOURNAMENT_YEARS
from src.model.predict import project_game, season_label, data_as_of
from src.ingest.bracket import fetch_and_store_bracket, load_bracket_from_db, get_bracket_status

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

# ── Bracket source: DB first, then hardcoded projections as fallback ───────────
# Hardcoded pre-Selection-Sunday projections (used only when DB has no data)
_PROJECTED_TEAMS = {
    "East": {
        1: "Duke", 16: "Baylor", 8: "Alabama", 9: "St. John's (NY)",
        5: "Michigan State", 12: "New Mexico", 4: "Mississippi State", 13: "BYU",
        6: "Wisconsin", 11: "Montana", 3: "Arizona", 14: "Akron",
        7: "Troy", 10: "Ole Miss", 2: "Maryland", 15: "Grand Canyon",
    },
    "West": {
        1: "Auburn", 16: "Alabama State", 8: "Louisville", 9: "Creighton",
        5: "Michigan", 12: "UC San Diego", 4: "Texas A&M", 13: "Yale",
        6: "Saint Mary's", 11: "VCU", 3: "Iowa State", 14: "Lipscomb",
        7: "Marquette", 10: "New Mexico State", 2: "Michigan State", 15: "Bryant",
    },
    "South": {
        1: "Florida", 16: "Norfolk State", 8: "Connecticut", 9: "Oklahoma",
        5: "Memphis", 12: "Colorado State", 4: "Maryland", 13: "Grand Canyon",
        6: "Missouri", 11: "Drake", 3: "Texas Tech", 14: "UNC Wilmington",
        7: "Kansas", 10: "Arkansas", 2: "Tennessee", 15: "Wofford",
    },
    "Midwest": {
        1: "Houston", 16: "SIU Edwardsville", 8: "Gonzaga", 9: "Georgia",
        5: "Clemson", 12: "McNeese State", 4: "Purdue", 13: "High Point",
        6: "Illinois", 11: "TCU", 3: "Kentucky", 14: "Troy",
        7: "UCLA", 10: "Utah State", 2: "Connecticut", 15: "Tennessee St.",
    },
}

# Load official bracket from DB if available, otherwise use projections
# Wrap in try/except in case tournament_bracket table doesn't exist yet on this deployment
try:
    from src.utils.db import init_db
    init_db()  # ensure tournament_bracket table exists
    _bracket_status = get_bracket_status(current_year)
    _db_bracket = load_bracket_from_db(current_year) if _bracket_status["stored"] else {}
except Exception:
    _bracket_status = {"stored": False, "n_teams": 0, "fetched_at": None}
    _db_bracket = {}
DEFAULT_TEAMS = _db_bracket if _db_bracket else _PROJECTED_TEAMS
_bracket_source = "official" if _db_bracket else "projected"

# ── Bracket fetch UI ──────────────────────────────────────────────────────────
_col_status, _col_btn = st.columns([3, 1])
with _col_status:
    if _bracket_source == "official":
        st.success(
            f"✅ **Official bracket loaded** · {_bracket_status['n_teams']} teams · "
            f"fetched {_bracket_status['fetched_at'][:16] if _bracket_status['fetched_at'] else ''}",
            icon=None,
        )
    else:
        st.info("📋 **Using projected bracket** — click to load the official bracket once released on Selection Sunday.", icon=None)

with _col_btn:
    if st.button("🔄 Load Official Bracket", use_container_width=True,
                 help="Fetches the official bracket from Sports Reference. Available after Selection Sunday."):
        with st.spinner("Fetching bracket from Sports Reference..."):
            try:
                bracket = fetch_and_store_bracket(current_year)
                if bracket:
                    n = sum(len(v) for v in bracket.values())
                    st.success(f"Loaded {n} teams across {len(bracket)} regions!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Bracket not yet available on Sports Reference. Try again after Selection Sunday.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")


def _resolve_default(name: str) -> str:
    """
    Find the best match for a default team name in team_list.
    Tries exact match first, then checks if any team_list entry is an alias
    for the same canonical name (handles DB naming variations like
    'Michigan St.' vs 'Michigan State').
    """
    if not name:
        return ""
    if name in team_list:
        return name
    # Try to resolve via team_map aliases
    try:
        from src.utils.team_map import normalize_team_name
        canonical = normalize_team_name(name)
        # canonical matches exact — find any team_list entry that normalizes to same canonical
        for t in team_list:
            if normalize_team_name(t) == canonical:
                return t
    except Exception:
        pass
    return ""  # not found — leave blank


def _team_selectbox(region: str, seed: int, defaults: dict) -> str:
    """Render a searchable selectbox for a team slot. Returns team name string."""
    default_name = _resolve_default(defaults.get(seed, ""))
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
    n_sims = st.selectbox("Monte Carlo simulations", [1000, 5000, 10000], index=0)
with col_sim2:
    st.caption("Win probabilities are pre-computed once, so simulations run instantly regardless of count.")

run_btn = st.button("🔢 Project Bracket", type="primary", use_container_width=True)


# ── Simulation helpers ─────────────────────────────────────────────────────────
# Win probability cache: (ta, tb) -> float.  Populated once before any simulation.
_wp_cache: dict = {}

def _get_win_prob(ta: str, sa: int, tb: str, sb: int, round_num: int,
                  use_cache: bool = False) -> float:
    """
    Return win prob for ta (better seed = lower number).
    use_cache=True: fast path for Monte Carlo — reads from pre-computed cache.
    use_cache=False: calls model directly with correct round_num (used for display).
    """
    if use_cache:
        # Cache is keyed (ta, tb) regardless of round — uses round 3 as proxy
        key = (ta, tb)
        if key in _wp_cache:
            return _wp_cache[key]
        rev = (tb, ta)
        if rev in _wp_cache:
            return 1.0 - _wp_cache[rev]
    proj = project_game(ta, tb, round_num=round_num, year=current_year, seed_a=sa, seed_b=sb)
    return 0.5 if "error" in proj else proj.get("win_prob_a", 0.5)


def _precompute_win_probs(seed_teams_by_region: dict) -> None:
    """
    Pre-compute all pairwise win probabilities for every team in the bracket.
    With 16 teams per region × 4 regions = 64 teams → at most ~2016 matchups.
    Runs in ~10-15 seconds once; Monte Carlo then uses only dict lookups.
    """
    all_teams = []  # list of (team, seed)
    for region, seed_team in seed_teams_by_region.items():
        for seed, team in seed_team.items():
            if team:
                all_teams.append((team, seed))

    total = len(all_teams) * (len(all_teams) - 1) // 2
    done = 0
    bar = st.progress(0.0, text=f"Pre-computing win probabilities (0/{total})...")

    for i in range(len(all_teams)):
        for j in range(i + 1, len(all_teams)):
            ta, sa = all_teams[i]
            tb, sb = all_teams[j]
            key = (ta, tb)
            if key not in _wp_cache:
                # Use better seed as team_a; round_num=3 (S16) is a neutral mid-tournament proxy
                if sa <= sb:
                    _get_win_prob(ta, sa, tb, sb, round_num=3)
                else:
                    wp_b = _get_win_prob(tb, sb, ta, sa, round_num=3)
                    _wp_cache[(ta, tb)] = 1.0 - wp_b
                    _wp_cache[(tb, ta)] = wp_b
            done += 1
            if done % 50 == 0 or done == total:
                bar.progress(done / total, text=f"Pre-computing win probabilities ({done}/{total})...")

    bar.empty()


def simulate_region(seed_team: dict, deterministic: bool = True,
                    use_cache: bool = False) -> list:
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

            win_prob_fav = _get_win_prob(fav, fs, dog, ds, round_num,
                                          use_cache=use_cache)

            if deterministic:
                winner = (fav, fs) if win_prob_fav >= 0.5 else (dog, ds)
            else:
                winner = (fav, fs) if np.random.random() < win_prob_fav else (dog, ds)
            next_round.append(winner)
        current_round = next_round
        all_rounds.append(current_round[:])
        round_num += 1

    return all_rounds


def _project_game_row(ta, sa, tb, sb, round_num, label, round_name):
    """
    Return a display row + winner for a game.
    team_a = lower seed (better seed by number). Model convention: positive spread = team_a favored.
    We display from MODEL's perspective: show who the model actually likes.
    """
    proj = project_game(ta, tb, round_num=round_num, year=current_year, seed_a=sa, seed_b=sb)
    if "error" in proj:
        wp_a, sp = 0.5, 0.0
    else:
        wp_a = proj.get("win_prob_a", 0.5)
        sp = proj.get("projected_spread", 0.0)  # model convention: positive = team_a wins

    # Determine model pick (team with higher win probability)
    if wp_a >= 0.5:
        model_pick, pick_seed, opp, opp_seed = ta, sa, tb, sb
        pick_wp = wp_a
        # Spread from pick's perspective in betting convention (negative = pick is favored)
        spread_display = f"{model_pick} {-sp:+.1f}"
    else:
        model_pick, pick_seed, opp, opp_seed = tb, sb, ta, sa
        pick_wp = 1 - wp_a
        # sp is team_a's spread; team_b's spread is -sp; betting convention: negate again
        spread_display = f"{model_pick} {sp:+.1f}"

    return {
        "Round": round_name,
        "Matchup": label,
        "Model Pick": f"#{pick_seed} {model_pick}",
        "Opponent": f"#{opp_seed} {opp}",
        "Pick Win Prob": f"{pick_wp:.1%}",
        "Model Spread": spread_display,
    }, wp_a


def simulate_final_four(region_winners: dict, deterministic: bool = True,
                        use_cache: bool = False):
    """East vs West, South vs Midwest in F4."""
    matchups = [
        (region_winners["East"], region_winners["West"], "East vs West"),
        (region_winners["South"], region_winners["Midwest"], "South vs Midwest"),
    ]
    f4_winners = []
    games = []
    for (ta, sa), (tb, sb), label in matchups:
        # Always put lower seed as team_a for canonical project_game call
        if sa <= sb:
            t_a, s_a, t_b, s_b = ta, sa, tb, sb
        else:
            t_a, s_a, t_b, s_b = tb, sb, ta, sa

        if use_cache:
            # Fast path: use cached win prob, skip display row generation
            fav_t, fav_s = (t_a, s_a) if s_a <= s_b else (t_b, s_b)
            dog_t, dog_s = (t_b, s_b) if s_a <= s_b else (t_a, s_a)
            wp_a = _get_win_prob(fav_t, fav_s, dog_t, dog_s, 5, use_cache=True)
            winner = (fav_t, fav_s) if np.random.random() < wp_a else (dog_t, dog_s)
            f4_winners.append(winner)
            continue

        row, wp_a = _project_game_row(t_a, s_a, t_b, s_b, 5, label, "Final Four")

        if deterministic:
            winner = (t_a, s_a) if wp_a >= 0.5 else (t_b, s_b)
        else:
            winner = (t_a, s_a) if np.random.random() < wp_a else (t_b, s_b)

        row["Winner"] = f"#{winner[1]} {winner[0]}"
        f4_winners.append(winner)
        games.append(row)

    # Championship
    (ca, csa), (cb, csb) = f4_winners[0], f4_winners[1]
    if csa <= csb:
        t_a, s_a, t_b, s_b = ca, csa, cb, csb
    else:
        t_a, s_a, t_b, s_b = cb, csb, ca, csa

    if use_cache:
        fav_t, fav_s = (t_a, s_a) if s_a <= s_b else (t_b, s_b)
        dog_t, dog_s = (t_b, s_b) if s_a <= s_b else (t_a, s_a)
        wp_a = _get_win_prob(fav_t, fav_s, dog_t, dog_s, 6, use_cache=True)
        champ = (fav_t, fav_s) if np.random.random() < wp_a else (dog_t, dog_s)
        return games, champ

    row, wp_a = _project_game_row(t_a, s_a, t_b, s_b, 6, "Championship", "Championship")

    if deterministic:
        champ = (t_a, s_a) if wp_a >= 0.5 else (t_b, s_b)
    else:
        champ = (t_a, s_a) if np.random.random() < wp_a else (t_b, s_b)

    row["Winner"] = f"#{champ[1]} {champ[0]}"
    games.append(row)
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

    # ── Step 1: pre-compute all pairwise win probs (single loading phase) ──────
    _wp_cache.clear()
    _precompute_win_probs(region_seed_team)

    # ── Step 2: deterministic bracket (instant from cache) ────────────────────
    region_all_rounds: dict = {}
    region_winners_det: dict = {}
    for region in REGIONS:
        region_all_rounds[region] = simulate_region(
            region_seed_team[region], deterministic=True, use_cache=True
        )
        region_winners_det[region] = region_all_rounds[region][-1][0]

    # ── Step 3: Monte Carlo simulations (instant from cache) ──────────────────
    champion_counts: dict = defaultdict(int)
    round_adv: dict = defaultdict(lambda: defaultdict(int))
    for sim in range(n_sims):
        sim_winners = {}
        for region in REGIONS:
            sim_rounds = simulate_region(
                region_seed_team[region], deterministic=False, use_cache=True
            )
            sim_winners[region] = sim_rounds[-1][0]
            for ri, survivors in enumerate(sim_rounds[1:], 2):
                for team, seed in survivors:
                    round_adv[team][ri] += 1
        _, sim_champ = simulate_final_four(
            sim_winners, deterministic=False, use_cache=True
        )
        champion_counts[sim_champ[0]] += 1

    # ── Display: bracket tabs ──────────────────────────────────────────────────
    st.subheader("Projected Bracket")
    result_tabs = st.tabs(REGIONS + ["🏆 Final Four"])
    round_labels = {1: "R64 → R32", 2: "R32 → S16", 3: "S16 → E8", 4: "Elite Eight"}

    for region, rtab in zip(REGIONS, result_tabs[:4]):
        with rtab:
            all_rounds = region_all_rounds[region]
            for ri in range(1, len(all_rounds)):
                label = round_labels.get(ri, f"Round {ri}")
                st.markdown(f"#### {label}")
                survivors = all_rounds[ri]
                prev = all_rounds[ri - 1]
                prev_per_winner = len(prev) // len(survivors)
                cols = st.columns(len(survivors))
                for ci, (team, seed) in enumerate(survivors):
                    with cols[ci]:
                        start = ci * prev_per_winner
                        prev_slot = prev[start: start + prev_per_winner]
                        opp = next((t for t, s in prev_slot if t != team), "?")
                        opp_seed = next((s for t, s in prev_slot if t != team), "?")
                        st.success(f"**#{seed} {team}**\ndef. #{opp_seed} {opp}")
                st.markdown("")

            winner, wseed = region_winners_det[region]
            st.markdown(f"### 🏆 {region} Winner: **#{wseed} {winner}**")

    with result_tabs[4]:
        st.subheader("Final Four & Championship")
        # Final Four display calls project_game directly for spreads/win probs
        f4_games, champion = simulate_final_four(region_winners_det, deterministic=True)
        f4_df = pd.DataFrame(f4_games)[["Round", "Matchup", "Model Pick", "Opponent", "Pick Win Prob", "Model Spread", "Winner"]]
        st.dataframe(f4_df, use_container_width=True, hide_index=True)
        st.caption("**Model Pick** = team the model projects to win. **Pick Win Prob** = that team's win probability. **Model Spread** = projected margin in betting convention (negative = pick is favored).")
        st.markdown(f"## 🏆 National Champion: **{champion[0]}** (#{champion[1]} seed)")

    st.divider()

    # ── Display: Monte Carlo results ───────────────────────────────────────────
    st.subheader(f"Monte Carlo Probabilities ({n_sims:,} simulations)")

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
