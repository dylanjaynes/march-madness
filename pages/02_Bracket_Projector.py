import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict

from src.utils.config import (
    TOURNAMENT_YEARS, MISMATCH_SEED_DIFF_THRESHOLD, MISMATCH_BARTHAG_THRESHOLD,
)
from src.model.predict import project_game, season_label, data_as_of, spread_to_win_prob
from src.model.train import load_model
from src.features.matchup import build_matchup_features, MATCHUP_FEATURES
from src.ingest.bracket import (
    fetch_and_store_bracket, fetch_and_store_projected_bracket_espn,
    load_bracket_from_db, get_bracket_status,
)
from src.features.team_ratings import load_ratings_cache, clear_ratings_cache

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
# 2026 official NCAA tournament bracket (Selection Sunday, March 15 2026)
# First Four play-in winners: South #16 (Prairie View A&M/Lehigh), West #11 (Texas/NC State),
#   Midwest #11 (Miami OH/SMU), Midwest #16 (UMBC/Howard)
# ── 2026 Official NCAA Tournament Bracket (Selection Sunday, March 15 2026) ──
# First Four play-in results:
#   South  #16: Lehigh vs Prairie View A&M  → placeholder: "Lehigh"
#   West   #11: Texas vs NC State           → placeholder: "North Carolina State"
#   Midwest #11: SMU vs Miami (OH)          → placeholder: "Miami OH"
#   Midwest #16: Howard vs UMBC             → placeholder: "Howard"
_OFFICIAL_2026 = {
    "East": {
        1: "Duke",          16: "Siena",
        8: "Ohio State",     9: "TCU",
        5: "St. John's (NY)", 12: "Northern Iowa",
        4: "Kansas",        13: "Cal Baptist",
        6: "Louisville",    11: "South Florida",
        3: "Michigan State", 14: "North Dakota State",
        7: "UCLA",          10: "UCF",
        2: "Connecticut",   15: "Furman",
    },
    "West": {
        1: "Arizona",       16: "LIU",
        8: "Villanova",      9: "Utah State",
        5: "Wisconsin",     12: "High Point",
        4: "Arkansas",      13: "Hawaii",
        6: "BYU",           11: "North Carolina State",
        3: "Gonzaga",       14: "Kennesaw State",
        7: "Miami FL",      10: "Missouri",
        2: "Purdue",        15: "Queens",
    },
    "South": {
        1: "Florida",       16: "Lehigh",
        8: "Clemson",        9: "Iowa",
        5: "Vanderbilt",    12: "McNeese State",
        4: "Nebraska",      13: "Troy",
        6: "North Carolina", 11: "VCU",
        3: "Illinois",      14: "Penn",
        7: "Saint Mary's",  10: "Texas A&M",
        2: "Houston",       15: "Idaho",
    },
    "Midwest": {
        1: "Michigan",      16: "Howard",
        8: "Georgia",        9: "Saint Louis",
        5: "Texas Tech",    12: "Akron",
        4: "Alabama",       13: "Hofstra",
        6: "Tennessee",     11: "Miami OH",
        3: "Virginia",      14: "Wright State",
        7: "Kentucky",      10: "Santa Clara",
        2: "Iowa State",    15: "Tennessee St.",
    },
}
# Keep alias for any legacy references
_PROJECTED_TEAMS = _OFFICIAL_2026

# Always start from the hardcoded 2026 official bracket.
# The DB bracket can still be fetched/saved via the UI buttons below,
# but the hardcoded teams are the authoritative source on every load.
DEFAULT_TEAMS = _OFFICIAL_2026
_bracket_source = "official"

# Still initialise DB so fetch/save buttons work
try:
    from src.utils.db import init_db
    init_db()
    _bracket_status = get_bracket_status(current_year)
    if not _bracket_status["stored"]:
        _bracket_status = {"stored": True, "n_teams": 64, "fetched_at": "2026-03-15"}
except Exception:
    _bracket_status = {"stored": True, "n_teams": 64, "fetched_at": "2026-03-15"}

# ── Bracket fetch UI ──────────────────────────────────────────────────────────
_col_status, _col_official, _col_save, _col_proj = st.columns([3, 1, 1, 1])
with _col_status:
    st.success(
        "✅ **2026 Official NCAA Bracket** · 64 teams · Selection Sunday March 15 2026",
        icon=None,
    )

with _col_official:
    if st.button("🔄 Fetch SR", use_container_width=True, type="primary",
                 help="Fetches the official 2026 NCAA bracket from Sports Reference. "
                      "SR usually publishes within a few hours of Selection Sunday."):
        with st.spinner("Fetching bracket from Sports Reference..."):
            try:
                bracket = fetch_and_store_bracket(current_year)
                if bracket:
                    n = sum(len(v) for v in bracket.values())
                    st.success(f"✅ Loaded {n} teams across {len(bracket)} regions!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(
                        "Sports Reference hasn't published the 2026 bracket yet — "
                        "use **💾 Save to DB** to store the known bracket now.",
                        icon="⚠️",
                    )
            except Exception as e:
                st.error(f"Fetch failed: {e}")

with _col_save:
    if st.button("💾 Save to DB", use_container_width=True,
                 help="Save the current hardcoded 2026 bracket to the database."):
        try:
            from src.ingest.bracket import store_bracket
            n = store_bracket(current_year, _PROJECTED_TEAMS)
            st.success(f"✅ Saved {n} teams!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")

with _col_proj:
    if st.button("📡 Load Model Picks", use_container_width=True,
                 help="Load pre-Selection Sunday bracket projections from CBS Sports (bracketology). "
                      "Now that the official bracket is released, use '🔄 Fetch SR' instead."):
        with st.spinner("Fetching CBS Sports Bracketology..."):
            try:
                bracket = fetch_and_store_projected_bracket_espn(current_year)
                if bracket:
                    n = sum(len(v) for v in bracket.values())
                    st.success(f"✅ Loaded {n} teams from CBS Sports Bracketology!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(
                        "⚠️ CBS Sports bracketology is no longer available — "
                        "their page now shows the official bracket (JS-rendered, not scrapable). "
                        "Use **🔄 Official Bracket** instead.",
                    )
            except Exception as e:
                st.error(f"CBS fetch failed: {e}")


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
col_sim1, col_sim2, col_sim3 = st.columns([1, 2, 1])
with col_sim1:
    n_sims = st.selectbox("Simulations", [1000, 5000, 10000], index=0)
with col_sim2:
    chaos = st.slider("🌪️ Chaos Meter", 0, 100, 0,
                      help="0 = pure model predictions · 100 = pure coin flip")
with col_sim3:
    st.metric("Chaos", f"{chaos}%", delta=None)

run_btn = st.button("🏀 Simulate Tournament", type="primary", use_container_width=True)


# ── Bracket HTML visualization ─────────────────────────────────────────────────
_SLOT_H     = 28   # height of one team row inside a game card
_SLOT_W     = 148  # width of a game card
_GAME_GAP   = 1    # height of the divider line between team rows
_CONN_W     = 14   # connector SVG width
_R64_GAME_H = 68   # height allocated per R64 game row
_STROKE     = "#c6c8cb"
_STROKE_W   = "1.5"


def _b_slot(team: str, seed: int, prob: float, is_winner: bool,
             spread: float = None, total: float = None) -> str:
    """One team row inside a game card (light theme)."""
    tc = "#1a1a1a" if is_winner else "#8a8c8d"
    fw = "600"    if is_winner else "400"
    pc = "#06c"   if is_winner else "#b0b2b4"
    nm = (team[:14] + "…") if len(team) > 15 else team

    seed_span = (
        f'<span style="font-size:9px;color:#555;background:#f2f4f6;border-radius:3px;'
        f'padding:1px 3px;min-width:14px;text-align:center;flex-shrink:0;margin-right:4px;">'
        f'{seed}</span>'
    )
    name_span = (
        f'<span style="flex:1;font-size:10px;color:{tc};font-weight:{fw};'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{team}">{nm}</span>'
    )
    if spread is not None:
        sp_rounded = round(spread * 2) / 2
        sp_sign    = "−" if sp_rounded > 0 else "+"
        sp_str     = f"{sp_sign}{abs(sp_rounded):.1f}"
        tooltip    = f"Spread: {sp_str}" + (f" · O/U {total:.1f}" if total is not None else "")
        spread_span = (
            f'<span style="font-size:8px;color:#888;margin-left:3px;white-space:nowrap;"'
            f' title="{tooltip}">{sp_str}</span>'
        )
    else:
        spread_span = ""
    prob_span = (
        f'<span style="font-size:10px;color:{pc};margin-left:auto;padding-left:4px;'
        f'white-space:nowrap;">{prob:.0f}%</span>'
    )
    return (
        f'<div style="display:flex;align-items:center;height:{_SLOT_H}px;padding:0 6px;">'
        + seed_span + name_span + spread_span + prob_span +
        f'</div>'
    )


def _b_game_card(game: dict, width: int = None) -> str:
    """Render one matchup as a styled card (two team rows + divider)."""
    w = width if width is not None else _SLOT_W
    a, b, winner = game['a'], game['b'], game.get('winner', 'a')
    slot_a = _b_slot(a['team'], a['seed'], a['prob'], winner == 'a',
                     spread=a.get('spread'), total=a.get('total'))
    slot_b = _b_slot(b['team'], b['seed'], b['prob'], winner == 'b',
                     spread=b.get('spread'), total=None)
    return (
        f'<div style="width:{w}px;border:1px solid #d8dade;border-radius:6px;'
        f'background:linear-gradient(180deg,#fff 0%,#fcfcfd 100%);'
        f'box-shadow:0 1px 2px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.08);overflow:hidden;">'
        + slot_a
        + f'<div style="height:1px;background:#eceef1;"></div>'
        + slot_b
        + f'</div>'
    )


def _b_conn_ltr(n_left: int, left_game_h: float) -> str:
    """LTR connector SVG: n_left games (left) → n_left/2 games (right)."""
    total_h = n_left * left_game_h
    lines, xm = [], _CONN_W // 2
    for i in range(n_left // 2):
        y1 = (2 * i)     * left_game_h + left_game_h / 2
        y2 = (2 * i + 1) * left_game_h + left_game_h / 2
        ym = (y1 + y2) / 2
        lines += [
            f'<line x1="0" y1="{y1:.1f}" x2="{xm}" y2="{y1:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
            f'<line x1="0" y1="{y2:.1f}" x2="{xm}" y2="{y2:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
            f'<line x1="{xm}" y1="{y1:.1f}" x2="{xm}" y2="{y2:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
            f'<line x1="{xm}" y1="{ym:.1f}" x2="{_CONN_W}" y2="{ym:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
        ]
    return (f'<svg width="{_CONN_W}" height="{total_h:.0f}" style="flex-shrink:0;">'
            + "".join(lines) + '</svg>')


def _b_conn_rtl(n_right: int, right_game_h: float) -> str:
    """RTL connector SVG: n_right/2 games (left) → n_right games (right)."""
    left_game_h = right_game_h * 2
    n_left      = n_right // 2
    total_h     = n_left * left_game_h
    lines, xm   = [], _CONN_W // 2
    for i in range(n_left):
        y_l = i * left_game_h + left_game_h / 2
        yr1 = (2 * i)     * right_game_h + right_game_h / 2
        yr2 = (2 * i + 1) * right_game_h + right_game_h / 2
        lines += [
            f'<line x1="0" y1="{y_l:.1f}" x2="{xm}" y2="{y_l:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
            f'<line x1="{xm}" y1="{yr1:.1f}" x2="{_CONN_W}" y2="{yr1:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
            f'<line x1="{xm}" y1="{yr2:.1f}" x2="{_CONN_W}" y2="{yr2:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
            f'<line x1="{xm}" y1="{yr1:.1f}" x2="{xm}" y2="{yr2:.1f}" stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>',
        ]
    return (f'<svg width="{_CONN_W}" height="{total_h:.0f}" style="flex-shrink:0;">'
            + "".join(lines) + '</svg>')


def _b_r32s16_col(r32_games: list, s16_games: list, side: str) -> str:
    """Combined R32+S16 column: S16 card floats between the two R32 cards it feeds from,
    indented toward E8 (which is always the innermost column of the region)."""
    half_h  = 4 * _R64_GAME_H        # height of one half-group (covers 4 R64 rows)
    total_h = 8 * _R64_GAME_H        # full column height
    game_h  = 2 * _SLOT_H + _GAME_GAP
    indent  = 8                      # S16 indent pixels toward E8
    s16_w   = _SLOT_W - indent

    html = f'<div style="position:relative;width:{_SLOT_W}px;height:{total_h}px;flex-shrink:0;">'
    for grp in range(2):
        yo = grp * half_h

        # R32-top: vertically centered in the top half of the group
        r32_top_y = yo + (half_h / 2 - game_h) / 2
        html += (
            f'<div style="position:absolute;top:{r32_top_y:.1f}px;left:0;right:0;">'
            + _b_game_card(r32_games[grp * 2], _SLOT_W)
            + '</div>'
        )

        # S16: centered across the full group, indented toward E8
        s16_y = yo + (half_h - game_h) / 2
        s16_x = indent if side == 'left' else 0
        html += (
            f'<div style="position:absolute;top:{s16_y:.1f}px;left:{s16_x}px;">'
            + _b_game_card(s16_games[grp], s16_w)
            + '</div>'
        )

        # R32-bot: vertically centered in the bottom half of the group
        r32_bot_y = yo + half_h / 2 + (half_h / 2 - game_h) / 2
        html += (
            f'<div style="position:absolute;top:{r32_bot_y:.1f}px;left:0;right:0;">'
            + _b_game_card(r32_games[grp * 2 + 1], _SLOT_W)
            + '</div>'
        )
    html += '</div>'
    return html


def _b_region(rounds_data: list, name: str, flip: bool = False) -> str:
    """Render one bracket region (light theme, R32+S16 combined column).
    flip=False → left side  (R64 outermost, E8 innermost)
    flip=True  → right side (E8 outermost, R64 innermost)
    """
    r64 = rounds_data[0]   # 8 games
    r32 = rounds_data[1]   # 4 games
    s16 = rounds_data[2]   # 2 games
    e8  = rounds_data[3]   # 1 game

    total_h = 8 * _R64_GAME_H
    side    = 'right' if flip else 'left'

    # Region label above the grid
    label_align = 'right' if flip else 'left'
    label_html  = (
        f'<div style="font-size:10px;font-weight:800;color:#6c6e6f;text-transform:uppercase;'
        f'letter-spacing:.08em;text-align:{label_align};padding:0 4px 6px;">{name}</div>'
    )

    # R64 column — 8 game cards, each centered inside a _R64_GAME_H slot
    r64_col = f'<div style="display:flex;flex-direction:column;width:{_SLOT_W}px;flex-shrink:0;">'
    for g in r64:
        r64_col += (
            f'<div style="height:{_R64_GAME_H}px;display:flex;align-items:center;">'
            + _b_game_card(g, _SLOT_W) + '</div>'
        )
    r64_col += '</div>'

    # R32+S16 combined column
    r32s16_col = _b_r32s16_col(r32, s16, side)

    # E8 column — 1 game card vertically centered across full height
    e8_col = (
        f'<div style="display:flex;align-items:center;width:{_SLOT_W}px;'
        f'height:{total_h}px;flex-shrink:0;">'
        + _b_game_card(e8[0], _SLOT_W) + '</div>'
    )

    # Connectors (all 544 px tall to match region columns)
    # LTR: used for left-side region (R64→R32 and S16→E8)
    conn_r64_r32 = _b_conn_ltr(8, _R64_GAME_H)           # 8 → 4
    conn_s16_e8  = _b_conn_ltr(2, 4 * _R64_GAME_H)       # 2 → 1
    # RTL: used for right-side region (E8→S16 and R32→R64)
    conn_e8_s16  = _b_conn_rtl(2, 4 * _R64_GAME_H)       # 1 → 2
    conn_r32_r64 = _b_conn_rtl(8, _R64_GAME_H)           # 4 → 8

    if not flip:
        inner = r64_col + conn_r64_r32 + r32s16_col + conn_s16_e8 + e8_col
    else:
        inner = e8_col + conn_e8_s16 + r32s16_col + conn_r32_r64 + r64_col

    return (
        f'<div>'
        + label_html
        + f'<div style="display:flex;align-items:flex-start;">{inner}</div>'
        + '</div>'
    )


def _build_bracket_vis(region_all_rounds: dict) -> dict:
    """Convert simulation results to bracket_vis structure for HTML rendering."""
    vis = {}
    for region, all_rounds in region_all_rounds.items():
        rounds_data = []
        for ri in range(min(4, len(all_rounds) - 1)):
            teams = all_rounds[ri]
            games = []
            next_teams = [t for t, s in all_rounds[ri + 1]] if ri + 1 < len(all_rounds) else []
            for gi in range(len(teams) // 2):
                ta, sa = teams[gi * 2]
                tb, sb = teams[gi * 2 + 1]
                wp_a = _wp_cache.get((ta, tb), 1.0 - _wp_cache.get((tb, ta), 0.5))
                winner = 'a' if ta in next_teams else 'b'

                # Spread: from each team's perspective (cache keyed by better-seed-first)
                spread_a = _spread_cache.get((ta, tb))
                if spread_a is None and (tb, ta) in _spread_cache:
                    spread_a = -_spread_cache[(tb, ta)]
                spread_b = -spread_a if spread_a is not None else None

                # Total: game-level, symmetric; show only on team_a's slot
                total_ab = _total_cache.get((ta, tb), _total_cache.get((tb, ta)))

                games.append({
                    'a': {'team': ta, 'seed': sa, 'prob': wp_a * 100,
                          'spread': spread_a, 'total': total_ab},
                    'b': {'team': tb, 'seed': sb, 'prob': (1 - wp_a) * 100,
                          'spread': spread_b, 'total': None},
                    'winner': winner,
                })
            rounds_data.append(games)
        vis[region] = rounds_data
    return vis


def _build_full_bracket_html(bracket_vis: dict, f4_games: list, champion: tuple,
                              champion_counts: dict, n_sims: int) -> str:
    """Build the complete bracket HTML — leerob-inspired light theme.

    Layout (vertical stack):
        Top pair:    East (left) + West (right)   — E8s face each other
        Center:      [FF East/West] ─ [Championship] ─ [FF South/Midwest]
        Bottom pair: South (left) + Midwest (right) — E8s face each other
    """
    east    = bracket_vis.get('East',    [[]] * 4)
    west    = bracket_vis.get('West',    [[]] * 4)
    south   = bracket_vis.get('South',   [[]] * 4)
    midwest = bracket_vis.get('Midwest', [[]] * 4)

    champ_name, champ_seed = champion
    champ_pct = champion_counts.get(champ_name, 0) / n_sims * 100 if n_sims else 0

    # ── Center cluster: FF-left | h-conn | Championship | h-conn | FF-right ──
    ff_w    = 155   # FF game card width
    champ_w = 170   # Championship card width
    h_cw    = 28    # horizontal connector width
    game_h  = 2 * _SLOT_H + _GAME_GAP

    # Labels row (fixed-width cells align labels above each card)
    labels_row = (
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<div style="width:{ff_w}px;text-align:center;font-size:10px;font-weight:800;'
        f'color:#121213;text-transform:uppercase;letter-spacing:.06em;">Final Four</div>'
        f'<div style="width:{h_cw}px;"></div>'
        f'<div style="width:{champ_w}px;text-align:center;font-size:10px;font-weight:800;'
        f'color:#121213;text-transform:uppercase;letter-spacing:.06em;">Championship</div>'
        f'<div style="width:{h_cw}px;"></div>'
        f'<div style="width:{ff_w}px;text-align:center;font-size:10px;font-weight:800;'
        f'color:#121213;text-transform:uppercase;letter-spacing:.06em;">Final Four</div>'
        f'</div>'
    )

    # Horizontal connector — a short SVG line at mid-height of the card
    h_conn = (
        f'<svg width="{h_cw}" height="{game_h}" style="flex-shrink:0;" '
        f'viewBox="0 0 {h_cw} {game_h}" preserveAspectRatio="none" aria-hidden>'
        f'<line x1="0" y1="{game_h / 2:.1f}" x2="{h_cw}" y2="{game_h / 2:.1f}" '
        f'stroke="{_STROKE}" stroke-width="{_STROKE_W}"/>'
        f'</svg>'
    )

    # FF cards
    ff_left_card  = _b_game_card(f4_games[0], ff_w) if len(f4_games) >= 1 else (
        f'<div style="width:{ff_w}px;height:{game_h}px;"></div>')
    ff_right_card = _b_game_card(f4_games[1], ff_w) if len(f4_games) >= 2 else (
        f'<div style="width:{ff_w}px;height:{game_h}px;"></div>')

    # Championship card (or fallback placeholder)
    if len(f4_games) >= 3:
        champ_card = _b_game_card(f4_games[2], champ_w)
    else:
        champ_card = (
            f'<div style="width:{champ_w}px;border:1px solid #d8dade;border-radius:6px;'
            f'background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:10px;text-align:center;">'
            f'<div style="font-size:13px;font-weight:700;">🏆 {champ_name}</div>'
            f'<div style="font-size:10px;color:#555;">#{champ_seed} seed · {champ_pct:.1f}%</div>'
            f'</div>'
        )

    cards_row = (
        f'<div style="display:flex;align-items:center;">'
        + ff_left_card + h_conn + champ_card + h_conn + ff_right_card +
        f'</div>'
    )

    # Champion callout below championship card
    champ_callout = (
        f'<div style="text-align:center;margin-top:6px;">'
        f'<span style="font-size:12px;font-weight:700;color:#1a1a1a;">🏆 {champ_name}</span>'
        f'<span style="font-size:10px;color:#06c;padding-left:6px;">'
        f'#{champ_seed} · {champ_pct:.1f}% to win</span>'
        f'</div>'
    )

    center_cluster = (
        f'<div style="display:flex;flex-direction:column;align-items:center;padding:4px 0;">'
        + labels_row + cards_row + champ_callout +
        f'</div>'
    )

    # ── Region pairs ──
    top_pair = (
        f'<div style="display:flex;gap:24px;">'
        + _b_region(east,    'East',    flip=False)
        + _b_region(west,    'West',    flip=True)
        + '</div>'
    )
    bottom_pair = (
        f'<div style="display:flex;gap:24px;">'
        + _b_region(south,   'South',   flip=False)
        + _b_region(midwest, 'Midwest', flip=True)
        + '</div>'
    )

    body = (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:20px;">'
        + top_pair + center_cluster + bottom_pair +
        f'</div>'
    )

    return (
        '<!DOCTYPE html><html><body style="margin:0;padding:12px;'
        'background:#ececec;font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        'display:inline-block;">'
        + body +
        '</body></html>'
    )


# ── Simulation helpers ─────────────────────────────────────────────────────────
# Win probability cache: (ta, tb) -> float.  Populated once before any simulation.
_wp_cache: dict = {}
# Spread cache: (ta, tb) -> float, where ta = better seed (lower number).
# Positive spread = ta is favored by that many points.
_spread_cache: dict = {}
# Total cache: (ta, tb) -> float (game total, symmetric).
_total_cache: dict = {}

def _get_win_prob(ta: str, sa: int, tb: str, sb: int, round_num: int,
                  use_cache: bool = False, chaos: int = 0) -> float:
    """
    Return win prob for ta (better seed = lower number).
    use_cache=True: fast path for Monte Carlo — reads from pre-computed cache.
    use_cache=False: calls model directly with correct round_num (used for display).
    chaos: 0 = pure model, 100 = coin flip. Interpolates between model and 0.5.
    """
    if use_cache:
        # Cache is keyed (ta, tb) regardless of round — uses round 3 as proxy
        key = (ta, tb)
        if key in _wp_cache:
            raw_wp = _wp_cache[key]
        elif (tb, ta) in _wp_cache:
            raw_wp = 1.0 - _wp_cache[(tb, ta)]
        else:
            raw_wp = 0.5
    else:
        proj = project_game(ta, tb, round_num=round_num, year=current_year, seed_a=sa, seed_b=sb)
        raw_wp = 0.5 if "error" in proj else proj.get("win_prob_a", 0.5)

    if chaos > 0:
        alpha = (chaos / 100.0) ** 2  # quadratic: gentler at low values
        raw_wp = alpha * 0.5 + (1 - alpha) * raw_wp
    return raw_wp


def _precompute_win_probs(seed_teams_by_region: dict) -> None:
    """
    Pre-compute all pairwise win probabilities for every team in the bracket.
    With 16 teams per region × 4 regions = 64 teams → at most ~2016 matchups.

    Vectorized: builds all feature vectors first, then does a single batch
    model.predict() call instead of loading models 2016 times.
    """
    all_teams = []  # list of (team, seed)
    for region, seed_team in seed_teams_by_region.items():
        for seed, team in seed_team.items():
            if team:
                all_teams.append((team, seed))

    # Batch-load all team ratings in one DB query
    all_team_names = [t for t, s in all_teams]
    clear_ratings_cache()
    load_ratings_cache(all_team_names, current_year)

    # Build the list of ordered pairs (better seed = team_a)
    pairs = []  # [(ta, sa, tb, sb), ...]
    for i in range(len(all_teams)):
        for j in range(i + 1, len(all_teams)):
            ta, sa = all_teams[i]
            tb, sb = all_teams[j]
            if sa <= sb:
                pairs.append((ta, sa, tb, sb))
            else:
                pairs.append((tb, sb, ta, sa))

    total = len(pairs)
    bar = st.progress(0.0, text="Pre-computing win probabilities — building features...")

    # Build all feature vectors (fast: uses ratings cache, no model loading)
    feat_rows = []
    valid_pairs = []
    for idx, (ta, sa, tb, sb) in enumerate(pairs):
        feats = build_matchup_features(ta, tb, current_year, round_num=3,
                                       seed_a=sa, seed_b=sb)
        if not feats.isna().all():
            feat_rows.append(feats[MATCHUP_FEATURES].values)
            valid_pairs.append((ta, sa, tb, sb))
        else:
            # No ratings — default to 50/50
            _wp_cache[(ta, tb)] = 0.5
            _wp_cache[(tb, ta)] = 0.5
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            bar.progress((idx + 1) / total / 2,  # first half = feature building
                         text=f"Building features ({idx + 1}/{total})...")

    if feat_rows:
        bar.progress(0.5, text="Running model predictions...")
        # Single batch predict for all matchups using hybrid model routing.
        try:
            X = np.array(feat_rows)

            # Build mismatch mask from seed_diff (and barthag_diff if available)
            seed_diffs = np.array([abs(sa - sb) for (ta, sa, tb, sb) in valid_pairs])
            is_mismatch = seed_diffs >= MISMATCH_SEED_DIFF_THRESHOLD
            if "barthag_diff" in MATCHUP_FEATURES:
                bidx = list(MATCHUP_FEATURES).index("barthag_diff")
                is_mismatch = is_mismatch | (np.abs(X[:, bidx]) >= MISMATCH_BARTHAG_THRESHOLD)

            # Load hybrid models; fall back to legacy spread_model if missing
            try:
                comp_model = load_model("spread_competitive")
                mis_model = load_model("spread_mismatch")
                cal_comp = load_model("cal_competitive")
                cal_mis = load_model("cal_mismatch")
                spreads = np.zeros(len(X))
                if is_mismatch.any():
                    spreads[is_mismatch] = cal_mis.predict(
                        mis_model.predict(X[is_mismatch])
                    )
                if (~is_mismatch).any():
                    spreads[~is_mismatch] = cal_comp.predict(
                        comp_model.predict(X[~is_mismatch])
                    )
            except FileNotFoundError:
                spreads = load_model("spread_model").predict(X)

            # Also compute totals (pace-adjusted)
            try:
                total_model = load_model("total_model")
                from src.features.adjustments import apply_tournament_pace_adjustment
                raw_totals = total_model.predict(X)
                totals = np.array([apply_tournament_pace_adjustment(float(t)) for t in raw_totals])
            except Exception:
                totals = np.full(len(X), np.nan)

            for (ta, sa, tb, sb), spread, tot in zip(valid_pairs, spreads, totals):
                wp_a = spread_to_win_prob(float(spread))
                _wp_cache[(ta, tb)] = wp_a
                _wp_cache[(tb, ta)] = 1.0 - wp_a
                # Spread: ta is always the better seed (pairs built with sa<=sb)
                _spread_cache[(ta, tb)] = float(spread)   # positive = ta favored
                _spread_cache[(tb, ta)] = -float(spread)  # from tb's perspective
                if not np.isnan(tot):
                    _total_cache[(ta, tb)] = float(tot)
                    _total_cache[(tb, ta)] = float(tot)
        except Exception as e:
            # Fallback: compute individually if batch fails
            for ta, sa, tb, sb in valid_pairs:
                proj = project_game(ta, tb, round_num=3, year=current_year, seed_a=sa, seed_b=sb)
                if "error" not in proj:
                    wp_a = proj.get("win_prob_a", 0.5)
                    spread = proj.get("projected_spread", 0.0)
                    total  = proj.get("projected_total", None)
                else:
                    wp_a, spread, total = 0.5, 0.0, None
                _wp_cache[(ta, tb)] = wp_a
                _wp_cache[(tb, ta)] = 1.0 - wp_a
                _spread_cache[(ta, tb)] = spread
                _spread_cache[(tb, ta)] = -spread
                if total is not None:
                    _total_cache[(ta, tb)] = total
                    _total_cache[(tb, ta)] = total

    bar.progress(1.0, text=f"Done — {total} matchups pre-computed.")
    bar.empty()


def simulate_region(seed_team: dict, deterministic: bool = True,
                    use_cache: bool = False, chaos: int = 0) -> list:
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
                                          use_cache=use_cache, chaos=chaos)

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
                        use_cache: bool = False, chaos: int = 0):
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
            wp_a = _get_win_prob(fav_t, fav_s, dog_t, dog_s, 5, use_cache=True, chaos=chaos)
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
        wp_a = _get_win_prob(fav_t, fav_s, dog_t, dog_s, 6, use_cache=True, chaos=chaos)
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
    _spread_cache.clear()
    _total_cache.clear()
    _precompute_win_probs(region_seed_team)

    # ── Step 2: Monte Carlo simulations (instant from cache) ──────────────────
    champion_counts: dict = defaultdict(int)
    # Key: (team, region) to avoid double-counting when the same team name
    # appears in multiple regions (e.g. "Connecticut" in South AND Midwest).
    round_adv: dict = defaultdict(lambda: defaultdict(int))
    # Save last sim's bracket for the visual display (shows realistic upsets)
    region_all_rounds: dict = {}
    region_winners_det: dict = {}
    f4_display_games_last: list = []
    champ_candidate = None

    for sim in range(n_sims):
        sim_winners = {}
        sim_rounds_by_region = {}
        for region in REGIONS:
            sim_rounds = simulate_region(
                region_seed_team[region], deterministic=False, use_cache=True, chaos=chaos
            )
            sim_winners[region] = sim_rounds[-1][0]
            sim_rounds_by_region[region] = sim_rounds
            for ri, survivors in enumerate(sim_rounds[1:], 2):
                for team, seed in survivors:
                    round_adv[(team, region)][ri] += 1   # scoped to region
        _, sim_champ = simulate_final_four(
            sim_winners, deterministic=False, use_cache=True, chaos=chaos
        )
        champion_counts[sim_champ[0]] += 1
        # Use the last simulation run for the bracket display
        region_all_rounds = sim_rounds_by_region
        region_winners_det = sim_winners
        champ_candidate = sim_champ

    # ── Build bracket visualization ────────────────────────────────────────────
    bracket_vis = _build_bracket_vis(region_all_rounds)

    # Build F4 games data for center display using last sim's actual region winners
    f4_display_games = []
    for region_pair in [('East', 'West'), ('South', 'Midwest')]:
        ra, rb = region_pair
        ta, sa = region_winners_det[ra]
        tb, sb = region_winners_det[rb]
        if sa <= sb:
            fav, fs, dog, ds = ta, sa, tb, sb
        else:
            fav, fs, dog, ds = tb, sb, ta, sa
        wp = _wp_cache.get((fav, dog), _wp_cache.get((dog, fav), 0.5))
        if (fav, dog) not in _wp_cache and (dog, fav) in _wp_cache:
            wp = 1.0 - _wp_cache[(dog, fav)]
        if chaos > 0:
            alpha = (chaos / 100) ** 2  # quadratic: gentler at low values
            wp = alpha * 0.5 + (1 - alpha) * wp
        f4_winner = 'a' if np.random.random() < wp else 'b'

        # Spread/total for F4 center display
        sp_fav = _spread_cache.get((fav, dog))
        if sp_fav is None and (dog, fav) in _spread_cache:
            sp_fav = -_spread_cache[(dog, fav)]
        sp_dog = -sp_fav if sp_fav is not None else None
        tot_f4 = _total_cache.get((fav, dog), _total_cache.get((dog, fav)))

        f4_display_games.append({
            'a': {'team': fav, 'seed': fs, 'prob': wp * 100,
                  'spread': sp_fav, 'total': tot_f4},
            'b': {'team': dog, 'seed': ds, 'prob': (1-wp) * 100,
                  'spread': sp_dog, 'total': None},
            'winner': f4_winner,
        })

    # Build championship game from F4 winners
    if len(f4_display_games) >= 2:
        g0, g1 = f4_display_games[0], f4_display_games[1]
        c1 = g0['a'] if g0['winner'] == 'a' else g0['b']
        c2 = g1['a'] if g1['winner'] == 'a' else g1['b']
        # fav = lower seed number (better seed)
        if c1['seed'] <= c2['seed']:
            cf, cfs, cd, cds = c1['team'], c1['seed'], c2['team'], c2['seed']
        else:
            cf, cfs, cd, cds = c2['team'], c2['seed'], c1['team'], c1['seed']
        wp_champ = _wp_cache.get((cf, cd), 1.0 - _wp_cache.get((cd, cf), 0.5))
        sp_champ = _spread_cache.get((cf, cd))
        if sp_champ is None and (cd, cf) in _spread_cache:
            sp_champ = -_spread_cache[(cd, cf)]
        tot_champ = _total_cache.get((cf, cd), _total_cache.get((cd, cf)))
        champ_winner_side = 'a' if (champ_candidate and champ_candidate[0] == cf) else 'b'
        f4_display_games.append({
            'a': {'team': cf, 'seed': cfs, 'prob': wp_champ * 100,
                  'spread': sp_champ, 'total': tot_champ},
            'b': {'team': cd, 'seed': cds, 'prob': (1 - wp_champ) * 100,
                  'spread': -sp_champ if sp_champ is not None else None, 'total': None},
            'winner': champ_winner_side,
        })

    # Derive the displayed champion from the actual championship game winner —
    # NOT from champ_candidate (which is the last MC sim's champion and may
    # differ from the display game's random outcome).
    if len(f4_display_games) >= 3:
        _cg = f4_display_games[2]
        _cw = _cg['a'] if _cg['winner'] == 'a' else _cg['b']
        display_champion = (_cw['team'], _cw['seed'])
    else:
        display_champion = champ_candidate

    bracket_html = _build_full_bracket_html(
        bracket_vis, f4_display_games, display_champion,
        champion_counts, n_sims,
    )

    # ── Render visual bracket ─────────────────────────────────────────────────
    st.subheader("Simulated Bracket")
    st.caption("One probabilistic run — upsets happen per model win probabilities · % = win prob for that matchup · re-simulate for a different outcome · scroll right for full bracket")
    import streamlit.components.v1 as components
    components.html(bracket_html, height=1500, scrolling=True)

    st.divider()

    # ── Monte Carlo summary (collapsible) ─────────────────────────────────────
    with st.expander(f"📊 Monte Carlo Results ({n_sims:,} simulations)", expanded=False):
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
                coloraxis_showscale=False, height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Build team-region pairs; if a team name appears in >1 region,
        # label it as "Team (Region)" to distinguish the two bracket slots.
        team_region_pairs = [
            (team, region)
            for region in REGIONS
            for team in region_seed_team[region].values()
            if team
        ]
        # Remove exact duplicates (same team + same region listed twice)
        seen = set()
        team_region_pairs = [
            tr for tr in team_region_pairs
            if not (tr in seen or seen.add(tr))
        ]
        team_name_counts = defaultdict(int)
        for team, region in team_region_pairs:
            team_name_counts[team] += 1

        adv_rows = []
        for team, region in sorted(team_region_pairs):
            display_name = (
                f"{team} ({region[:2]})"
                if team_name_counts[team] > 1
                else team
            )
            key = (team, region)
            adv_rows.append({
                "Team":    display_name,
                "R32 %":   round(round_adv[key].get(2, 0) / n_sims * 100, 1),
                "S16 %":   round(round_adv[key].get(3, 0) / n_sims * 100, 1),
                "E8 %":    round(round_adv[key].get(4, 0) / n_sims * 100, 1),
                "F4 %":    round(round_adv[key].get(5, 0) / n_sims * 100, 1),
                "Champ %": round(champion_counts.get(team, 0) / n_sims * 100, 1),
            })
        if adv_rows:
            adv_df = pd.DataFrame(adv_rows).sort_values("Champ %", ascending=False)
            st.subheader("Round Advancement Probabilities")
            st.dataframe(adv_df, use_container_width=True, hide_index=True)

else:
    st.info("Select teams above and click **🏀 Simulate Tournament** to run projections.")
    st.caption(f"Data as of: {data_note}")
