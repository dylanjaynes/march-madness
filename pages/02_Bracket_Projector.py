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
_PROJECTED_TEAMS = {
    "East": {
        1: "Duke", 16: "Siena",
        8: "Ohio State", 9: "TCU",
        5: "St. John's (NY)", 12: "Northern Iowa",
        4: "Kansas", 13: "Cal Baptist",
        6: "Louisville", 11: "South Florida",
        3: "Michigan State", 14: "North Dakota State",
        7: "UCLA", 10: "UCF",
        2: "Connecticut", 15: "Furman",
    },
    "West": {
        1: "Arizona", 16: "LIU",
        8: "Villanova", 9: "Utah State",
        5: "Wisconsin", 12: "High Point",
        4: "Arkansas", 13: "Hawaii",
        6: "BYU", 11: "North Carolina State",
        3: "Gonzaga", 14: "Kennesaw State",
        7: "Miami FL", 10: "Missouri",
        2: "Purdue", 15: "Queens",
    },
    "South": {
        1: "Florida", 16: "Lehigh",
        8: "Clemson", 9: "Iowa",
        5: "Vanderbilt", 12: "McNeese State",
        4: "Nebraska", 13: "Troy",
        6: "North Carolina", 11: "VCU",
        3: "Illinois", 14: "Penn",
        7: "Saint Mary's", 10: "Texas A&M",
        2: "Houston", 15: "Idaho",
    },
    "Midwest": {
        1: "Michigan", 16: "Howard",
        8: "Georgia", 9: "Saint Louis",
        5: "Texas Tech", 12: "Akron",
        4: "Alabama", 13: "Hofstra",
        6: "Tennessee", 11: "Miami OH",
        3: "Virginia", 14: "Wright State",
        7: "Kentucky", 10: "Santa Clara",
        2: "Iowa State", 15: "Tennessee St.",
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
_col_status, _col_official, _col_save, _col_proj = st.columns([3, 1, 1, 1])
with _col_status:
    if _bracket_source == "official":
        st.success(
            f"✅ **Official bracket loaded** · {_bracket_status['n_teams']} teams · "
            f"fetched {_bracket_status['fetched_at'][:16] if _bracket_status['fetched_at'] else ''}",
            icon=None,
        )
    else:
        st.info(
            "📋 **2026 bracket pre-loaded** (Selection Sunday) — "
            "click **💾 Save to DB** to persist it, or **🔄 Fetch SR** to pull from Sports Reference.",
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
    if st.button("📡 Projected", use_container_width=True,
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
_SLOT_H = 36     # 24px main row + 12px stats row (spread / total)
_SLOT_W = 128    # slightly wider to fit spread + total text
_GAME_GAP = 3
_CONN_W = 14     # tighter connectors
_LABEL_H = 16
_R64_GAME_H = 82  # height per R64 game slot (accommodates 2-line slots)


def _b_slot(team: str, seed: int, prob: float, is_winner: bool,
             spread: float = None, total: float = None) -> str:
    bg   = "#1b2e40" if is_winner else "#0f1925"
    bord = "#2d5a87" if is_winner else "#1a2535"
    tc   = "#ddeeff" if is_winner else "#7799aa"
    pc   = "#5aadff" if is_winner else "#3a6a8f"
    sc   = "#6699bb"   # stats line color
    nm   = (team[:12] + "…") if len(team) > 13 else team

    # Stats second line (spread + total)
    if spread is not None and total is not None:
        sp_sign = "−" if spread < 0 else "+"
        sp_str  = f"{sp_sign}{abs(spread):.1f}"
        stats = (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:0 4px 1px;font-size:9px;color:{sc};line-height:12px;">'
            f'<span title="Projected spread">{sp_str}</span>'
            f'<span title="Projected total (O/U)">O/U {total:.0f}</span>'
            f'</div>'
        )
    elif spread is not None:
        sp_sign = "−" if spread < 0 else "+"
        sp_str  = f"{sp_sign}{abs(spread):.1f}"
        stats = f'<div style="padding:0 4px 1px;font-size:9px;color:{sc};line-height:12px;">{sp_str}</div>'
    elif total is not None:
        stats = f'<div style="padding:0 4px 1px;font-size:9px;color:{sc};line-height:12px;text-align:right;">O/U {total:.0f}</div>'
    else:
        stats = f'<div style="height:12px;"></div>'

    # prob = model win probability for THIS matchup (not cumulative advancement)
    return (
        f'<div style="background:{bg};border:1px solid {bord};border-radius:2px;">'
        f'<div style="display:flex;align-items:center;height:24px;padding:0 4px;">'
        f'<span style="font-size:10px;color:#445;min-width:13px;font-weight:700;">{seed}</span>'
        f'<span style="flex:1;font-size:10px;color:{tc};overflow:hidden;text-overflow:ellipsis;'
        f'white-space:nowrap;padding:0 3px;">{nm}</span>'
        f'<span style="font-size:10px;color:{pc};min-width:32px;text-align:right;" '
        f'title="Win probability for this matchup">{prob:.0f}%</span>'
        f'</div>'
        + stats +
        f'</div>'
    )


def _b_round_col(games: list, game_h: float, label: str) -> str:
    lh = f'<div style="height:{_LABEL_H}px;font-size:9px;color:#445;text-align:center;letter-spacing:.4px;text-transform:uppercase;display:flex;align-items:center;justify-content:center;">{label}</div>'
    col = f'<div style="display:flex;flex-direction:column;width:{_SLOT_W}px;">{lh}'
    for g in games:
        inner = 2 * _SLOT_H + _GAME_GAP
        pad   = max(0, (game_h - inner) / 2)
        col  += (f'<div style="height:{game_h:.1f}px;display:flex;flex-direction:column;justify-content:center;">'
                 f'<div style="padding:{pad:.1f}px 0;">'
                 + _b_slot(g['a']['team'], g['a']['seed'], g['a']['prob'], g.get('winner') == 'a',
                            spread=g['a'].get('spread'), total=g['a'].get('total'))
                 + f'<div style="height:{_GAME_GAP}px;"></div>'
                 + _b_slot(g['b']['team'], g['b']['seed'], g['b']['prob'], g.get('winner') == 'b',
                            spread=g['b'].get('spread'), total=None)
                 + '</div></div>')
    col += '</div>'
    return col


def _b_conn_ltr(n_left: int, left_game_h: float) -> str:
    """LTR connector: n_left games → n_left/2 games."""
    total_h = n_left * left_game_h + _LABEL_H
    lines = []
    xm = _CONN_W // 2
    for i in range(n_left // 2):
        y1 = _LABEL_H + (2*i)   * left_game_h + left_game_h / 2
        y2 = _LABEL_H + (2*i+1) * left_game_h + left_game_h / 2
        ym = (y1 + y2) / 2
        lines += [
            f'<line x1="0" y1="{y1:.1f}" x2="{xm}" y2="{y1:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
            f'<line x1="0" y1="{y2:.1f}" x2="{xm}" y2="{y2:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
            f'<line x1="{xm}" y1="{y1:.1f}" x2="{xm}" y2="{y2:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
            f'<line x1="{xm}" y1="{ym:.1f}" x2="{_CONN_W}" y2="{ym:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
        ]
    return f'<svg width="{_CONN_W}" height="{total_h:.0f}" style="flex-shrink:0;overflow:visible;">{"".join(lines)}</svg>'


def _b_conn_rtl(n_right: int, right_game_h: float) -> str:
    """RTL connector: n_right/2 games (left) → n_right games (right)."""
    left_game_h = right_game_h * 2
    n_left = n_right // 2
    total_h = n_left * left_game_h + _LABEL_H
    lines = []
    xm = _CONN_W // 2
    for i in range(n_left):
        y_l = _LABEL_H + i * left_game_h + left_game_h / 2
        yr1 = _LABEL_H + (2*i)   * right_game_h + right_game_h / 2
        yr2 = _LABEL_H + (2*i+1) * right_game_h + right_game_h / 2
        lines += [
            f'<line x1="0" y1="{y_l:.1f}" x2="{xm}" y2="{y_l:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
            f'<line x1="{xm}" y1="{yr1:.1f}" x2="{_CONN_W}" y2="{yr1:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
            f'<line x1="{xm}" y1="{yr2:.1f}" x2="{_CONN_W}" y2="{yr2:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
            f'<line x1="{xm}" y1="{yr1:.1f}" x2="{xm}" y2="{yr2:.1f}" stroke="#1e3a54" stroke-width="1.5"/>',
        ]
    return f'<svg width="{_CONN_W}" height="{total_h:.0f}" style="flex-shrink:0;overflow:visible;">{"".join(lines)}</svg>'


def _b_region(rounds_data: list, name: str, flip: bool = False) -> str:
    """
    4 rounds: [R64_games, R32_games, S16_games, E8_games]
    flip=True for right-side regions (display order reversed: E8→S16→R32→R64)
    """
    labels = ['R64', 'R32', 'S16', 'E8']
    items  = [(rounds_data[ri], _R64_GAME_H * (2**ri), labels[ri]) for ri in range(4)]
    if flip:
        items = list(reversed(items))

    rlabel = (f'<div style="font-size:10px;font-weight:700;color:#5588aa;'
              f'text-align:center;padding:3px 0;letter-spacing:1px;'
              f'text-transform:uppercase;">{name}</div>')
    row = f'<div>{rlabel}<div style="display:flex;align-items:flex-start;">'

    for idx, (gms, gh, lbl) in enumerate(items):
        row += _b_round_col(gms, gh, lbl)
        if idx < len(items) - 1:
            if not flip:
                row += _b_conn_ltr(len(gms), gh)
            else:
                next_n = len(items[idx+1][0])
                next_gh = items[idx+1][1]
                row += _b_conn_rtl(next_n, next_gh)

    row += '</div></div>'
    return row


def _b_center_slot(team: str, seed: int, prob: float, is_winner: bool, width: int = 150) -> str:
    bg   = "#1b2e40" if is_winner else "#0f1925"
    bord = "#2d5a87" if is_winner else "#1a2535"
    tc   = "#ddeeff" if is_winner else "#7799aa"
    pc   = "#5aadff"
    nm   = (team[:16] + "…") if len(team) > 17 else team
    return (
        f'<div style="display:flex;align-items:center;height:{_SLOT_H}px;width:{width}px;'
        f'padding:0 5px;background:{bg};border:1px solid {bord};border-radius:2px;">'
        f'<span style="font-size:10px;color:#445;min-width:14px;font-weight:700;">{seed}</span>'
        f'<span style="flex:1;font-size:11px;color:{tc};overflow:hidden;text-overflow:ellipsis;'
        f'white-space:nowrap;padding:0 4px;">{nm}</span>'
        f'<span style="font-size:10px;color:{pc};min-width:34px;text-align:right;">{prob:.1f}%</span>'
        f'</div>'
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
    """Build the complete bracket HTML string."""
    # F4 / Championship center column
    cw = 155  # center slot width
    f4_h = _R64_GAME_H * 16  # total height of one region pair (East+Midwest)

    def _f4_game_html(game_dict, region_label):
        a, b = game_dict['a'], game_dict['b']
        w = game_dict.get('winner', 'a')
        return (
            f'<div style="margin:4px 0;">'
            f'<div style="font-size:9px;color:#445;text-align:center;letter-spacing:.4px;'
            f'text-transform:uppercase;padding-bottom:2px;">{region_label}</div>'
            + _b_center_slot(a['team'], a['seed'], a['prob'], w == 'a', cw)
            + f'<div style="height:{_GAME_GAP}px;"></div>'
            + _b_center_slot(b['team'], b['seed'], b['prob'], w == 'b', cw)
            + '</div>'
        )

    champ_name, champ_seed = champion
    champ_pct = champion_counts.get(champ_name, 0) / n_sims * 100 if n_sims else 0

    center_html = (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'justify-content:center;width:{cw + 20}px;padding-top:{_LABEL_H}px;">'
    )
    if len(f4_games) >= 2:
        center_html += _f4_game_html(f4_games[0], 'Final Four')
    center_html += (
        f'<div style="margin:8px 0;text-align:center;">'
        f'<div style="font-size:9px;color:#445;text-transform:uppercase;letter-spacing:.4px;padding-bottom:4px;">Championship</div>'
        f'<div style="background:#1a3045;border:1px solid #2d5a87;border-radius:4px;padding:6px 10px;min-width:{cw}px;">'
        f'<div style="font-size:13px;font-weight:700;color:#ffd700;">🏆 {champ_name}</div>'
        f'<div style="font-size:10px;color:#5aadff;">#{champ_seed} seed · {champ_pct:.1f}%</div>'
        f'</div></div>'
    )
    if len(f4_games) >= 3:
        center_html += _f4_game_html(f4_games[2], 'Final Four')
    elif len(f4_games) >= 2:
        center_html += _f4_game_html(f4_games[1], 'Final Four')
    center_html += '</div>'

    east_mw  = bracket_vis.get('East',    [[]] * 4)
    west_mw  = bracket_vis.get('Midwest', [[]] * 4)
    south    = bracket_vis.get('South',   [[]] * 4)
    west     = bracket_vis.get('West',    [[]] * 4)

    left_html  = (f'<div>{_b_region(east_mw,  "East",    flip=False)}'
                  f'<div style="height:8px;"></div>'
                  f'{_b_region(west_mw,  "Midwest", flip=False)}</div>')
    right_html = (f'<div>{_b_region(south,     "South",   flip=True)}'
                  f'<div style="height:8px;"></div>'
                  f'{_b_region(west,     "West",    flip=True)}</div>')

    # Left-aligned (not centered) so horizontal overflow is only to the right,
    # making the bracket scrollable from R64 → Final Four on narrow screens.
    bracket_row = (
        f'<div style="display:inline-flex;align-items:center;">'
        f'{left_html}{center_html}{right_html}'
        f'</div>'
    )

    return (
        '<!DOCTYPE html><html><body style="margin:0;padding:8px;'
        'background:#080e17;font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        'overflow-x:auto;white-space:nowrap;">'
        + bracket_row +
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
        alpha = chaos / 100.0
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
            wp = (chaos/100)*0.5 + (1 - chaos/100)*wp
        f4_winner = 'a' if np.random.random() < wp else 'b'
        f4_display_games.append({
            'a': {'team': fav, 'seed': fs, 'prob': wp * 100},
            'b': {'team': dog, 'seed': ds, 'prob': (1-wp) * 100},
            'winner': f4_winner,
        })

    bracket_html = _build_full_bracket_html(
        bracket_vis, f4_display_games, champ_candidate,
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
