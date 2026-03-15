import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.utils.db import query_df
from src.model.predict import project_game

st.set_page_config(page_title="Model Results", page_icon="📋", layout="wide")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem; font-weight: 600; padding: 6px 16px; color: #555 !important;
}
.stTabs [aria-selected="true"] {
    color: #e74c3c !important; border-bottom: 2px solid #e74c3c !important;
}
.result-win  { color: #2ecc71; font-weight: bold; }
.result-loss { color: #e74c3c; font-weight: bold; }
.result-push { color: #f39c12; font-weight: bold; }
.result-none { color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

current_year = TOURNAMENT_YEARS[-1]

# ── Grading helpers ────────────────────────────────────────────────────────────

def _grade_ats(actual_margin_a: float, market_spread_a: float | None,
               spread_edge: float | None) -> str:
    """
    Grade ATS from the MODEL's pick perspective.
    spread_edge > 0 → model likes team_a → win if actual_margin > market_spread
    spread_edge < 0 → model likes team_b → win if actual_margin < market_spread
    """
    if market_spread_a is None or spread_edge is None:
        return "—"
    if spread_edge >= 0:
        if actual_margin_a > market_spread_a:  return "WIN"
        if actual_margin_a == market_spread_a: return "PUSH"
        return "LOSS"
    else:
        if actual_margin_a < market_spread_a:  return "WIN"
        if actual_margin_a == market_spread_a: return "PUSH"
        return "LOSS"


def _grade_ou(actual_total: float, market_total: float | None,
              total_edge: float | None) -> str:
    if market_total is None or total_edge is None:
        return "—"
    if total_edge > 0:
        if actual_total > market_total: return "WIN"
        if actual_total == market_total: return "PUSH"
        return "LOSS"
    elif total_edge < 0:
        if actual_total < market_total: return "WIN"
        if actual_total == market_total: return "PUSH"
        return "LOSS"
    return "PUSH"


def _roi(results: list) -> float:
    """ROI at -110 juice on WIN/LOSS results (excluding PUSH and —)."""
    graded = [r for r in results if r in ("WIN", "LOSS")]
    if not graded:
        return float("nan")
    wins = graded.count("WIN")
    stake = 110.0
    payout = 100.0
    total_wagered = len(graded) * stake
    total_returned = wins * (stake + payout) + (len(graded) - wins) * 0
    return (total_returned - total_wagered) / total_wagered * 100


def _record_str(results: list) -> str:
    graded = [r for r in results if r in ("WIN", "LOSS", "PUSH")]
    if not graded:
        return "—"
    w = graded.count("WIN")
    l = graded.count("LOSS")
    p = graded.count("PUSH")
    pct = f" ({w/(w+l)*100:.0f}%)" if (w + l) > 0 else ""
    return f"{w}-{l}{f'-{p}' if p else ''}{pct}"


# ── Core data loader ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Grading games…")
def load_graded_data(year: int) -> pd.DataFrame:
    """
    Build a graded DataFrame for all completed tournament games in `year`.
    Joins historical_results with historical_lines (pre-game closing lines),
    runs the model projection for each game, and grades ATS + O/U.
    """
    # ── 1. Actual results ──────────────────────────────────────────────────────
    results = query_df(
        "SELECT * FROM historical_results WHERE year = ? ORDER BY game_date, round_number",
        params=[year],
    )
    if results.empty:
        return pd.DataFrame()

    # ── 2. Pre-game lines ──────────────────────────────────────────────────────
    # For current year: check odds_history (earliest poll per game) first,
    # then fall back to historical_lines.
    lines = query_df(
        "SELECT * FROM historical_lines WHERE year = ?",
        params=[year],
    )

    # Build a quick lookup: frozenset{team1_lower, team2_lower} → line row
    def _line_key(t1, t2):
        return frozenset({str(t1).lower(), str(t2).lower()})

    lines_lookup = {}
    for _, lr in lines.iterrows():
        k = _line_key(lr["team1"], lr["team2"])
        lines_lookup[k] = lr

    # ── 3. Project each game & grade ───────────────────────────────────────────
    rows = []
    for _, g in results.iterrows():
        t1 = str(g["team1"])
        t2 = str(g["team2"])
        round_num = int(g["round_number"]) if pd.notna(g.get("round_number")) else 1

        # Run model
        try:
            proj = project_game(t1, t2, round_num=round_num, year=year)
        except Exception:
            proj = {"error": "model failed"}

        if "error" in proj:
            # Store result even without projection (shows score, no grade)
            rows.append({
                "game_date": g["game_date"],
                "round_num": round_num,
                "round_name": g.get("round_name") or ROUND_NAMES.get(round_num, "?"),
                "team_a": t1, "team_b": t2,
                "actual_score_a": g.get("score1"), "actual_score_b": g.get("score2"),
                "actual_margin": g.get("margin"), "actual_total": g.get("total_points"),
                "model_spread": None, "model_total": None,
                "market_spread_a": None, "market_total": None,
                "spread_edge": None, "total_edge": None,
                "ats_result": "—", "ou_result": "—",
                "has_line": False,
            })
            continue

        team_a = proj["team_a"]  # canonical ordering (alphabetically first)
        team_b = proj["team_b"]
        model_spread = round(proj["projected_spread"], 1)   # + = team_a favored
        model_total  = round(proj["projected_total"], 1)

        # Actual result from team_a perspective
        if t1.lower() == team_a.lower():
            actual_margin_a = float(g["margin"]) if pd.notna(g.get("margin")) else None
            score_a, score_b = g.get("score1"), g.get("score2")
        else:
            actual_margin_a = -float(g["margin"]) if pd.notna(g.get("margin")) else None
            score_a, score_b = g.get("score2"), g.get("score1")
        actual_total = float(g["total_points"]) if pd.notna(g.get("total_points")) else None

        # Pre-game market line from team_a perspective
        lr = lines_lookup.get(_line_key(t1, t2))
        market_spread_a = market_total = None
        if lr is not None and pd.notna(lr.get("spread_line")):
            raw = float(lr["spread_line"])
            # historical_lines convention: positive = team1 favored
            # If team_a == team1 → same direction; else flip
            market_spread_a = raw if t1.lower() == team_a.lower() else -raw
        if lr is not None and pd.notna(lr.get("total_line")):
            market_total = float(lr["total_line"])

        spread_edge = round(model_spread - market_spread_a, 1) if market_spread_a is not None else None
        total_edge  = round(model_total  - market_total,   1) if market_total    is not None else None

        ats = _grade_ats(actual_margin_a, market_spread_a, spread_edge)
        ou  = _grade_ou(actual_total, market_total, total_edge)

        rows.append({
            "game_date":      g["game_date"],
            "round_num":      round_num,
            "round_name":     g.get("round_name") or ROUND_NAMES.get(round_num, "?"),
            "team_a":         team_a,
            "team_b":         team_b,
            "actual_score_a": score_a,
            "actual_score_b": score_b,
            "actual_margin":  actual_margin_a,
            "actual_total":   actual_total,
            "model_spread":   model_spread,
            "model_total":    model_total,
            "market_spread_a": market_spread_a,
            "market_total":   market_total,
            "spread_edge":    spread_edge,
            "total_edge":     total_edge,
            "ats_result":     ats,
            "ou_result":      ou,
            "has_line":       market_spread_a is not None,
        })

    return pd.DataFrame(rows)


# ── Display helpers ────────────────────────────────────────────────────────────

def _result_badge(result: str) -> str:
    css = {"WIN": "result-win", "LOSS": "result-loss",
           "PUSH": "result-push"}.get(result, "result-none")
    return f"<span class='{css}'>{result}</span>"


def _spread_str(team: str, spread: float | None) -> str:
    if spread is None:
        return "—"
    return f"{team} {spread:+.1f}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    years_with_results = []
    for y in reversed(TOURNAMENT_YEARS):
        n = query_df("SELECT COUNT(*) as c FROM historical_results WHERE year=?",
                     params=[y]).iloc[0]["c"]
        if n > 0:
            years_with_results.append(y)

    if not years_with_results:
        st.warning("No results in DB yet.")
        st.stop()

    selected_year = st.selectbox(
        "Tournament year",
        years_with_results,
        format_func=lambda y: f"{y} ({y-1}–{str(y)[2:]} season)",
    )

    min_edge = st.slider("Min |spread edge| to show grade", 0.0, 8.0, 0.0, 0.5,
                         help="0 = show all graded lines")
    show_no_line = st.checkbox("Show games with no market line", value=False)
    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Load data ──────────────────────────────────────────────────────────────────
st.title("📋 Model Results vs Pre-Game Lines")
st.caption(
    f"ATS/O·U grading from the model's pick perspective · "
    f"Edge = model spread − pre-game closing line · "
    f"Win = model's predicted side covered"
)

df = load_graded_data(selected_year)

if df.empty:
    st.info(f"No completed results found for {selected_year}. "
            "Results populate as games are scraped from Sports Reference.")
    st.stop()

# Apply filters
view = df.copy()
if not show_no_line:
    view = view[view["has_line"]]
if min_edge > 0:
    view = view[view["spread_edge"].abs() >= min_edge]


# ── Summary KPIs ───────────────────────────────────────────────────────────────
graded = df[df["ats_result"].isin(["WIN", "LOSS", "PUSH"])]
graded_3 = graded[graded["spread_edge"].abs() >= 3]
ou_graded = df[df["ou_result"].isin(["WIN", "LOSS", "PUSH"])]
ou_graded_3 = ou_graded[ou_graded["total_edge"].abs() >= 3]

ats_all_str   = _record_str(graded["ats_result"].tolist())
ats_3_str     = _record_str(graded_3["ats_result"].tolist())
ou_all_str    = _record_str(ou_graded["ou_result"].tolist())
roi_3         = _roi(graded_3["ats_result"].tolist())
ou_roi_3      = _roi(ou_graded_3["ou_result"].tolist())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Games",         len(df))
k2.metric("w/ Lines",      int(graded["has_line"].sum()))
k3.metric("ATS (All)",     ats_all_str)
k4.metric("ATS (Edge≥3)",  ats_3_str)
k5.metric("ROI (Edge≥3)",  f"{roi_3:.1f}%" if not np.isnan(roi_3) else "—")
k6.metric("O/U (All)",     ou_all_str)

st.divider()


# ── Round summary bar ──────────────────────────────────────────────────────────
round_summary = []
for rn in range(1, 7):
    rdf = graded[graded["round_num"] == rn]
    if rdf.empty:
        continue
    w = (rdf["ats_result"] == "WIN").sum()
    l = (rdf["ats_result"] == "LOSS").sum()
    p = (rdf["ats_result"] == "PUSH").sum()
    round_summary.append({
        "Round": ROUND_NAMES.get(rn, f"R{rn}"),
        "W": int(w), "L": int(l), "P": int(p),
        "ATS %": f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "—",
        "ROI": f"{_roi(rdf['ats_result'].tolist()):.1f}%" if (w+l) > 0 else "—",
    })

if round_summary:
    st.subheader(f"📊 {selected_year} by Round")
    st.dataframe(
        pd.DataFrame(round_summary).set_index("Round"),
        use_container_width=True,
    )
    st.divider()


# ── Day-by-day tabs ────────────────────────────────────────────────────────────
dates = sorted(view["game_date"].unique())
if not dates:
    st.info("No games match current filters.")
    st.stop()

def _date_label(d: str) -> str:
    try:
        from datetime import date
        dt = date.fromisoformat(d)
        return dt.strftime("%a %b %d")
    except Exception:
        return d

date_tabs = st.tabs([_date_label(d) for d in dates])

for date_str, tab in zip(dates, date_tabs):
    day_df = view[view["game_date"] == date_str].copy()
    day_graded = day_df[day_df["ats_result"].isin(["WIN", "LOSS", "PUSH"])]

    with tab:
        # Day header with W-L summary
        day_ats = _record_str(day_graded["ats_result"].tolist())
        day_roi = _roi(day_graded["ats_result"].tolist())
        roi_str = f" · ROI {day_roi:.1f}%" if not np.isnan(day_roi) else ""
        st.markdown(
            f"**{_date_label(date_str)}** &nbsp;·&nbsp; ATS: {day_ats}{roi_str}",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Per-game rows
        for _, row in day_df.sort_values("round_num").iterrows():
            rnd_label = row["round_name"]

            # Score / result
            sa, sb = row.get("actual_score_a"), row.get("actual_score_b")
            if pd.notna(sa) and pd.notna(sb):
                score_str = f"{int(sa)} – {int(sb)}"
                winner = row["team_a"] if row["actual_margin"] > 0 else row["team_b"]
            else:
                score_str = "TBD"
                winner = None

            # Spread display strings (team_a perspective)
            model_sp  = _spread_str(row["team_a"], row.get("model_spread"))
            market_sp = _spread_str(row["team_a"], row.get("market_spread_a"))
            actual_sp = (f"{row['actual_margin']:+.0f}" if pd.notna(row.get("actual_margin")) else "—")

            edge_val = row.get("spread_edge")
            edge_str = f"{edge_val:+.1f}" if pd.notna(edge_val) and edge_val is not None else "—"

            ats = row["ats_result"]
            ou  = row["ou_result"]

            # Total display
            mt   = row.get("market_total")
            mod_t = row.get("model_total")
            act_t = row.get("actual_total")
            te    = row.get("total_edge")
            if mt is not None:
                ou_str = f"O/U {mt:.1f} · model {mod_t:.1f} · actual {int(act_t) if pd.notna(act_t) else '?'}"
            else:
                ou_str = ""

            # Pick direction for display
            if pd.notna(edge_val) and edge_val is not None:
                pick_team = row["team_a"] if edge_val >= 0 else row["team_b"]
                pick_spread = row["market_spread_a"] if edge_val >= 0 else (
                    -row["market_spread_a"] if row["market_spread_a"] is not None else None
                )
                pick_str = _spread_str(pick_team, pick_spread)
            else:
                pick_str = "—"

            c1, c2, c3, c4, c5, c6 = st.columns([3, 1.5, 2, 2, 1.2, 1.2])

            with c1:
                winner_sym = "✓" if winner == row["team_a"] else ("✗" if winner else "")
                st.markdown(
                    f"<div style='padding:4px 0'>"
                    f"<span style='font-weight:bold'>{row['team_a']} vs {row['team_b']}</span> "
                    f"<span style='color:#aaa;font-size:0.75rem'>{rnd_label}</span><br>"
                    f"<span style='color:#ccc;font-size:0.85rem'>Final: {score_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>ACTUAL</div>"
                    f"<div style='font-size:1rem;font-weight:bold'>{actual_sp}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>MODEL</div>"
                    f"<div style='font-size:0.9rem'>{model_sp}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>PRE-GAME LINE</div>"
                    f"<div style='font-size:0.9rem'>{market_sp}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c4:
                edge_color = "#2ecc71" if (pd.notna(edge_val) and edge_val and abs(edge_val) >= 3) else "#aaa"
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>MODEL'S BET</div>"
                    f"<div style='font-size:0.9rem'>{pick_str}</div>"
                    f"<div style='color:{edge_color};font-size:0.8rem'>edge {edge_str} pts</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c5:
                ats_badge = _result_badge(ats)
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>ATS</div>"
                    f"<div style='font-size:1.05rem'>{ats_badge}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c6:
                ou_badge = _result_badge(ou)
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.75rem'>O/U</div>"
                    f"<div style='font-size:1.05rem'>{ou_badge}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if ou_str:
                st.caption(f"&nbsp;&nbsp;&nbsp;{ou_str}")
            st.divider()


# ── Running totals table ───────────────────────────────────────────────────────
if len(dates) > 1:
    st.subheader("🏃 Running Totals by Day")
    running_rows = []
    cum_w = cum_l = cum_p = 0
    for d in dates:
        day_graded = df[(df["game_date"] == d) & df["ats_result"].isin(["WIN", "LOSS", "PUSH"])]
        w = (day_graded["ats_result"] == "WIN").sum()
        l = (day_graded["ats_result"] == "LOSS").sum()
        p = (day_graded["ats_result"] == "PUSH").sum()
        cum_w += w; cum_l += l; cum_p += p
        roi_d = _roi(day_graded["ats_result"].tolist())
        running_rows.append({
            "Date": _date_label(d),
            "Day ATS": f"{w}-{l}{f'-{p}' if p else ''}",
            "Cumulative": f"{cum_w}-{cum_l}{f'-{cum_p}' if cum_p else ''}",
            "Cum Win%": f"{cum_w/(cum_w+cum_l)*100:.0f}%" if (cum_w+cum_l) > 0 else "—",
            "Day ROI": f"{roi_d:.1f}%" if not np.isnan(roi_d) else "—",
        })
    st.dataframe(pd.DataFrame(running_rows).set_index("Date"), use_container_width=True)
