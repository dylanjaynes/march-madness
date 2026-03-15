import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.utils.db import query_df, db_conn, upsert_df
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
</style>
""", unsafe_allow_html=True)

current_year = TOURNAMENT_YEARS[-1]


# ── Grading helpers ────────────────────────────────────────────────────────────

def _grade_ats(actual_margin_a: float, market_spread_a, spread_edge) -> str:
    """
    Grade ATS from the model's pick perspective.
    Convention throughout: positive = team_a is favored / wins.

    market_spread_a > 0  → team_a is the favorite (e.g. -7 line → stored as +7)
    spread_edge    > 0  → model more bullish on team_a than market → bet team_a
    spread_edge    < 0  → model likes team_b → bet team_b

    team_a covers if actual_margin_a > market_spread_a  (exceeds the line)
    team_b covers if actual_margin_a < market_spread_a  (falls short of the line)
    """
    if market_spread_a is None or spread_edge is None:
        return "—"
    if spread_edge >= 0:                           # bet team_a
        if actual_margin_a > market_spread_a:  return "WIN"
        if actual_margin_a == market_spread_a: return "PUSH"
        return "LOSS"
    else:                                          # bet team_b
        if actual_margin_a < market_spread_a:  return "WIN"
        if actual_margin_a == market_spread_a: return "PUSH"
        return "LOSS"


def _grade_ou(actual_total, market_total, total_edge) -> str:
    if market_total is None or total_edge is None or actual_total is None:
        return "—"
    if total_edge > 0:   # model says Over
        if actual_total > market_total: return "WIN"
        if actual_total == market_total: return "PUSH"
        return "LOSS"
    elif total_edge < 0:  # model says Under
        if actual_total < market_total: return "WIN"
        if actual_total == market_total: return "PUSH"
        return "LOSS"
    return "PUSH"


def _roi(results: list) -> float:
    """ROI at -110 juice. Pushes excluded from denominator."""
    graded = [r for r in results if r in ("WIN", "LOSS")]
    if not graded:
        return float("nan")
    wins = graded.count("WIN")
    stake, payout = 110.0, 100.0
    total_wagered = len(graded) * stake
    total_returned = wins * (stake + payout)
    return (total_returned - total_wagered) / total_wagered * 100


def _record_str(results: list, with_pct: bool = True) -> str:
    graded = [r for r in results if r in ("WIN", "LOSS", "PUSH")]
    if not graded:
        return "—"
    w = graded.count("WIN")
    l = graded.count("LOSS")
    p = graded.count("PUSH")
    pct = f" ({w/(w+l)*100:.0f}%)" if with_pct and (w + l) > 0 else ""
    return f"{w}-{l}{f'-{p}' if p else ''}{pct}"


def _result_color(r: str) -> str:
    return {"WIN": "#2ecc71", "LOSS": "#e74c3c", "PUSH": "#f39c12"}.get(r, "#7f8c8d")


def _result_html(r: str) -> str:
    color = _result_color(r)
    return f"<span style='color:{color};font-weight:bold'>{r}</span>"


def _date_label(d: str) -> str:
    try:
        import datetime
        dt = datetime.date.fromisoformat(d)
        return dt.strftime("%a %b %d")
    except Exception:
        return d


# ── Fetch latest results for current year ────────────────────────────────────

def _fetch_and_store_results(year: int) -> int:
    """Scrape Sports Reference for tournament results and upsert into DB."""
    from src.ingest.historical import scrape_tournament_results
    df = scrape_tournament_results(year)
    if df.empty:
        return 0
    # Delete existing and reinsert (scraper is idempotent)
    with db_conn() as conn:
        conn.execute("DELETE FROM historical_results WHERE year = ?", [year])
    upsert_df(df, "historical_results", if_exists="append")
    return len(df)


# ── Core data loader ───────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner="Grading games…")
def load_graded_data(year: int) -> pd.DataFrame:
    """
    Grade all completed tournament games for `year`.

    SIGN CONVENTION (important — everything expressed from team_a perspective):
      team_a          = lower-seed (better seed) — matches training data ordering
      projected_spread = positive  → team_a wins by that many pts
      market_spread_a  = positive  → team_a is the favorite
      actual_margin_a  = score_a − score_b  (positive = team_a won, NEVER abs())
      spread_edge      = model_spread − market_spread_a
        > 0 → model more bullish on team_a → bet team_a
        < 0 → model more bullish on team_b → bet team_b
    """
    # ── 1. Completed game results ──────────────────────────────────────────────
    results = query_df(
        "SELECT * FROM historical_results WHERE year = ? AND score1 IS NOT NULL AND score2 IS NOT NULL "
        "ORDER BY game_date, round_number",
        params=[year],
    )
    if results.empty:
        return pd.DataFrame()

    # ── 2. Pre-game closing lines ──────────────────────────────────────────────
    lines = query_df(
        "SELECT * FROM historical_lines WHERE year = ?",
        params=[year],
    )

    # Build lookup: frozenset{team1_lower, team2_lower} → row
    lines_lookup = {}
    for _, lr in lines.iterrows():
        k = frozenset({str(lr["team1"]).lower(), str(lr["team2"]).lower()})
        lines_lookup[k] = lr

    # ── 3. Grade each game ─────────────────────────────────────────────────────
    rows = []
    for _, g in results.iterrows():
        t1 = str(g["team1"])
        t2 = str(g["team2"])
        round_num = int(g["round_number"]) if pd.notna(g.get("round_number")) else 1

        # Raw scores — use directly, NEVER use g["margin"] (it's always abs())
        score1 = float(g["score1"])
        score2 = float(g["score2"])
        seed1 = int(g["seed1"]) if pd.notna(g.get("seed1")) else 8
        seed2 = int(g["seed2"]) if pd.notna(g.get("seed2")) else 8

        # Project — pass seeds so team_a = lower seed, matching training convention
        try:
            proj = project_game(t1, t2, round_num=round_num, year=year,
                                seed_a=seed1, seed_b=seed2)
        except Exception:
            proj = {"error": "model failed"}

        if "error" in proj:
            rows.append({
                "game_date": g["game_date"],
                "round_num": round_num,
                "round_name": g.get("round_name") or ROUND_NAMES.get(round_num, "?"),
                "team_a": t1, "team_b": t2,
                "score_a": score1, "score_b": score2,
                "actual_margin_a": score1 - score2,
                "actual_total": score1 + score2,
                "model_spread": None, "model_total": None,
                "market_spread_a": None, "market_total": None,
                "spread_edge": None, "total_edge": None,
                "ats_result": "—", "ou_result": "—",
                "has_line": False,
            })
            continue

        team_a = proj["team_a"]   # canonical: lower-seed team (matches training)
        team_b = proj["team_b"]
        model_spread = round(proj["projected_spread"], 1)  # + = team_a wins
        model_total  = round(proj["projected_total"], 1)

        # Actual margin from team_a's perspective using raw scores (NOT margin column)
        if t1.lower() == team_a.lower():
            actual_margin_a = score1 - score2   # team_a=team1: a−b
            score_a, score_b = score1, score2
        else:
            actual_margin_a = score2 - score1   # team_a=team2: a−b = score2−score1
            score_a, score_b = score2, score1

        actual_total = score1 + score2

        # Pre-game market line from team_a perspective
        # historical_lines.spread_line convention: positive = team1 favored
        # We convert to team_a perspective (positive = team_a favored)
        lr = lines_lookup.get(frozenset({t1.lower(), t2.lower()}))
        market_spread_a = market_total = None
        if lr is not None and pd.notna(lr.get("spread_line")):
            raw = float(lr["spread_line"])
            market_spread_a = raw if t1.lower() == team_a.lower() else -raw
        if lr is not None and pd.notna(lr.get("total_line")):
            market_total = float(lr["total_line"])

        spread_edge = round(model_spread - market_spread_a, 1) if market_spread_a is not None else None
        total_edge  = round(model_total  - market_total,   1) if market_total    is not None else None

        ats = _grade_ats(actual_margin_a, market_spread_a, spread_edge)
        ou  = _grade_ou(actual_total, market_total, total_edge)

        rows.append({
            "game_date":       g["game_date"],
            "round_num":       round_num,
            "round_name":      g.get("round_name") or ROUND_NAMES.get(round_num, "?"),
            "team_a":          team_a,
            "team_b":          team_b,
            "score_a":         int(score_a),
            "score_b":         int(score_b),
            "actual_margin_a": actual_margin_a,
            "actual_total":    actual_total,
            "model_spread":    model_spread,
            "model_total":     model_total,
            "market_spread_a": market_spread_a,
            "market_total":    market_total,
            "spread_edge":     spread_edge,
            "total_edge":      total_edge,
            "ats_result":      ats,
            "ou_result":       ou,
            "has_line":        market_spread_a is not None,
        })

    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")

    # Check which years have results in DB
    years_with_results = {}
    for y in reversed(TOURNAMENT_YEARS):
        n = int(query_df("SELECT COUNT(*) as c FROM historical_results WHERE year=?",
                         params=[y]).iloc[0]["c"])
        years_with_results[y] = n

    # Always include current year (even if 0 games — shows fetch button)
    displayable = [y for y, n in years_with_results.items() if n > 0 or y == current_year]
    if not displayable:
        displayable = [current_year]

    selected_year = st.selectbox(
        "Tournament year",
        displayable,
        format_func=lambda y: (
            f"{y} ({y-1}–{str(y)[2:]} season)"
            + (f" — {years_with_results.get(y, 0)} games" if years_with_results.get(y, 0) == 0 else "")
        ),
    )

    min_edge = st.slider("Min |spread edge| to grade", 0.0, 8.0, 0.0, 0.5,
                         help="0 = show all games with a pre-game line")
    show_no_line = st.checkbox("Show games with no pre-game line", value=False)

    st.divider()

    # Fetch/refresh results for selected year
    if st.button(f"📥 Fetch {selected_year} results", use_container_width=True,
                 help="Scrapes Sports Reference for the latest tournament scores"):
        with st.spinner(f"Fetching {selected_year} results from Sports Reference…"):
            try:
                n = _fetch_and_store_results(selected_year)
                st.cache_data.clear()
                st.success(f"Fetched {n} games for {selected_year}")
                st.rerun()
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    if st.button("🔄 Refresh cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Load data ──────────────────────────────────────────────────────────────────
st.title("📋 Model Results vs Pre-Game Lines")
st.caption(
    "ATS/O·U graded from the model's pick perspective · "
    "Edge = model spread − pre-game closing line · "
    "Pre-game lines from Odds API closing snapshot (T−14:00 UTC)"
)

n_in_db = years_with_results.get(selected_year, 0)
if n_in_db == 0:
    st.info(
        f"No completed results for {selected_year} in the database yet. "
        "Click **📥 Fetch results** in the sidebar after games are played."
    )
    st.stop()

df = load_graded_data(selected_year)

if df.empty:
    st.warning(f"No gradeable data found for {selected_year}.")
    st.stop()

# Apply filters
view = df.copy()
if not show_no_line:
    view = view[view["has_line"]]
if min_edge > 0:
    view = view[view["spread_edge"].notna() & (view["spread_edge"].abs() >= min_edge)]


# ── Summary KPIs ───────────────────────────────────────────────────────────────
graded   = df[df["ats_result"].isin(["WIN", "LOSS", "PUSH"])]
graded_3 = graded[graded["spread_edge"].abs() >= 3]
ou_all   = df[df["ou_result"].isin(["WIN", "LOSS", "PUSH"])]
ou_3     = ou_all[ou_all["total_edge"].abs() >= 3]

roi_3    = _roi(graded_3["ats_result"].tolist())
ou_roi_3 = _roi(ou_3["ou_result"].tolist())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Games",         len(df))
c2.metric("w/ Lines",      int(df["has_line"].sum()))
c3.metric("ATS (All)",     _record_str(graded["ats_result"].tolist()))
c4.metric("ATS (Edge≥3)",  _record_str(graded_3["ats_result"].tolist()))
c5.metric("ATS ROI (E≥3)", f"{roi_3:.1f}%" if not np.isnan(roi_3) else "—")
c6.metric("O/U (All)",     _record_str(ou_all["ou_result"].tolist(), with_pct=False))

st.divider()


# ── Round breakdown ────────────────────────────────────────────────────────────
round_rows = []
for rn in range(1, 7):
    rdf = graded[graded["round_num"] == rn]
    if rdf.empty:
        continue
    w = (rdf["ats_result"] == "WIN").sum()
    l = (rdf["ats_result"] == "LOSS").sum()
    p = (rdf["ats_result"] == "PUSH").sum()
    roi = _roi(rdf["ats_result"].tolist())
    round_rows.append({
        "Round":  ROUND_NAMES.get(rn, f"R{rn}"),
        "W": int(w), "L": int(l), "P": int(p),
        "ATS %":  f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "—",
        "ROI (-110)": f"{roi:.1f}%" if not np.isnan(roi) else "—",
    })

if round_rows:
    st.subheader(f"📊 {selected_year} Round Breakdown")
    st.dataframe(pd.DataFrame(round_rows).set_index("Round"), use_container_width=True)
    st.divider()


# ── Day-by-day tabs ────────────────────────────────────────────────────────────
dates = sorted(view["game_date"].unique())
if not dates:
    st.info("No games match current filters.")
    st.stop()

date_tabs = st.tabs([_date_label(d) for d in dates])

for date_str, tab in zip(dates, date_tabs):
    day_df = view[view["game_date"] == date_str].sort_values("round_num")
    day_graded = day_df[day_df["ats_result"].isin(["WIN", "LOSS", "PUSH"])]
    day_roi = _roi(day_graded["ats_result"].tolist())
    roi_str = f" · ROI {day_roi:.1f}%" if not np.isnan(day_roi) else ""

    with tab:
        st.markdown(
            f"**{_date_label(date_str)}** &nbsp;·&nbsp; "
            f"ATS: {_record_str(day_graded['ats_result'].tolist())}{roi_str}",
            unsafe_allow_html=True,
        )
        st.markdown("")

        for _, row in day_df.iterrows():
            score_a = row.get("score_a")
            score_b = row.get("score_b")
            final_str = f"{int(score_a)} – {int(score_b)}" if pd.notna(score_a) and pd.notna(score_b) else "TBD"
            winner_team = row["team_a"] if row["actual_margin_a"] > 0 else row["team_b"]

            # Spread strings — always from team_a perspective
            def sp_str(team, val):
                return f"{team} {val:+.1f}" if val is not None and pd.notna(val) else "—"

            model_sp  = sp_str(row["team_a"], row.get("model_spread"))
            market_sp = sp_str(row["team_a"], row.get("market_spread_a"))
            actual_sp = f"{row['actual_margin_a']:+.0f}" if pd.notna(row["actual_margin_a"]) else "—"

            edge_val = row.get("spread_edge")
            edge_str = f"{edge_val:+.1f} pts" if pd.notna(edge_val) and edge_val is not None else "—"
            edge_col = "#2ecc71" if (pd.notna(edge_val) and edge_val and abs(edge_val) >= 3) else "#aaa"

            # Which side did the model back?
            if pd.notna(edge_val) and edge_val is not None:
                if edge_val >= 0:
                    pick_team = row["team_a"]
                    pick_line = row.get("market_spread_a")
                    pick_disp = sp_str(pick_team, pick_line)
                else:
                    pick_team = row["team_b"]
                    pick_line = -row["market_spread_a"] if row.get("market_spread_a") is not None else None
                    pick_disp = sp_str(pick_team, pick_line)
            else:
                pick_team = row["team_a"]
                pick_disp = "—"

            ats = row["ats_result"]
            ou  = row["ou_result"]

            # O/U caption
            mt = row.get("market_total")
            mt_str = f"O/U {mt:.1f} · model {row['model_total']:.1f} · actual {int(row['actual_total'])}" if mt else ""

            c1, c2, c3, c4, c5, c6 = st.columns([3, 1.4, 2.2, 2.2, 1.2, 1.2])

            with c1:
                winner_sym = " ✓" if winner_team == row["team_a"] else ""
                st.markdown(
                    f"<div style='padding:4px 0'>"
                    f"<b>{row['team_a']}{winner_sym}</b> vs <b>{row['team_b']}{'  ✓' if winner_team == row['team_b'] else ''}</b> "
                    f"<span style='color:#aaa;font-size:0.75rem'>({row['round_name']})</span><br>"
                    f"<span style='color:#ccc;font-size:0.85rem'>Final: {final_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.72rem'>ACTUAL</div>"
                    f"<div style='font-size:1rem;font-weight:bold'>{actual_sp}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.72rem'>MODEL SPREAD</div>"
                    f"<div style='font-size:0.9rem'>{model_sp}</div>"
                    f"<div style='color:#aaa;font-size:0.72rem;margin-top:2px'>PRE-GAME LINE</div>"
                    f"<div style='font-size:0.9rem'>{market_sp}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c4:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.72rem'>MODEL'S BET</div>"
                    f"<div style='font-size:0.9rem'>{pick_disp}</div>"
                    f"<div style='color:{edge_col};font-size:0.8rem;margin-top:2px'>edge {edge_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c5:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.72rem'>ATS</div>"
                    f"<div style='font-size:1.05rem'>{_result_html(ats)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c6:
                st.markdown(
                    f"<div style='text-align:center;padding:4px'>"
                    f"<div style='color:#aaa;font-size:0.72rem'>O/U</div>"
                    f"<div style='font-size:1.05rem'>{_result_html(ou)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if mt_str:
                st.caption(f"&nbsp;&nbsp;&nbsp;{mt_str}")
            st.divider()


# ── Running totals ─────────────────────────────────────────────────────────────
if len(dates) > 1:
    st.subheader("🏃 Running ATS Totals")
    cum_w = cum_l = cum_p = 0
    run_rows = []
    for d in dates:
        day_g = df[(df["game_date"] == d) & df["ats_result"].isin(["WIN", "LOSS", "PUSH"])]
        w = int((day_g["ats_result"] == "WIN").sum())
        l = int((day_g["ats_result"] == "LOSS").sum())
        p = int((day_g["ats_result"] == "PUSH").sum())
        cum_w += w; cum_l += l; cum_p += p
        day_roi = _roi(day_g["ats_result"].tolist())
        run_rows.append({
            "Date":       _date_label(d),
            "Day":        f"{w}-{l}{f'-{p}' if p else ''}",
            "Cumulative": f"{cum_w}-{cum_l}{f'-{cum_p}' if cum_p else ''}",
            "Cum Win%":   f"{cum_w/(cum_w+cum_l)*100:.0f}%" if (cum_w+cum_l) > 0 else "—",
            "Day ROI":    f"{day_roi:.1f}%" if not np.isnan(day_roi) else "—",
        })
    st.dataframe(pd.DataFrame(run_rows).set_index("Date"), use_container_width=True)
