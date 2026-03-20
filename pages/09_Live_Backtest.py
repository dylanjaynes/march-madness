import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.model.predict import season_label

st.set_page_config(page_title="Live Backtest", page_icon="📊", layout="wide")
st.title("📊 Live Model Backtest")
st.caption(
    "Historical evaluation of the live in-game betting formula against "
    "tournament results. Uses closing pre-game lines as a proxy for "
    "live halftime market lines."
)

_HERE = Path(__file__).resolve().parent
LIVE_BT_PATH = _HERE.parent / "data" / "processed" / "live_backtest_results.json"


@st.cache_data(ttl=86400)
def load_live_backtest():
    if not LIVE_BT_PATH.exists():
        return None
    with open(LIVE_BT_PATH) as f:
        return json.load(f)


results = load_live_backtest()

if results is None:
    st.info(
        "No live backtest results found at "
        f"`data/processed/live_backtest_results.json`.\n\n"
        "Run the live backtest script when available to generate this file."
    )
    st.stop()

if st.button("🔄 Refresh cache"):
    st.cache_data.clear()
    st.rerun()

computed_at = results.get("computed_at", "unknown")
st.caption(f"Last computed: {computed_at}")

# ── Prominent caveat banner ────────────────────────────────────────────────────
st.warning(
    "**Methodology caveat:** This historical backtest uses closing pre-game lines as "
    "a proxy for live halftime lines. Live lines at halftime are tighter and incorporate "
    "in-game information — treat these numbers as directional upper bounds, not realistic "
    "expected returns. The 2026 live data will be the first valid backtest of this system.",
    icon="⚠️",
)

# ── Summary KPIs ───────────────────────────────────────────────────────────────
n_games      = results.get("n_games", 0)
n_with_edge  = results.get("n_with_edge", 0)
n_total_half = results.get("n_halftime_snaps", n_games)

overall_wins   = results.get("overall_wins", 0)
overall_losses = results.get("overall_losses", 0)
overall_total  = overall_wins + overall_losses
overall_cover  = overall_wins / overall_total * 100 if overall_total > 0 else 0.0

st.subheader("Summary")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Games Evaluated", f"{n_games:,}")
with k2:
    st.metric("Games with Edge Signal", f"{n_with_edge:,}",
              f"{n_with_edge / max(n_games, 1) * 100:.0f}% of games")
with k3:
    label = f"{overall_wins}-{overall_losses} ({overall_cover:.1f}%)" if overall_total > 0 else "—"
    st.metric("Overall Cover Rate", label, f"n={overall_total}")
with k4:
    st.metric("Halftime Snapshots", f"{n_total_half:,}")

st.divider()

# ── Cover rate by scenario ─────────────────────────────────────────────────────
st.subheader("Cover Rate by Game Scenario")
st.caption(
    "FAV_AHEAD = favorite is winning at halftime. "
    "DOG_WINNING = underdog is winning at halftime. "
    "n= shown for all rates."
)

scenarios = results.get("by_scenario", {})
if scenarios:
    scenario_rows = []
    for scenario_key, data in scenarios.items():
        w   = data.get("wins", 0)
        l   = data.get("losses", 0)
        tot = w + l
        pct = w / tot * 100 if tot > 0 else 0.0
        scenario_rows.append({
            "Scenario":    scenario_key,
            "W":           w,
            "L":           l,
            "n":           tot,
            "Cover Rate":  f"{pct:.1f}%" if tot > 0 else "—",
            "Win %":       round(pct, 1) if tot > 0 else 0.0,
        })

    scen_df = pd.DataFrame(scenario_rows)

    s_left, s_right = st.columns(2)
    with s_left:
        st.dataframe(
            scen_df[["Scenario", "W", "L", "n", "Cover Rate"]],
            use_container_width=True,
            hide_index=True,
        )
    with s_right:
        if not scen_df.empty and scen_df["n"].sum() > 0:
            fig_scen = go.Figure(go.Bar(
                x=scen_df["Scenario"],
                y=scen_df["Win %"],
                marker_color=["#2ecc71" if v >= 52.4 else "#e74c3c"
                              for v in scen_df["Win %"]],
                text=[f"n={r['n']}" for _, r in scen_df.iterrows()],
                textposition="outside",
            ))
            fig_scen.add_hline(
                y=52.4, line_dash="dash", line_color="#aaa",
                annotation_text="Break-even (52.4%)",
            )
            fig_scen.update_layout(
                yaxis_title="Cover Rate %",
                yaxis=dict(range=[0, 80]),
                title="Cover Rate by Halftime Scenario",
            )
            st.plotly_chart(fig_scen, use_container_width=True)
else:
    st.info("No scenario breakdown available in results file.")

st.divider()

# ── Cover rate by edge threshold ───────────────────────────────────────────────
st.subheader("Cover Rate by Edge Threshold")
st.caption("Covers when model's bet side outperforms the closing spread at that snapshot.")

by_edge = results.get("by_edge_threshold", {})
if by_edge:
    edge_rows = []
    for threshold_key in sorted(by_edge.keys(), key=lambda x: float(str(x).replace(">=", "").strip())):
        data = by_edge[threshold_key]
        w   = data.get("wins", 0)
        l   = data.get("losses", 0)
        tot = w + l
        pct = w / tot * 100 if tot > 0 else 0.0
        low_sample = tot < 20
        cover_str = f"{pct:.1f}%" if tot > 0 else "—"
        if low_sample and tot > 0:
            cover_str = f"{cover_str} ⚠️ low sample"
        edge_rows.append({
            "Edge Threshold": threshold_key,
            "W":    w,
            "L":    l,
            "n":    tot,
            "Cover Rate": cover_str,
            "Win %":      round(pct, 1) if tot > 0 else 0.0,
            "_low_sample": low_sample,
        })

    edge_df = pd.DataFrame(edge_rows)

    e_left, e_right = st.columns(2)
    with e_left:
        display_cols = ["Edge Threshold", "W", "L", "n", "Cover Rate"]
        st.dataframe(
            edge_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
        low_n_rows = edge_df[edge_df["_low_sample"] & (edge_df["n"] > 0)]
        if not low_n_rows.empty:
            st.caption(
                "⚠️ = n < 20 games. Rates at these thresholds are not statistically reliable."
            )

    with e_right:
        plot_df = edge_df[edge_df["n"] > 0]
        if not plot_df.empty:
            fig_edge = go.Figure()
            fig_edge.add_trace(go.Scatter(
                x=plot_df["Edge Threshold"].astype(str),
                y=plot_df["Win %"],
                mode="lines+markers",
                name="Cover Rate",
                line_color="#3498db",
                marker=dict(
                    size=10,
                    color=["#f39c12" if ls else "#3498db"
                           for ls in plot_df["_low_sample"]],
                ),
                text=[f"n={n}" for n in plot_df["n"]],
                hovertemplate="%{x}: %{y:.1f}% (%{text})<extra></extra>",
            ))
            fig_edge.add_hline(
                y=52.4, line_dash="dash", line_color="#e74c3c",
                annotation_text="Break-even (52.4%)",
            )
            fig_edge.update_layout(
                yaxis_title="Cover Rate %",
                yaxis=dict(range=[0, 80]),
                title="Live Model Cover Rate by Edge Threshold",
            )
            st.plotly_chart(fig_edge, use_container_width=True)
else:
    st.info("No edge-threshold breakdown available in results file.")

st.divider()

# ── Year-by-year breakdown ─────────────────────────────────────────────────────
per_year = results.get("per_year", {})
if per_year:
    st.subheader("Year-by-Year Performance")

    year_rows = []
    for year_key, data in sorted(per_year.items(), key=lambda kv: int(kv[0])):
        yr   = int(year_key)
        w    = data.get("wins", 0)
        l    = data.get("losses", 0)
        tot  = w + l
        pct  = w / tot * 100 if tot > 0 else 0.0
        n_gs = data.get("n_games", tot)
        year_rows.append({
            "Year":        yr,
            "Season":      season_label(yr),
            "Games":       n_gs,
            "W":           w,
            "L":           l,
            "n (graded)":  tot,
            "Cover Rate":  f"{pct:.1f}% (n={tot})" if tot > 0 else "—",
        })

    yr_df = pd.DataFrame(year_rows)
    st.dataframe(yr_df, use_container_width=True, hide_index=True)
    st.divider()

# ── Full game log ──────────────────────────────────────────────────────────────
game_log = results.get("game_log", [])
if game_log:
    with st.expander("Full Live Backtest Game Log", expanded=False):
        log_df = pd.DataFrame(game_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

# ── Methodology footnote ───────────────────────────────────────────────────────
st.caption(
    "**Methodology:** Formula-based live projection blends current score margin "
    "(weighted by time elapsed) with pre-game model spread (weighted by time remaining), "
    "plus in-game eFG%, rebounding, and turnover adjustments. "
    "Closing pre-game spread used as proxy for live halftime line — actual live lines "
    "are tighter, making these cover rates upper-bound estimates. "
    "Break-even at standard -110 juice = 52.4% win rate."
)
