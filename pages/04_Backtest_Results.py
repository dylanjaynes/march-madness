import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.model.backtest import run_backtest, calculate_roi
from src.utils.config import TOURNAMENT_YEARS, ROUND_NAMES
from src.model.predict import season_label

st.set_page_config(page_title="Backtest Results", page_icon="📈", layout="wide")
st.title("📈 Backtest Results")
st.caption(
    "Walk-forward backtest: train on all prior years, test on each year in sequence. "
    "True ATS = actual margin vs. real market spread (not directional accuracy)."
)


@st.cache_data(ttl=3600)
def load_backtest():
    return run_backtest()


with st.spinner("Running walk-forward backtest..."):
    results = load_backtest()

preds_df: pd.DataFrame = results.get("predictions_df", pd.DataFrame())
per_year = results.get("per_year", {})


# ── True ATS calculation (vs real market lines) ───────────────────────────────
def compute_true_ats(df: pd.DataFrame, min_edge: float = 0.0):
    """
    True ATS: filter to real market lines (market_spread != 0),
    then check if actual_margin > market_spread (team_a covers).
    """
    if df.empty:
        return pd.DataFrame()
    filtered = df[df["market_spread"].abs() > 0].copy()
    if min_edge > 0:
        filtered = filtered[filtered["spread_edge"].abs() >= min_edge]
    if filtered.empty:
        return filtered
    filtered["true_ats"] = filtered.apply(
        lambda r: "WIN" if r["actual_margin"] > r["market_spread"]
        else ("PUSH" if r["actual_margin"] == r["market_spread"] else "LOSS"),
        axis=1,
    )
    return filtered


def ats_record(df: pd.DataFrame, col: str = "true_ats"):
    if df.empty or col not in df.columns:
        return (0, 0, 0)
    w = (df[col] == "WIN").sum()
    l = (df[col] == "LOSS").sum()
    p = (df[col] == "PUSH").sum()
    return (w, l, p)


def ats_str(record):
    w, l, p = record
    total = w + l
    pct = w / total * 100 if total > 0 else 0
    return f"{w}-{l}-{p} ({pct:.1f}%)"


# ── Top-level KPIs ────────────────────────────────────────────────────────────
st.subheader("Summary")

ats_all = compute_true_ats(preds_df, min_edge=0.0)
ats_3 = compute_true_ats(preds_df, min_edge=3.0)
ats_5 = compute_true_ats(preds_df, min_edge=5.0)

rec_all = ats_record(ats_all)
rec_3 = ats_record(ats_3)
rec_5 = ats_record(ats_5)

roi_3 = calculate_roi([r for r in ats_3["true_ats"].tolist() if r != "PUSH"]) if not ats_3.empty else 0.0
roi_5 = calculate_roi([r for r in ats_5["true_ats"].tolist() if r != "PUSH"]) if not ats_5.empty else 0.0

spread_rmse = results.get("spread_rmse", 0)
total_rmse = results.get("total_rmse", 0)
mkt_rmse = results.get("vs_market_spread_rmse", 0)
n_games = results.get("n_games_total", 0)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    w, l, p = rec_all
    total = w + l
    pct = w / total * 100 if total > 0 else 0
    st.metric("ATS (All games w/ lines)", f"{w}-{l}-{p}", f"{pct:.1f}%")
with k2:
    w3, l3, _ = rec_3
    total3 = w3 + l3
    pct3 = w3 / total3 * 100 if total3 > 0 else 0
    st.metric("ATS @ Edge ≥ 3", f"{w3}-{l3}", f"{pct3:.1f}% · {roi_3:+.1f}% ROI")
with k3:
    w5, l5, _ = rec_5
    total5 = w5 + l5
    pct5 = w5 / total5 * 100 if total5 > 0 else 0
    st.metric("ATS @ Edge ≥ 5", f"{w5}-{l5}", f"{pct5:.1f}% · {roi_5:+.1f}% ROI")
with k4:
    st.metric("Spread RMSE", f"{spread_rmse:.2f} pts",
              f"vs Market {mkt_rmse:.2f} pts")
with k5:
    st.metric("Total RMSE", f"{total_rmse:.2f} pts")

st.divider()

# ── Year-by-year table ────────────────────────────────────────────────────────
st.subheader("Year-by-Year Performance")

year_rows = []
for year, metrics in per_year.items():
    yr_df = compute_true_ats(preds_df[preds_df["year"] == year], min_edge=0.0)
    yr_df_3 = compute_true_ats(preds_df[preds_df["year"] == year], min_edge=3.0)
    rec = ats_record(yr_df)
    rec3 = ats_record(yr_df_3)
    roi = calculate_roi([r for r in yr_df_3["true_ats"].tolist() if r != "PUSH"]) if not yr_df_3.empty else 0.0
    w, l, _ = rec
    pct = w / (w + l) * 100 if (w + l) > 0 else 0
    year_rows.append({
        "Year": str(year),
        "Season": season_label(year),
        "Games (w/ lines)": len(yr_df),
        "ATS (All)": ats_str(rec),
        "ATS (Edge≥3)": ats_str(rec3),
        "ROI (Edge≥3)": f"{roi:+.1f}%",
        "Spread RMSE": round(metrics.get("spread_rmse", 0), 2),
        "Market RMSE": round(metrics.get("vs_market_spread_rmse", 0), 2),
        "Train Years": str(metrics.get("train_years", [])),
    })

year_df = pd.DataFrame(year_rows)
st.dataframe(year_df, use_container_width=True, hide_index=True)

st.divider()

# ── RMSE chart ────────────────────────────────────────────────────────────────
st.subheader("Model vs Market RMSE by Year")

if not year_df.empty:
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        x=year_df["Year"],
        y=year_df["Spread RMSE"],
        name="Model RMSE",
        marker_color="#3498db",
    ))
    fig_rmse.add_trace(go.Bar(
        x=year_df["Year"],
        y=year_df["Market RMSE"],
        name="Market RMSE",
        marker_color="#e74c3c",
    ))
    fig_rmse.update_layout(
        barmode="group",
        yaxis_title="RMSE (pts)",
        legend_title="",
        title="Lower is better — model should approach market RMSE",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

st.divider()

# ── Edge threshold performance ────────────────────────────────────────────────
st.subheader("True ATS Performance by Edge Threshold")

if not preds_df.empty:
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7]
    thresh_rows = []
    for t in thresholds:
        tdf = compute_true_ats(preds_df, min_edge=float(t))
        if tdf.empty:
            continue
        w, l, p = ats_record(tdf)
        total = w + l
        pct = w / total * 100 if total > 0 else 0
        roi = calculate_roi([r for r in tdf["true_ats"].tolist() if r != "PUSH"])
        thresh_rows.append({
            "Edge Threshold": f"≥{t} pts",
            "Games": total,
            "ATS W-L": f"{w}-{l}",
            "Win %": round(pct, 1),
            "ROI": round(roi, 1),
        })

    if thresh_rows:
        tdf_display = pd.DataFrame(thresh_rows)
        c_left, c_right = st.columns(2)
        with c_left:
            st.dataframe(tdf_display, use_container_width=True, hide_index=True)
        with c_right:
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(
                x=tdf_display["Edge Threshold"],
                y=tdf_display["Win %"],
                mode="lines+markers",
                name="ATS Win %",
                line_color="#2ecc71",
                marker_size=8,
            ))
            fig_thresh.add_hline(y=52.4, line_dash="dash", line_color="#e74c3c",
                                  annotation_text="Break-even (52.4%)")
            fig_thresh.update_layout(
                yaxis_title="ATS Win %",
                title="True ATS Win% by Edge Threshold",
                yaxis=dict(range=[40, 75]),
            )
            st.plotly_chart(fig_thresh, use_container_width=True)

    st.divider()

    # ROI curve
    st.subheader("ROI by Edge Threshold")
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Bar(
        x=tdf_display["Edge Threshold"],
        y=tdf_display["ROI"],
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in tdf_display["ROI"]],
        name="ROI %",
    ))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_roi.update_layout(yaxis_title="ROI (%)", title="Flat-bet ROI by Minimum Edge")
    st.plotly_chart(fig_roi, use_container_width=True)

    st.divider()

    # ── O/U analysis ──────────────────────────────────────────────────────────
    st.subheader("Over/Under Analysis")
    ou_df = preds_df[preds_df["market_total"] > 0].copy() if not preds_df.empty else pd.DataFrame()
    if not ou_df.empty:
        ou_df["true_ou"] = ou_df.apply(
            lambda r: "WIN" if (r["total_edge"] > 0 and r["actual_total"] > r["market_total"])
                               or (r["total_edge"] < 0 and r["actual_total"] < r["market_total"])
                      else "LOSS",
            axis=1,
        )
        ou_thresholds = [0, 1, 2, 3, 4, 5]
        ou_rows = []
        for t in ou_thresholds:
            tdf_ou = ou_df[ou_df["total_edge"].abs() >= t] if t > 0 else ou_df
            if tdf_ou.empty:
                continue
            w = (tdf_ou["true_ou"] == "WIN").sum()
            l = (tdf_ou["true_ou"] == "LOSS").sum()
            pct = w / (w + l) * 100 if (w + l) > 0 else 0
            roi_ou = calculate_roi(tdf_ou["true_ou"].tolist())
            ou_rows.append({
                "Edge Threshold": f"≥{t} pts",
                "Games": w + l,
                "W-L": f"{w}-{l}",
                "Win %": round(pct, 1),
                "ROI": round(roi_ou, 1),
            })
        if ou_rows:
            st.dataframe(pd.DataFrame(ou_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── By-round breakdown ────────────────────────────────────────────────────
    st.subheader("Performance by Round")
    if "round_number" in preds_df.columns:
        round_rows = []
        for rn in range(1, 7):
            rdf = compute_true_ats(preds_df[preds_df["round_number"] == rn])
            if rdf.empty:
                continue
            rmse_val = ((preds_df[preds_df["round_number"] == rn]["actual_margin"] -
                         preds_df[preds_df["round_number"] == rn]["model_spread"]) ** 2).mean() ** 0.5
            rec = ats_record(rdf)
            w, l, _ = rec
            pct = w / (w + l) * 100 if (w + l) > 0 else 0
            round_rows.append({
                "Round": ROUND_NAMES.get(rn, f"R{rn}"),
                "Games (w/ lines)": len(rdf),
                "Spread RMSE": round(rmse_val, 1),
                "ATS W-L": f"{w}-{l}",
                "ATS Win %": round(pct, 1),
            })
        if round_rows:
            st.dataframe(pd.DataFrame(round_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Full game log ─────────────────────────────────────────────────────────
    with st.expander("Full Backtest Game Log", expanded=False):
        display_cols = [
            "year", "round_number", "model_spread", "model_total",
            "market_spread", "market_total", "actual_margin", "actual_total",
            "spread_edge", "total_edge",
        ]
        log_df = preds_df[[c for c in display_cols if c in preds_df.columns]].copy()
        log_df["round_name"] = log_df["round_number"].map(ROUND_NAMES)
        log_df["true_ats"] = log_df.apply(
            lambda r: "WIN" if r["actual_margin"] > r["market_spread"]
            else ("PUSH" if r["actual_margin"] == r["market_spread"] else "LOSS")
            if r["market_spread"] != 0 else "NO LINE",
            axis=1,
        )
        log_df = log_df.sort_values(["year", "round_number"])
        for col in ["model_spread", "model_total", "market_spread", "market_total",
                    "actual_margin", "actual_total", "spread_edge", "total_edge"]:
            if col in log_df.columns:
                log_df[col] = log_df[col].round(1)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

else:
    st.info("No backtest predictions available. Ensure training data is loaded.")

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance")
st.caption("XGBoost feature importances from the final trained models (full dataset).")

@st.cache_data(ttl=3600)
def load_feature_importance():
    from src.model.train import load_model, get_feature_importance
    try:
        spread_model = load_model("spread_model")
        total_model  = load_model("total_model")
        return get_feature_importance(spread_model), get_feature_importance(total_model)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

fi_spread, fi_total = load_feature_importance()

if not fi_spread.empty:
    fi_col1, fi_col2 = st.columns(2)

    with fi_col1:
        st.markdown("**Spread Model**")
        fig_fi_spread = px.bar(
            fi_spread,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig_fi_spread.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_fi_spread, use_container_width=True)

    with fi_col2:
        st.markdown("**Total Model**")
        fig_fi_total = px.bar(
            fi_total,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Oranges",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig_fi_total.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_fi_total, use_container_width=True)

    # Dead-weight features (importance < 1%)
    dead = fi_spread[fi_spread["importance"] < 0.01]["feature"].tolist()
    if dead:
        st.caption(f"Low-impact features (< 1% importance): {', '.join(dead)} — candidates for removal.")
else:
    st.info("Train models first to see feature importance.")

st.divider()
st.caption(
    "**Methodology:** Walk-forward: train on all years before test year. "
    "True ATS uses real market spreads: team_a covers if `actual_margin > market_spread`. "
    "Market lines sourced from The Odds API historical data + SBRO Excel files. "
    "Break-even at standard -110 juice = 52.4% win rate."
)
