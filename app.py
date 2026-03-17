import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Home | March Madness Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.db import init_db, get_table_count, query_df
from src.utils.config import TOURNAMENT_YEARS
from src.model.predict import season_label, data_as_of
from datetime import datetime


@st.cache_resource
def initialize_app():
    init_db()
    from pathlib import Path
    from src.utils.config import MODELS_DIR
    spread_path = MODELS_DIR / "spread_model.pkl"
    total_path  = MODELS_DIR / "total_model.pkl"

    # Check both models AND game data — DB may be empty even if models exist
    game_count = get_table_count("historical_results")
    models_ok  = spread_path.exists() and total_path.exists()

    needs_data    = game_count < 100
    needs_models  = not models_ok

    if needs_data or needs_models:
        msg = "First run: ingesting data & training models (~3 min)..." if needs_data else "First run: training models (~30 seconds)..."
        with st.spinner(msg):
            try:
                if needs_data:
                    from src.ingest.historical import build_historical_dataset
                    build_historical_dataset()
                    # Join any stored historical lines into training data
                    from src.ingest.join_lines import join_lines_to_training
                    join_lines_to_training()
                if needs_data or needs_models:
                    from src.model.train import run_full_training_pipeline
                    run_full_training_pipeline()
            except Exception as e:
                st.error(f"Initialization error: {e}")
    return True


def main():
    initialize_app()

    current_year = TOURNAMENT_YEARS[-1]
    season = season_label(current_year)
    data_note = data_as_of(current_year)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🏀 March Madness")
        st.caption(f"**{season} Season**")
        st.caption(f"Data: {data_note}")
        st.divider()
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/06_Live_Games.py", label="Live Games", icon="📡")
        st.page_link("pages/01_Bet_Board.py", label="Bet Board", icon="🎯")
        st.page_link("pages/02_Bracket_Projector.py", label="Bracket Projector", icon="🔢")
        st.page_link("pages/03_Matchup_Builder.py", label="Matchup Builder", icon="⚔️")
        st.page_link("pages/04_Backtest_Results.py", label="Backtest Results", icon="📈")
        st.page_link("pages/05_Teams.py", label="Teams", icon="📋")
        st.divider()
        st.caption(f"Refreshed: {datetime.now().strftime('%b %d %H:%M')}")

        st.divider()
        st.markdown("**Data**")
        if st.button("🔄 Refresh Ratings", use_container_width=True,
                     help="Pull latest BartTorvik ratings and retrain models"):
            with st.spinner("Fetching ratings from BartTorvik..."):
                try:
                    from src.ingest.torvik import store_team_ratings
                    n = len(store_team_ratings(current_year))
                    st.success(f"Updated {n} teams")
                except Exception as e:
                    st.error(f"Ratings fetch failed: {e}")
            with st.spinner("Retraining models..."):
                try:
                    from src.model.train import run_full_training_pipeline
                    run_full_training_pipeline()
                    st.success("Models retrained")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # Header
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.title("March Madness Projection Model")
        st.caption("Hybrid Ridge+XGBoost spread model · isotonic calibration · BartTorvik efficiency ratings")
    with col_badge:
        st.metric("Season", season)
        st.caption(data_note)

    st.divider()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        n_games = get_table_count("mm_training_data")
        st.metric("Training Games", n_games)
    with col2:
        n_ratings = get_table_count("torvik_ratings")
        st.metric("Team-Seasons", n_ratings)
    with col3:
        st.metric("Tournament Years", len(TOURNAMENT_YEARS))
    with col4:
        n_lines = get_table_count("historical_lines")
        st.metric("Historical Lines", n_lines)
    with col5:
        n_odds = get_table_count("odds_history")
        st.metric("Live Odds Snapshots", n_odds)

    st.divider()

    # Model performance summary
    st.subheader("Model Performance (Walk-Forward Backtest)")
    import json
    from pathlib import Path

    _HERE = Path(__file__).resolve().parent
    RESULTS_PATH = _HERE / "data" / "processed" / "backtest_results.json"

    def _load_backtest_summary():
        if not RESULTS_PATH.exists():
            return None
        try:
            with open(RESULTS_PATH) as f:
                return json.load(f)
        except Exception:
            return None

    bt = _load_backtest_summary()
    if bt is None:
        st.info("Run `python scripts/run_backtest.py` to generate backtest results.")
    else:
        computed_at = bt.get("computed_at", "unknown")
        st.caption(f"Last computed: {computed_at}")
        # Show 3-4 headline metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spread RMSE", f"{bt.get('spread_rmse', 0):.2f} pts")
        m2.metric("ATS Win % (competitive)", f"{bt.get('ats_pct_competitive', bt.get('ats_pct', 0)):.1f}%")
        m3.metric("Sample Size", str(bt.get("n_games", bt.get("sample_size", "—"))))
        m4.metric("O/U Win %", f"{bt.get('ou_pct', 0):.1f}%")

    st.divider()

    # Quick matchup tool
    st.subheader("Quick Matchup")
    st.caption("Enter any two teams for a one-click projection.")

    col_a, col_b, col_c, col_d = st.columns([2, 2, 1, 1])
    with col_a:
        team_a = st.text_input("Team A (favorite / higher seed)", value="Duke")
    with col_b:
        team_b = st.text_input("Team B (underdog / lower seed)", value="Houston")
    with col_c:
        round_num = st.selectbox(
            "Round", [1, 2, 3, 4, 5, 6],
            format_func=lambda r: {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}[r],
        )
    with col_d:
        market_spread = st.number_input(
            "Team A spread (opt)", value=0.0, step=0.5,
            help="Use betting convention: -7 means Team A is a 7-point favorite, +7 means 7-point underdog. Leave 0 to skip edge calc."
        )

    if st.button("Project", type="primary"):
        try:
            from src.model.predict import project_game, coverage_probability, bet_tier
            proj = project_game(team_a, team_b, round_num, year=current_year)
            if "error" in proj:
                st.error(proj["error"])
            else:
                # project_game may canonically reorder teams; normalise to caller's team_a perspective
                if proj["team_a"] == team_a:
                    ps  = proj["projected_spread"]
                    wpa = proj["win_prob_a"]
                else:
                    ps  = -proj["projected_spread"]   # flip to original team_a's view
                    wpa = proj["win_prob_b"]          # = 1 - win_prob_a
                pt  = proj["projected_total"]
                # Convert betting convention input (−7 = team_a favored) → model convention (+7)
                mkt = -market_spread if market_spread != 0.0 else None

                r1, r2, r3, r4, r5 = st.columns(5)
                with r1:
                    # Display in betting convention: negate so favorite shows as negative
                    st.metric("Model Spread", f"{team_a} {-ps:+.1f}")
                with r2:
                    st.metric("Model Total", f"{pt:.1f}")
                with r3:
                    st.metric(f"{team_a} Win Prob", f"{wpa:.1%}")
                with r4:
                    if mkt is not None:
                        edge = ps - mkt   # both model convention; positive = team_a covers
                        lean = team_a if edge > 0 else proj["team_b"]
                        st.metric("Edge", f"{edge:+.1f} pts", delta=f"lean {lean}")
                    else:
                        st.metric("Edge", "—")
                with r5:
                    if mkt is not None:
                        cov = coverage_probability(ps, mkt)
                        tier, emoji = bet_tier(edge, cov)
                        st.metric("Rating", f"{emoji} {tier}")
                    else:
                        st.metric("Rating", "—")

                st.caption(
                    f"Projected score: **{proj['team_a']} {proj['projected_score_a']:.0f}** – "
                    f"**{proj['team_b']} {proj['projected_score_b']:.0f}**  |  "
                    f"Data: {data_note}"
                )
        except Exception as e:
            st.error(f"Projection error: {e}")


if __name__ == "__main__":
    main()
