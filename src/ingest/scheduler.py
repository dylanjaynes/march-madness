from apscheduler.schedulers.background import BackgroundScheduler
from src.ingest.odds import poll_and_store_odds
from src.ingest.torvik import refresh_team_ratings


def _sync_results():
    """Fetch latest ESPN results + closing lines for the current tournament year."""
    try:
        from src.utils.config import TOURNAMENT_YEARS
        from src.ingest.odds import fetch_and_store_scores
        year = TOURNAMENT_YEARS[-1]
        counts = fetch_and_store_scores(year)
        print(f"  [scheduler] results sync: {counts}")
    except Exception as e:
        print(f"  [scheduler] results sync error: {e}")


def start_scheduler() -> BackgroundScheduler:
    """Start background scheduler for odds + ratings + results refresh."""
    scheduler = BackgroundScheduler()

    # Refresh odds every 30 minutes during tournament
    scheduler.add_job(poll_and_store_odds, "interval", minutes=30, id="odds_refresh")

    # Refresh team ratings once daily at 6 AM ET
    scheduler.add_job(refresh_team_ratings, "cron", hour=6, id="ratings_refresh")

    # Sync current-year results + closing lines every 6 hours
    scheduler.add_job(_sync_results, "interval", hours=6, id="results_sync")

    scheduler.start()
    print("  [scheduler] Started: odds every 30min, ratings daily 6AM, results every 6h")
    return scheduler
