from apscheduler.schedulers.background import BackgroundScheduler
from src.ingest.odds import poll_and_store_odds
from src.ingest.torvik import refresh_team_ratings


def start_scheduler() -> BackgroundScheduler:
    """Start background scheduler for odds + ratings refresh."""
    scheduler = BackgroundScheduler()

    # Refresh odds every 30 minutes during tournament
    scheduler.add_job(poll_and_store_odds, "interval", minutes=30, id="odds_refresh")

    # Refresh team ratings once daily at 6 AM ET
    scheduler.add_job(refresh_team_ratings, "cron", hour=6, id="ratings_refresh")

    scheduler.start()
    print("  [scheduler] Started: odds every 30min, ratings daily at 6AM ET")
    return scheduler
