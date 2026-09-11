# workers/scheduler.py
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from database import SessionLocal
from services.reminder_service import ReminderService

log = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def run_30_day_reminder_job():
    log.info("⏰ [APScheduler] Running 30-day reassessment check...")
    db = SessionLocal()
    try:
        notifications = ReminderService.check_and_trigger_30_day_reminders(db)
        log.info(f"⏰ [APScheduler] Check complete — {len(notifications)} new notification(s) generated.")
    except Exception as e:
        log.error(f"❌ [APScheduler] Error in 30-day reminder job: {e}")
    finally:
        db.close()


def start_scheduler():
    if not scheduler.running:
        # Schedule daily check at 08:00 UTC
        scheduler.add_job(
            run_30_day_reminder_job,
            CronTrigger(hour=8, minute=0),
            id="daily_30_day_reassessment_check",
            replace_existing=True
        )
        scheduler.start()
        log.info("✓ APScheduler started successfully for automated 30-day reminders")


def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("APScheduler stopped.")
