# reminder_service.py
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from models.domain_models import ChildProfile, ScreeningAssessment, ReminderNotification

log = logging.getLogger(__name__)


class ReminderService:
    @staticmethod
    def check_and_trigger_30_day_reminders(db: Session) -> List[ReminderNotification]:
        """
        Evaluates all child profiles to detect children whose last screening assessment
        occurred 30 or more days ago. Creates pending reminder notifications for parents.
        """
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)
        new_notifications: List[ReminderNotification] = []

        children: List[ChildProfile] = db.query(ChildProfile).all()

        for child in children:
            latest_assessment = (
                db.query(ScreeningAssessment)
                .filter(ScreeningAssessment.child_id == child.id)
                .order_by(ScreeningAssessment.completed_at.desc())
                .first()
            )

            if not latest_assessment:
                continue

            # Ensure completed_at is timezone-aware for comparison
            c_at = latest_assessment.completed_at
            if c_at.tzinfo is None:
                c_at = c_at.replace(tzinfo=timezone.utc)

            if c_at <= thirty_days_ago:
                # Check if an active reminder already exists for this 30-day cycle
                existing_reminder = (
                    db.query(ReminderNotification)
                    .filter(
                        ReminderNotification.child_id == child.id,
                        ReminderNotification.notification_type == "30_day_reassessment",
                        ReminderNotification.status == "pending"
                    )
                    .first()
                )

                if not existing_reminder:
                    due_date = (c_at + timedelta(days=30)).date()
                    reminder = ReminderNotification(
                        child_id=child.id,
                        user_id=child.user_id,
                        notification_type="30_day_reassessment",
                        status="pending",
                        message=(
                            f"{child.first_name} is due for a 30-day developmental reassessment. "
                            f"Update your screening answers to track developmental trajectory and milestone shifts."
                        ),
                        due_date=due_date,
                        sent_at=now
                    )
                    db.add(reminder)
                    new_notifications.append(reminder)
                    log.info(
                        f"⏰ Created 30-day reassessment reminder for child '{child.first_name}' "
                        f"(Parent User ID: {child.user_id}, Due: {due_date})"
                    )

        if new_notifications:
            db.commit()
            for r in new_notifications:
                db.refresh(r)

        return new_notifications

    @staticmethod
    def get_user_reminders(
        db: Session,
        user_id: int,
        status: Optional[str] = "pending"
    ) -> List[ReminderNotification]:
        query = db.query(ReminderNotification).filter(ReminderNotification.user_id == user_id)
        if status:
            query = query.filter(ReminderNotification.status == status)
        return query.order_by(ReminderNotification.created_at.desc()).all()

    @staticmethod
    def dismiss_reminder(db: Session, reminder_id: UUID, user_id: int) -> bool:
        reminder = (
            db.query(ReminderNotification)
            .filter(
                ReminderNotification.id == reminder_id,
                ReminderNotification.user_id == user_id
            )
            .first()
        )
        if not reminder:
            return False

        reminder.status = "dismissed"
        db.commit()
        return True
