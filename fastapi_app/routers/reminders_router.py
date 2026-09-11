# routers/reminders_router.py
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.auth_models import User
from schemas.domain_schemas import ReminderNotificationResponse
from services.auth_service import get_current_user
from services.reminder_service import ReminderService

reminders_router = APIRouter(prefix="/api/v1/reminders", tags=["Automated Reminders & Notifications"])


@reminders_router.get("", response_model=List[ReminderNotificationResponse])
def get_user_reminders(
    status: Optional[str] = "pending",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve pending or historical notification reminders for the authenticated parent."""
    return ReminderService.get_user_reminders(db, current_user.id, status=status)


@reminders_router.post("/{reminder_id}/dismiss")
def dismiss_reminder(
    reminder_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dismiss an active reassessment reminder notification."""
    success = ReminderService.dismiss_reminder(db, reminder_id, current_user.id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reminder not found.")
    return {"status": "success", "message": "Reminder dismissed."}


@reminders_router.post("/check-now")
def trigger_manual_reminder_check(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually trigger the 30-day reassessment evaluation job."""
    created = ReminderService.check_and_trigger_30_day_reminders(db)
    return {
        "status": "success",
        "message": f"Evaluated 30-day screening reminders. Generated {len(created)} new notification(s).",
        "new_reminders_count": len(created)
    }
