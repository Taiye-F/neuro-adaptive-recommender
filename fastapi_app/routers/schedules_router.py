# routers/schedules_router.py
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.auth_models import User
from models.domain_models import VisualSchedule, ScheduleTask
from schemas.domain_schemas import (
    ScheduleCreate, ScheduleResponse, TaskCreate, TaskResponse,
    TaskReorderRequest, TaskCompletionResponse
)
from services.auth_service import get_current_user
from services.schedule_service import ScheduleService
from routers.children_router import get_authorized_child

schedules_router = APIRouter(prefix="/api/v1/schedules", tags=["Visual Schedules"])


@schedules_router.get("/{child_id}", response_model=List[ScheduleResponse])
def get_child_schedules(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all visual schedules and today completion states for a child."""
    child = get_authorized_child(child_id, current_user, db)
    return ScheduleService.get_child_schedules(db, child.id)


@schedules_router.post("/{child_id}", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
def create_child_schedule(
    child_id: UUID,
    schedule_in: ScheduleCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new visual routine schedule for a child."""
    child = get_authorized_child(child_id, current_user, db)
    sched = VisualSchedule(
        child_id=child.id,
        title=schedule_in.title.strip(),
        schedule_type=schedule_in.schedule_type,
        is_active=True
    )
    db.add(sched)
    db.commit()
    db.refresh(sched)

    return ScheduleResponse(
        id=sched.id,
        child_id=sched.child_id,
        title=sched.title,
        schedule_type=sched.schedule_type,
        is_active=sched.is_active,
        tasks=[],
        completed_tasks_count=0,
        total_tasks_count=0,
        streak_days=ScheduleService.calculate_streak(db, child.id)
    )


@schedules_router.post("/{schedule_id}/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def add_schedule_task(
    schedule_id: UUID,
    task_in: TaskCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new visual routine task card to a schedule."""
    sched = db.query(VisualSchedule).filter(VisualSchedule.id == schedule_id).first()
    if not sched:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found.")
    get_authorized_child(sched.child_id, current_user, db)

    # Determine sequence order if not specified
    if task_in.sequence_order is None:
        max_seq = db.query(ScheduleTask).filter(ScheduleTask.schedule_id == schedule_id).count()
        seq = max_seq + 1
    else:
        seq = task_in.sequence_order

    task = ScheduleTask(
        schedule_id=schedule_id,
        title=task_in.title.strip(),
        icon_key=task_in.icon_key,
        scheduled_time=task_in.scheduled_time,
        sequence_order=seq
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    return TaskResponse(
        id=task.id,
        schedule_id=task.schedule_id,
        title=task.title,
        icon_key=task.icon_key,
        scheduled_time=task.scheduled_time,
        sequence_order=task.sequence_order,
        completed_today=False
    )


@schedules_router.put("/{schedule_id}/tasks/reorder", status_code=status.HTTP_200_OK)
def reorder_tasks(
    schedule_id: UUID,
    reorder_in: TaskReorderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update sequence ordering of tasks in a schedule."""
    sched = db.query(VisualSchedule).filter(VisualSchedule.id == schedule_id).first()
    if not sched:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found.")
    get_authorized_child(sched.child_id, current_user, db)

    ScheduleService.reorder_tasks(
        db,
        schedule_id,
        [{"task_id": item.task_id, "sequence_order": item.sequence_order} for item in reorder_in.tasks]
    )
    return {"status": "success", "message": "Tasks reordered successfully."}


@schedules_router.post("/tasks/{task_id}/complete", response_model=TaskCompletionResponse)
@schedules_router.post("/tasks/{task_id}/toggle", response_model=TaskCompletionResponse)
def toggle_task_completion(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle 1-tap check-off for a routine task today (triggers rewards & streaks)."""
    task = db.query(ScheduleTask).filter(ScheduleTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    
    sched = db.query(VisualSchedule).filter(VisualSchedule.id == task.schedule_id).first()
    if sched:
        get_authorized_child(sched.child_id, current_user, db)

    return ScheduleService.toggle_task(db, task_id)


@schedules_router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a task card from a schedule."""
    task = db.query(ScheduleTask).filter(ScheduleTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    sched = db.query(VisualSchedule).filter(VisualSchedule.id == task.schedule_id).first()
    if sched:
        get_authorized_child(sched.child_id, current_user, db)

    db.delete(task)
    db.commit()
    return None
