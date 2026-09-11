# schedule_service.py
from datetime import datetime, timezone, date, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func
from models.domain_models import VisualSchedule, ScheduleTask, TaskCompletion, ChildProfile
from schemas.domain_schemas import ScheduleResponse, TaskResponse, TaskCompletionResponse


DEFAULT_TASKS = [
    {"title": "Brush Teeth", "icon_key": "brush_teeth"},
    {"title": "Wash Face", "icon_key": "wash_face"},
    {"title": "Get Dressed", "icon_key": "get_dressed"},
    {"title": "Eat Breakfast", "icon_key": "breakfast"},
    {"title": "Put on Shoes", "icon_key": "shoes"},
]


class ScheduleService:
    @staticmethod
    def get_or_create_default_schedule(db: Session, child_id: UUID) -> VisualSchedule:
        schedule = (
            db.query(VisualSchedule)
            .filter(VisualSchedule.child_id == child_id)
            .order_by(VisualSchedule.created_at.asc())
            .first()
        )
        if not schedule:
            schedule = VisualSchedule(
                child_id=child_id,
                title="Daily Morning Routine",
                schedule_type="daily",
                is_active=True
            )
            db.add(schedule)
            db.flush()

            for i, item in enumerate(DEFAULT_TASKS, 1):
                task = ScheduleTask(
                    schedule_id=schedule.id,
                    title=item["title"],
                    icon_key=item["icon_key"],
                    sequence_order=i
                )
                db.add(task)
            db.commit()
            db.refresh(schedule)

        return schedule

    @staticmethod
    def calculate_streak(db: Session, child_id: UUID) -> int:
        """Calculates consecutive active days with task completions."""
        # Get all distinct completion dates for this child's tasks
        dates = (
            db.query(TaskCompletion.completion_date)
            .join(ScheduleTask, TaskCompletion.task_id == ScheduleTask.id)
            .join(VisualSchedule, ScheduleTask.schedule_id == VisualSchedule.id)
            .filter(VisualSchedule.child_id == child_id)
            .distinct()
            .order_by(TaskCompletion.completion_date.desc())
            .all()
        )
        if not dates:
            return 0

        date_set = {d[0] for d in dates}
        today = datetime.now(timezone.utc).date()
        
        # Streak can be active if completed today or yesterday
        current_check = today
        if current_check not in date_set:
            current_check = today - timedelta(days=1)
            if current_check not in date_set:
                return 0

        streak = 0
        while current_check in date_set:
            streak += 1
            current_check -= timedelta(days=1)

        return streak

    @staticmethod
    def get_child_schedules(db: Session, child_id: UUID) -> List[ScheduleResponse]:
        # Ensure default schedule exists
        ScheduleService.get_or_create_default_schedule(db, child_id)

        schedules = (
            db.query(VisualSchedule)
            .filter(VisualSchedule.child_id == child_id, VisualSchedule.is_active == True)
            .order_by(VisualSchedule.created_at.asc())
            .all()
        )

        today = datetime.now(timezone.utc).date()
        streak = ScheduleService.calculate_streak(db, child_id)
        result: List[ScheduleResponse] = []

        for sched in schedules:
            task_responses: List[TaskResponse] = []
            completed_count = 0
            for t in sched.tasks:
                is_done = (
                    db.query(TaskCompletion)
                    .filter(
                        TaskCompletion.task_id == t.id,
                        TaskCompletion.completion_date == today
                    )
                    .first()
                    is not None
                )
                if is_done:
                    completed_count += 1
                task_responses.append(TaskResponse(
                    id=t.id,
                    schedule_id=t.schedule_id,
                    title=t.title,
                    icon_key=t.icon_key,
                    scheduled_time=t.scheduled_time,
                    sequence_order=t.sequence_order,
                    completed_today=is_done
                ))

            result.append(ScheduleResponse(
                id=sched.id,
                child_id=sched.child_id,
                title=sched.title,
                schedule_type=sched.schedule_type,
                is_active=sched.is_active,
                tasks=task_responses,
                completed_tasks_count=completed_count,
                total_tasks_count=len(sched.tasks),
                streak_days=streak
            ))

        return result

    @staticmethod
    def toggle_task(db: Session, task_id: UUID) -> TaskCompletionResponse:
        today = datetime.now(timezone.utc).date()
        existing = (
            db.query(TaskCompletion)
            .filter(
                TaskCompletion.task_id == task_id,
                TaskCompletion.completion_date == today
            )
            .first()
        )

        task = db.query(ScheduleTask).filter(ScheduleTask.id == task_id).first()
        if not task:
            raise ValueError("Task not found.")

        schedule = db.query(VisualSchedule).filter(VisualSchedule.id == task.schedule_id).first()

        if existing:
            db.delete(existing)
            db.commit()
            streak = ScheduleService.calculate_streak(db, schedule.child_id) if schedule else 0
            return TaskCompletionResponse(
                task_id=task_id,
                completed_today=False,
                streak_days=streak,
                reward_animation="none",
                message=f"Task '{task.title}' marked as incomplete."
            )
        else:
            completion = TaskCompletion(
                task_id=task_id,
                completion_date=today,
                completed_at=datetime.now(timezone.utc),
                rewarded=True
            )
            db.add(completion)
            db.commit()
            streak = ScheduleService.calculate_streak(db, schedule.child_id) if schedule else 1
            return TaskCompletionResponse(
                task_id=task_id,
                completed_today=True,
                streak_days=streak,
                reward_animation="stars_confetti",
                message=f"Great job! Task '{task.title}' completed!"
            )

    @staticmethod
    def reorder_tasks(db: Session, schedule_id: UUID, task_orders: List[Dict[str, Any]]) -> None:
        for item in task_orders:
            t_id = item.get("task_id")
            seq = item.get("sequence_order")
            if t_id and seq is not None:
                db.query(ScheduleTask).filter(
                    ScheduleTask.id == t_id,
                    ScheduleTask.schedule_id == schedule_id
                ).update({ScheduleTask.sequence_order: seq})
        db.commit()
