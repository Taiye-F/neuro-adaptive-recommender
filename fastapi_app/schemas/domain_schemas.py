# domain_schemas.py
from datetime import datetime, date, time as time_type
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────────────────────
# CHILD PROFILES
# ─────────────────────────────────────────────────────────────

class ChildCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50, description="Child first name or nickname")
    date_of_birth: date = Field(..., description="Child date of birth (YYYY-MM-DD)")
    biological_sex: int = Field(..., ge=0, le=1, description="1 = Male, 0 = Female")


class ChildUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    date_of_birth: Optional[date] = None
    biological_sex: Optional[int] = Field(None, ge=0, le=1)


class ChildResponse(BaseModel):
    id: UUID
    user_id: int
    first_name: str
    date_of_birth: date
    biological_sex: int
    age_months: int
    total_assessments: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────────
# TRAJECTORY & LONGITUDINAL JOURNEY
# ─────────────────────────────────────────────────────────────

class MilestoneShiftItem(BaseModel):
    code: str
    label: str
    previous_score: Optional[int] = None
    current_score: int
    previous_flagged: Optional[bool] = None
    current_flagged: bool
    status: str  # "Resolved", "Ongoing Focus Area", "New Flag Identified", "Stable Typical"


class TrajectoryTimelineItem(BaseModel):
    assessment_id: UUID
    completed_at: datetime
    age_months: int
    risk_probability: float
    is_high_risk: bool
    total_flags: int
    flagged_milestones: List[str]
    top_recommended_app: Optional[str] = None
    top_recommended_book: Optional[str] = None
    profile_explained: Optional[str] = None


class TrajectoryResponse(BaseModel):
    child_id: UUID
    child_name: str
    total_assessments: int
    trend_direction: str  # "improving", "concerning", "stable", "insufficient_data"
    baseline_risk: Optional[float] = None
    current_risk: Optional[float] = None
    risk_delta: Optional[float] = None
    assessments_timeline: List[TrajectoryTimelineItem]
    milestone_shifts: List[MilestoneShiftItem]
    summary_narrative: str


# ─────────────────────────────────────────────────────────────
# VISUAL SCHEDULES & TASKS
# ─────────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    icon_key: str = Field(default="star", max_length=50)
    scheduled_time: Optional[time_type] = None
    sequence_order: Optional[int] = None


class TaskResponse(BaseModel):
    id: UUID
    schedule_id: UUID
    title: str
    icon_key: str
    scheduled_time: Optional[time_type] = None
    sequence_order: int
    completed_today: bool = False

    model_config = ConfigDict(from_attributes=True)


class ScheduleCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    schedule_type: str = Field(default="daily", max_length=50)


class ScheduleResponse(BaseModel):
    id: UUID
    child_id: UUID
    title: str
    schedule_type: str
    is_active: bool
    tasks: List[TaskResponse]
    completed_tasks_count: int = 0
    total_tasks_count: int = 0
    streak_days: int = 0

    model_config = ConfigDict(from_attributes=True)


class TaskReorderItem(BaseModel):
    task_id: UUID
    sequence_order: int


class TaskReorderRequest(BaseModel):
    tasks: List[TaskReorderItem]


class TaskCompletionResponse(BaseModel):
    task_id: UUID
    completed_today: bool
    streak_days: int
    reward_animation: str
    message: str


# ─────────────────────────────────────────────────────────────
# PHASE 3: MELTDOWN CRISIS & REMINDER SCHEMAS
# ─────────────────────────────────────────────────────────────

class StrategyAppliedItem(BaseModel):
    strategy_name: str
    strategy_category: str = "general"
    efficacy: str = "effective"  # effective, partially_effective, ineffective
    notes: Optional[str] = None


class StrategyAppliedResponse(StrategyAppliedItem):
    id: UUID
    meltdown_id: UUID

    model_config = ConfigDict(from_attributes=True)


class MeltdownTriageCreate(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: int = Field(default=0, ge=0)
    intensity: int = Field(default=2, ge=1, le=5)
    location: str = Field(default="home", max_length=50)
    triggers: List[str] = Field(default_factory=list)
    strategies: List[StrategyAppliedItem] = Field(default_factory=list)
    notes: Optional[str] = None


class MeltdownIncidentResponse(BaseModel):
    id: UUID
    child_id: UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int
    intensity: int
    location: str
    triggers: List[str]
    notes: Optional[str] = None
    created_at: datetime
    strategies_applied: List[StrategyAppliedResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class MeltdownAnalyticsResponse(BaseModel):
    child_id: UUID
    child_name: str
    total_incidents: int
    avg_duration_minutes: float
    avg_intensity: float
    trigger_frequencies: Dict[str, int]
    location_frequencies: Dict[str, int]
    time_of_day_distribution: Dict[str, int]
    day_of_week_distribution: Dict[str, int]
    strategy_efficacy_ranking: List[Dict[str, Any]]
    plain_english_insights: List[str]


class ReminderNotificationResponse(BaseModel):
    id: UUID
    child_id: UUID
    user_id: int
    notification_type: str
    status: str
    message: str
    due_date: date
    sent_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────────
# PHASE 4: CLINICIAN PORTAL & RAG SCHEMAS
# ─────────────────────────────────────────────────────────────

class ClinicalNoteCreate(BaseModel):
    note_type: str = Field(default="consultation", max_length=50)  # consultation, recommendation_endorsement, diagnostic_followup
    content: str = Field(..., min_length=1)


class ClinicalNoteResponse(BaseModel):
    id: UUID
    child_id: UUID
    clinician_id: int
    clinician_username: Optional[str] = None
    note_type: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PatientAssignmentCreate(BaseModel):
    child_id: UUID
    access_level: str = Field(default="full_clinical", max_length=30)


class PatientRosterItem(BaseModel):
    child_id: UUID
    child_name: str
    age_months: int
    biological_sex: int
    parent_name: str
    latest_risk: Optional[float] = None
    risk_level: str = "Low"
    trajectory_delta: Optional[float] = None
    trajectory_trend: str = "insufficient_data"
    last_screening_date: Optional[datetime] = None
    days_since_last_screening: Optional[int] = None
    total_meltdowns: int = 0
    triage_priority: str = "stable"  # urgent, monitor, stable
    triage_reasons: List[str] = Field(default_factory=list)


class EvidenceSearchResponse(BaseModel):
    query: str
    total_found: int
    results: List[Dict[str, Any]]
