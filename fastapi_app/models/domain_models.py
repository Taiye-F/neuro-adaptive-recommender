# domain_models.py
import uuid
from datetime import datetime, timezone, date, time
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, DateTime, Date, Time,
    ForeignKey, Text, Uuid, JSON
)
from sqlalchemy.orm import relationship
from database import Base


class ChildProfile(Base):
    __tablename__ = "child_profiles"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    first_name = Column(String(50), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    biological_sex = Column(Integer, nullable=False)  # 1 = Male, 0 = Female
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="children")
    assessments = relationship(
        "ScreeningAssessment",
        back_populates="child",
        cascade="all, delete-orphan",
        order_by="ScreeningAssessment.completed_at.desc()"
    )
    schedules = relationship(
        "VisualSchedule",
        back_populates="child",
        cascade="all, delete-orphan",
        order_by="VisualSchedule.created_at.desc()"
    )
    meltdowns = relationship(
        "MeltdownIncident",
        back_populates="child",
        cascade="all, delete-orphan",
        order_by="MeltdownIncident.start_time.desc()"
    )
    reminders = relationship(
        "ReminderNotification",
        back_populates="child",
        cascade="all, delete-orphan",
        order_by="ReminderNotification.created_at.desc()"
    )
    clinician_assignments = relationship(
        "ClinicianPatientAssignment",
        back_populates="child",
        cascade="all, delete-orphan"
    )
    clinical_notes = relationship(
        "ClinicalNote",
        back_populates="child",
        cascade="all, delete-orphan",
        order_by="ClinicalNote.created_at.desc()"
    )


class ScreeningAssessment(Base):
    __tablename__ = "screening_assessments"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    child_id = Column(Uuid(as_uuid=True), ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    age_months = Column(Integer, nullable=False)

    # Q-CHAT-10 Scores (0-4 Ordinal Recodes)
    a1_score = Column(Integer, nullable=False)
    a2_score = Column(Integer, nullable=False)
    a3_score = Column(Integer, nullable=False)
    a4_score = Column(Integer, nullable=False)
    a5_score = Column(Integer, nullable=False)
    a6_score = Column(Integer, nullable=False)
    a7_score = Column(Integer, nullable=False)
    a8_score = Column(Integer, nullable=False)
    a9_score = Column(Integer, nullable=False)
    a10_score = Column(Integer, nullable=False)

    risk_probability = Column(Float, nullable=False)
    is_high_risk = Column(Boolean, nullable=False)
    total_flags = Column(Integer, nullable=False)
    profile_text = Column(Text, nullable=False)
    profile_explained = Column(Text, nullable=True)
    completed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    child = relationship("ChildProfile", back_populates="assessments")
    recommendations = relationship(
        "AssessmentRecommendation",
        back_populates="assessment",
        cascade="all, delete-orphan",
        order_by="AssessmentRecommendation.rank.asc()"
    )


class AssessmentRecommendation(Base):
    __tablename__ = "assessment_recommendations"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    assessment_id = Column(Uuid(as_uuid=True), ForeignKey("screening_assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    resource_type = Column(String(20), nullable=False)  # 'app' or 'book'
    item_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)
    match_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)

    # Relationships
    assessment = relationship("ScreeningAssessment", back_populates="recommendations")


class VisualSchedule(Base):
    __tablename__ = "visual_schedules"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    child_id = Column(Uuid(as_uuid=True), ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(100), nullable=False)  # e.g. "Morning Routine", "Bedtime"
    schedule_type = Column(String(50), default="daily", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    child = relationship("ChildProfile", back_populates="schedules")
    tasks = relationship(
        "ScheduleTask",
        back_populates="schedule",
        cascade="all, delete-orphan",
        order_by="ScheduleTask.sequence_order.asc()"
    )


class ScheduleTask(Base):
    __tablename__ = "schedule_tasks"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    schedule_id = Column(Uuid(as_uuid=True), ForeignKey("visual_schedules.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(100), nullable=False)
    icon_key = Column(String(50), nullable=False)  # e.g. "brush_teeth", "get_dressed", "breakfast"
    audio_prompt_url = Column(String(255), nullable=True)
    sequence_order = Column(Integer, nullable=False, default=1)
    scheduled_time = Column(Time, nullable=True)

    # Relationships
    schedule = relationship("VisualSchedule", back_populates="tasks")
    completions = relationship(
        "TaskCompletion",
        back_populates="task",
        cascade="all, delete-orphan"
    )


class TaskCompletion(Base):
    __tablename__ = "task_completions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    task_id = Column(Uuid(as_uuid=True), ForeignKey("schedule_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    completion_date = Column(Date, default=lambda: datetime.now(timezone.utc).date(), nullable=False, index=True)
    completed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    rewarded = Column(Boolean, default=True, nullable=False)

    # Relationships
    task = relationship("ScheduleTask", back_populates="completions")


class MeltdownIncident(Base):
    __tablename__ = "meltdown_incidents"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    child_id = Column(Uuid(as_uuid=True), ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=False, default=0)
    intensity = Column(Integer, nullable=False, default=2)  # 1=Mild, 2=Moderate, 3=Active Distress, 4=Severe, 5=Extreme Crisis
    location = Column(String(50), nullable=False, default="home")  # home, school, supermarket, transit, social, other
    triggers = Column(JSON, nullable=False, default=list)  # list of trigger strings: ['noise', 'transition', ...]
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    child = relationship("ChildProfile", back_populates="meltdowns")
    strategies_applied = relationship(
        "MeltdownStrategyApplied",
        back_populates="meltdown",
        cascade="all, delete-orphan"
    )


class MeltdownStrategyApplied(Base):
    __tablename__ = "meltdown_strategies_applied"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    meltdown_id = Column(Uuid(as_uuid=True), ForeignKey("meltdown_incidents.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False)
    strategy_category = Column(String(50), nullable=False, default="general")  # proprioceptive, auditory, visual, breathing, environmental
    efficacy = Column(String(30), nullable=False, default="effective")  # effective, partially_effective, ineffective
    notes = Column(String(255), nullable=True)

    # Relationships
    meltdown = relationship("MeltdownIncident", back_populates="strategies_applied")


class ReminderNotification(Base):
    __tablename__ = "reminder_notifications"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    child_id = Column(Uuid(as_uuid=True), ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    notification_type = Column(String(50), default="30_day_reassessment", nullable=False)
    status = Column(String(30), default="pending", nullable=False)  # pending, sent, dismissed
    message = Column(Text, nullable=False)
    due_date = Column(Date, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    child = relationship("ChildProfile", back_populates="reminders")
    user = relationship("User", back_populates="reminders")


class ClinicianPatientAssignment(Base):
    __tablename__ = "clinician_patient_assignments"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    clinician_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    child_id = Column(Uuid(as_uuid=True), ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    access_level = Column(String(30), default="full_clinical", nullable=False)  # read_only, full_clinical
    status = Column(String(30), default="active", nullable=False)  # active, revoked
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    clinician = relationship("User", back_populates="assigned_patients")
    child = relationship("ChildProfile", back_populates="clinician_assignments")


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    child_id = Column(Uuid(as_uuid=True), ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    clinician_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    note_type = Column(String(50), default="consultation", nullable=False)  # consultation, recommendation_endorsement, diagnostic_followup
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    child = relationship("ChildProfile", back_populates="clinical_notes")
    clinician = relationship("User", back_populates="clinical_notes")
