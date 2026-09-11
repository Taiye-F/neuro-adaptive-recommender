# clinician_service.py
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from models.auth_models import User
from models.domain_models import (
    ChildProfile, ScreeningAssessment, MeltdownIncident,
    ClinicianPatientAssignment, ClinicalNote
)
from schemas.domain_schemas import PatientRosterItem, ClinicalNoteResponse
from services.trajectory_service import TrajectoryService


class ClinicianService:
    @staticmethod
    def get_clinician_roster(db: Session, clinician_id: int) -> List[PatientRosterItem]:
        """
        Compiles the patient caseload for a clinician, computing clinical triage flags,
        trajectory velocity, and meltdown volatility for each child.
        """
        assignments = (
            db.query(ClinicianPatientAssignment)
            .filter(
                ClinicianPatientAssignment.clinician_id == clinician_id,
                ClinicianPatientAssignment.status == "active"
            )
            .all()
        )

        # If clinician has no specific assignments, fall back to accessible child profiles
        assigned_child_ids = [a.child_id for a in assignments]
        if assigned_child_ids:
            children = db.query(ChildProfile).filter(ChildProfile.id.in_(assigned_child_ids)).all()
        else:
            children = db.query(ChildProfile).limit(20).all()

        now = datetime.now(timezone.utc)
        roster: List[PatientRosterItem] = []

        for child in children:
            parent = db.query(User).filter(User.id == child.user_id).first()
            parent_name = parent.username if parent else "Unknown Parent"

            # Compute approximate age in months
            age_months = max(12, int((now.date() - child.date_of_birth).days / 30.44))

            # Latest screening assessment
            latest_assessment = (
                db.query(ScreeningAssessment)
                .filter(ScreeningAssessment.child_id == child.id)
                .order_by(ScreeningAssessment.completed_at.desc())
                .first()
            )

            latest_risk = latest_assessment.risk_probability if latest_assessment else None
            last_date = latest_assessment.completed_at if latest_assessment else None
            if last_date:
                if last_date.tzinfo is None:
                    last_date = last_date.replace(tzinfo=timezone.utc)
                days_since = (now - last_date).days
            else:
                days_since = None

            # Trajectory calculation
            trajectory = TrajectoryService.get_trajectory(db, child)
            delta = trajectory.risk_delta
            trend = trajectory.trend_direction

            # Meltdowns count
            meltdowns_count = (
                db.query(MeltdownIncident)
                .filter(MeltdownIncident.child_id == child.id)
                .count()
            )

            # Triage Priority Rules
            triage_reasons = []
            if latest_risk is not None and latest_risk >= 70.0:
                triage_reasons.append(f"High ASD Probability ({latest_risk:.1f}%)")
            if trend == "concerning":
                triage_reasons.append(f"Adverse Trajectory Shift (+{delta}%)")
            if days_since is not None and days_since >= 30:
                triage_reasons.append(f"Screening Overdue ({days_since}d elapsed)")
            if meltdowns_count >= 3:
                triage_reasons.append(f"Sensory Volatility ({meltdowns_count} meltdowns)")

            if triage_reasons:
                priority = "urgent"
            elif latest_risk is not None and latest_risk >= 40.0:
                priority = "monitor"
                triage_reasons.append(f"Moderate Risk Threshold ({latest_risk:.1f}%)")
            elif meltdowns_count >= 1:
                priority = "monitor"
                triage_reasons.append("Reported Sensory Episode")
            else:
                priority = "stable"
                triage_reasons.append("Development Consistent")

            # Risk level label
            if latest_risk is not None:
                risk_lvl = "High Risk" if latest_risk >= 40.0 else "Low Risk"
            else:
                risk_lvl = "Unscreened"

            roster.append(PatientRosterItem(
                child_id=child.id,
                child_name=child.first_name,
                age_months=age_months,
                biological_sex=child.biological_sex,
                parent_name=parent_name,
                latest_risk=latest_risk,
                risk_level=risk_lvl,
                trajectory_delta=delta,
                trajectory_trend=trend,
                last_screening_date=last_date,
                days_since_last_screening=days_since,
                total_meltdowns=meltdowns_count,
                triage_priority=priority,
                triage_reasons=triage_reasons
            ))

        # Order by priority: urgent first, then monitor, then stable
        priority_order = {"urgent": 0, "monitor": 1, "stable": 2}
        roster.sort(key=lambda x: priority_order.get(x.triage_priority, 3))
        return roster

    @staticmethod
    def assign_patient(
        db: Session,
        clinician_id: int,
        child_id: UUID,
        access_level: str = "full_clinical"
    ) -> ClinicianPatientAssignment:
        assignment = (
            db.query(ClinicianPatientAssignment)
            .filter(
                ClinicianPatientAssignment.clinician_id == clinician_id,
                ClinicianPatientAssignment.child_id == child_id
            )
            .first()
        )
        if assignment:
            assignment.status = "active"
            assignment.access_level = access_level
        else:
            assignment = ClinicianPatientAssignment(
                clinician_id=clinician_id,
                child_id=child_id,
                access_level=access_level,
                status="active"
            )
            db.add(assignment)

        db.commit()
        db.refresh(assignment)
        return assignment

    @staticmethod
    def add_clinical_note(
        db: Session,
        clinician: User,
        child_id: UUID,
        note_type: str,
        content: str
    ) -> ClinicalNote:
        note = ClinicalNote(
            child_id=child_id,
            clinician_id=clinician.id,
            note_type=note_type,
            content=content.strip()
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note

    @staticmethod
    def get_patient_clinical_notes(db: Session, child_id: UUID) -> List[ClinicalNoteResponse]:
        notes = (
            db.query(ClinicalNote)
            .filter(ClinicalNote.child_id == child_id)
            .order_by(ClinicalNote.created_at.desc())
            .all()
        )
        results = []
        for n in notes:
            clinician = db.query(User).filter(User.id == n.clinician_id).first()
            results.append(ClinicalNoteResponse(
                id=n.id,
                child_id=n.child_id,
                clinician_id=n.clinician_id,
                clinician_username=clinician.username if clinician else "Unknown",
                note_type=n.note_type,
                content=n.content,
                created_at=n.created_at
            ))
        return results
