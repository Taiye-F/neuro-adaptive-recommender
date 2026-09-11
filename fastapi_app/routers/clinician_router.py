# routers/clinician_router.py
from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.auth_models import User
from models.domain_models import ChildProfile
from schemas.domain_schemas import (
    PatientRosterItem,
    PatientAssignmentCreate,
    ClinicalNoteCreate,
    ClinicalNoteResponse,
    TrajectoryResponse,
    MeltdownAnalyticsResponse,
)
from services.auth_service import get_current_user, RoleChecker
from services.clinician_service import ClinicianService
from services.trajectory_service import TrajectoryService
from services.meltdown_service import MeltdownService
from services.schedule_service import ScheduleService

clinician_router = APIRouter(prefix="/api/v1/clinician", tags=["Clinician Portal & Patient Caseload"])

require_clinician = RoleChecker(["clinician"])


@clinician_router.get(
    "/patients",
    response_model=List[PatientRosterItem],
    dependencies=[Depends(require_clinician)]
)
def get_caseload_roster(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve the clinical caseload roster with triage priority classification."""
    return ClinicianService.get_clinician_roster(db, current_user.id)


@clinician_router.post(
    "/patients/assign",
    dependencies=[Depends(require_clinician)],
    status_code=status.HTTP_201_CREATED
)
def assign_patient_to_caseload(
    assignment_in: PatientAssignmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Assign a child patient to the authenticated clinician's caseload."""
    child = db.query(ChildProfile).filter(ChildProfile.id == assignment_in.child_id).first()
    if not child:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Child patient profile not found.")

    assignment = ClinicianService.assign_patient(
        db, current_user.id, child.id, access_level=assignment_in.access_level
    )
    return {
        "status": "success",
        "message": f"Patient '{child.first_name}' assigned to clinician caseload.",
        "assignment_id": assignment.id
    }


@clinician_router.get(
    "/patients/{child_id}",
    dependencies=[Depends(require_clinician)]
)
def get_patient_360_overview(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """360-degree longitudinal profile: Trajectory, Meltdowns, Schedules & Notes."""
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Child patient profile not found.")

    parent = db.query(User).filter(User.id == child.user_id).first()
    trajectory = TrajectoryService.get_trajectory(db, child)
    meltdown_analytics = MeltdownService.compute_environmental_patterns(db, child)
    schedules = ScheduleService.get_child_schedules(db, child.id)
    notes = ClinicianService.get_patient_clinical_notes(db, child.id)

    return {
        "child": {
            "id": child.id,
            "first_name": child.first_name,
            "date_of_birth": child.date_of_birth,
            "biological_sex": child.biological_sex,
            "created_at": child.created_at,
            "parent_username": parent.username if parent else "Unknown",
            "parent_email": parent.email if parent else "Unknown",
        },
        "trajectory": trajectory,
        "meltdown_analytics": meltdown_analytics,
        "schedules": schedules,
        "clinical_notes": notes
    }


@clinician_router.post(
    "/patients/{child_id}/notes",
    response_model=ClinicalNoteResponse,
    dependencies=[Depends(require_clinician)],
    status_code=status.HTTP_201_CREATED
)
def add_patient_clinical_note(
    child_id: UUID,
    note_in: ClinicalNoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a consultation note or intervention endorsement to a patient record."""
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Child patient profile not found.")

    note = ClinicianService.add_clinical_note(
        db, current_user, child.id, note_type=note_in.note_type, content=note_in.content
    )
    return ClinicalNoteResponse(
        id=note.id,
        child_id=note.child_id,
        clinician_id=note.clinician_id,
        clinician_username=current_user.username,
        note_type=note.note_type,
        content=note.content,
        created_at=note.created_at
    )


@clinician_router.get(
    "/patients/{child_id}/notes",
    response_model=List[ClinicalNoteResponse],
    dependencies=[Depends(require_clinician)]
)
def get_patient_clinical_notes(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve clinical consultation notes for a patient."""
    return ClinicianService.get_patient_clinical_notes(db, child_id)
