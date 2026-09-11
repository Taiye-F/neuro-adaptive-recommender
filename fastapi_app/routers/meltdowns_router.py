# routers/meltdowns_router.py
from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.auth_models import User
from schemas.domain_schemas import (
    MeltdownTriageCreate,
    MeltdownIncidentResponse,
    MeltdownAnalyticsResponse,
)
from services.auth_service import get_current_user
from services.meltdown_service import MeltdownService
from routers.children_router import get_authorized_child

meltdowns_router = APIRouter(prefix="/api/v1/meltdowns", tags=["Meltdown Emergency Suite & Analytics"])


@meltdowns_router.post(
    "/{child_id}/triage",
    response_model=MeltdownIncidentResponse,
    status_code=status.HTTP_201_CREATED
)
def submit_meltdown_triage(
    child_id: UUID,
    triage_in: MeltdownTriageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit post-incident triage intake with duration, triggers, location, and strategy efficacy."""
    child = get_authorized_child(child_id, current_user, db)
    incident = MeltdownService.log_meltdown_triage(db, child.id, triage_in)
    return incident


@meltdowns_router.get(
    "/{child_id}",
    response_model=List[MeltdownIncidentResponse]
)
def list_meltdown_incidents(
    child_id: UUID,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve past sensory meltdown logs for a child."""
    child = get_authorized_child(child_id, current_user, db)
    return MeltdownService.get_child_incidents(db, child.id, limit=limit)


@meltdowns_router.get(
    "/{child_id}/analytics",
    response_model=MeltdownAnalyticsResponse
)
def get_meltdown_analytics(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compute environmental trigger patterns, temporal distributions, strategy rankings, and plain-English caregiver insights."""
    child = get_authorized_child(child_id, current_user, db)
    return MeltdownService.compute_environmental_patterns(db, child)


@meltdowns_router.get(
    "/{child_id}/strategies",
    response_model=List[Dict[str, Any]]
)
def get_quick_strategies(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve quick calming sensory strategies tailored for active emergency response."""
    child = get_authorized_child(child_id, current_user, db)
    return MeltdownService.get_quick_calming_strategies(db, child.id)
