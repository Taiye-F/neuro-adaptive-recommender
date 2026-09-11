# routers/children_router.py
from datetime import datetime, timezone
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.auth_models import User
from models.domain_models import ChildProfile
from schemas.domain_schemas import ChildCreate, ChildUpdate, ChildResponse
from services.auth_service import get_current_user

children_router = APIRouter(prefix="/api/v1/children", tags=["Children Profiles"])


def get_authorized_child(child_id: UUID, current_user: User, db: Session) -> ChildProfile:
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Child profile not found.")
    # Clinicians can view if assigned or authorized; Parents can view their own children
    if current_user.role == "parent" and child.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this child profile.")
    return child


def compute_age_months(dob) -> int:
    today = datetime.now(timezone.utc).date()
    days = (today - dob).days
    return max(1, int(days / 30.44))


@children_router.get("", response_model=List[ChildResponse])
def list_children(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieve all child profiles belonging to current user."""
    if current_user.role == "clinician":
        children = db.query(ChildProfile).all()
    else:
        children = db.query(ChildProfile).filter(ChildProfile.user_id == current_user.id).all()

    results = []
    for c in children:
        results.append(ChildResponse(
            id=c.id,
            user_id=c.user_id,
            first_name=c.first_name,
            date_of_birth=c.date_of_birth,
            biological_sex=c.biological_sex,
            age_months=compute_age_months(c.date_of_birth),
            total_assessments=len(c.assessments),
            created_at=c.created_at,
            updated_at=c.updated_at
        ))
    return results


@children_router.post("", response_model=ChildResponse, status_code=status.HTTP_201_CREATED)
def create_child(child_in: ChildCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Register a new child profile."""
    child = ChildProfile(
        user_id=current_user.id,
        first_name=child_in.first_name.strip(),
        date_of_birth=child_in.date_of_birth,
        biological_sex=child_in.biological_sex
    )
    db.add(child)
    db.commit()
    db.refresh(child)

    return ChildResponse(
        id=child.id,
        user_id=child.user_id,
        first_name=child.first_name,
        date_of_birth=child.date_of_birth,
        biological_sex=child.biological_sex,
        age_months=compute_age_months(child.date_of_birth),
        total_assessments=0,
        created_at=child.created_at,
        updated_at=child.updated_at
    )


@children_router.get("/{child_id}", response_model=ChildResponse)
def get_child(child_id: UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get single child profile by UUID."""
    child = get_authorized_child(child_id, current_user, db)
    return ChildResponse(
        id=child.id,
        user_id=child.user_id,
        first_name=child.first_name,
        date_of_birth=child.date_of_birth,
        biological_sex=child.biological_sex,
        age_months=compute_age_months(child.date_of_birth),
        total_assessments=len(child.assessments),
        created_at=child.created_at,
        updated_at=child.updated_at
    )


@children_router.put("/{child_id}", response_model=ChildResponse)
def update_child(child_id: UUID, child_update: ChildUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update child demographic information."""
    child = get_authorized_child(child_id, current_user, db)
    if child_update.first_name is not None:
        child.first_name = child_update.first_name.strip()
    if child_update.date_of_birth is not None:
        child.date_of_birth = child_update.date_of_birth
    if child_update.biological_sex is not None:
        child.biological_sex = child_update.biological_sex

    child.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(child)

    return ChildResponse(
        id=child.id,
        user_id=child.user_id,
        first_name=child.first_name,
        date_of_birth=child.date_of_birth,
        biological_sex=child.biological_sex,
        age_months=compute_age_months(child.date_of_birth),
        total_assessments=len(child.assessments),
        created_at=child.created_at,
        updated_at=child.updated_at
    )


@children_router.delete("/{child_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_child(child_id: UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a child profile and all associated records."""
    child = get_authorized_child(child_id, current_user, db)
    db.delete(child)
    db.commit()
    return None
