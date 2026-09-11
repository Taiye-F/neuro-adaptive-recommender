# routers/journey_router.py
from uuid import UUID
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from database import get_db
from models.auth_models import User
from models.domain_models import ChildProfile, ScreeningAssessment
from schemas.domain_schemas import TrajectoryResponse
from services.auth_service import get_current_user
from services.trajectory_service import TrajectoryService
from services.pdf_service import PDFService
from routers.children_router import get_authorized_child

journey_router = APIRouter(prefix="/api/v1/journey", tags=["Journey Loop & Trajectory"])


@journey_router.get("/{child_id}/trajectory", response_model=TrajectoryResponse)
def get_child_trajectory(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve trajectory delta and milestone transitions for a child."""
    child = get_authorized_child(child_id, current_user, db)
    return TrajectoryService.get_trajectory(db, child)


@journey_router.get("/{child_id}/report.pdf")
def download_clinical_pdf(
    child_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate and stream a clinical pediatric PDF assessment summary."""
    child = get_authorized_child(child_id, current_user, db)
    trajectory = TrajectoryService.get_trajectory(db, child)

    # Fetch all assessments with recommendations
    assessments = (
        db.query(ScreeningAssessment)
        .filter(ScreeningAssessment.child_id == child.id)
        .order_by(ScreeningAssessment.completed_at.desc())
        .all()
    )

    context = {
        "child": child,
        "parent_user": current_user,
        "trajectory": trajectory,
        "assessments": assessments,
        "latest_assessment": assessments[0] if assessments else None,
        "generated_at": datetime.now(timezone.utc).strftime("%B %d, %Y"),
    }

    try:
        pdf_bytes = PDFService.generate_clinical_report_pdf(context)
        filename = f"NALR_Clinical_Report_{child.first_name.replace(' ', '_')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate clinical PDF: {str(e)}"
        )
