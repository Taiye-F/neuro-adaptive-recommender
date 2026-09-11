# main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from dotenv import load_dotenv
load_dotenv(override=True)

from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Form, Request, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
import time 
import traceback
import os
from typing import Optional, Union
from uuid import UUID
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from database import get_db, engine, Base
from services.auth_service import AuthService
from repositories.user_repository import UserRepository
import models
from models.auth_models import User
from models.domain_models import (
    ChildProfile, ScreeningAssessment, AssessmentRecommendation,
    VisualSchedule, ScheduleTask, TaskCompletion
)
from services.trajectory_service import TrajectoryService
from services.schedule_service import ScheduleService
from services.pdf_service import PDFService
from services.meltdown_service import MeltdownService
from services.reminder_service import ReminderService
from services.clinician_service import ClinicianService
from services.rag_evidence_service import EvidenceRAGService
from models.domain_models import ClinicalNote, ClinicianPatientAssignment
from workers.scheduler import start_scheduler, shutdown_scheduler
from core import (
    state, _load_model, _load_app_cache, _load_book_cache, _init_gemini, 
    predict_risk, recommend_apps, recommend_books,
    map_likert_standard, map_likert_reverse,
    build_profile_text, explain_profile, QUESTION_LABELS, log
)
from routers import api_router


def check_flag(k: str, val: str) -> bool:
    if k in ["A9", "A10"]:
        return map_likert_reverse(val) >= 3
    else:
        return map_likert_standard(val) >= 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up Neuro-Adaptive ASD Recommender...")

    state.startup_time = time.time()
    startup_success = True

    try:
        # Load .env only if it exists (for local development)
        load_dotenv(override=True)

        # Initialize database tables
        Base.metadata.create_all(bind=engine)
        log.info("✓ Database tables initialized successfully")

        _load_model()
        log.info("✓ Model loaded successfully")

        _load_app_cache()
        log.info(f"✓ App cache loaded — {len(state.df_apps)} apps")

        _load_book_cache()
        log.info(f"✓ Book cache loaded — {len(state.df_books)} books")

        _init_gemini()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            log.info("✓ GEMINI_API_KEY loaded successfully")
        else:
            log.warning("⚠️ GEMINI_API_KEY is missing — Chat feature will be disabled")

        # Start automated 30-day reassessment scheduler
        start_scheduler()

    except Exception as e:
        log.error(f"❌ Critical startup error: {e}")
        startup_success = False

    log.info(f"Startup completed in {time.time() - state.startup_time:.2f}s")
    log.info(f"Chat available: {state.gemini_client is not None}")
    log.info(f"Overall startup success: {startup_success}")

    yield
    shutdown_scheduler()
    log.info("Shutting down.")


# ─────────────────────────────
# FASTAPI APP
# ─────────────────────────────
app = FastAPI(
    title="Neuro-Adaptive ASD Learning Recommender",
    description="ASD early-screening microservice for toddlers with longitudinal tracking...",
    version="2.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("=== UNHANDLED EXCEPTION ===")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)}
    )


# Helper to retrieve current authenticated user from cookies (for Jinja UI routes)
async def get_current_user_from_cookie(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        token_data = AuthService.verify_access_token(token)
        return UserRepository.get_by_username(db, token_data.username)
    except Exception:
        return None


# ─────────────────────────────
# UI ROUTES
# ─────────────────────────────

@app.get("/login", response_class=HTMLResponse, tags=["UI"])
def login_page(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the login page. If already authenticated, redirect based on user role."""
    if user:
        if user.role == "clinician":
            return RedirectResponse(url="/clinician/dashboard", status_code=303)
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html")


@app.get("/register", response_class=HTMLResponse, tags=["UI"])
def register_page(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the registration page. If already authenticated, redirect to screening form."""
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "register.html")


@app.get("/logout", tags=["UI"])
def logout_ui():
    """Clear cookies and redirect to login page."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return response


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def index(
    request: Request,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Render the screening form. Restricted to authenticated users."""
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    all_children = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).all()
    active_child = all_children[0] if all_children else None
    pending_reminders = ReminderService.get_user_reminders(db, user.id, status="pending")

    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "user"              : user,
            "all_children"      : all_children,
            "active_child"      : active_child,
            "pending_reminders" : pending_reminders,
            "model_card"        : state.model_card,
            "apps_count"        : len(state.df_apps),
            "books_count"       : len(state.df_books),
            "chat_available"    : state.gemini_client is not None,  
        },
    )

        
@app.post("/screen", response_class=HTMLResponse, tags=["UI"])
async def screen(
    request: Request,
    age: int = Form(...), sex: int = Form(...),
    A1: str = Form(...), A2: str = Form(...), A3: str = Form(...),
    A4: str = Form(...), A5: str = Form(...), A6: str = Form(...),
    A7: str = Form(...), A8: str = Form(...), A9: str = Form(...),
    A10: str = Form(...),
    child_id: Optional[str] = Form(None),
    top_n: int = Form(3),
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db),
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        log.info(f"Processing screen request: user={user.username}, age={age}, sex={sex}, top_n={top_n}")
        scores = {
            "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
            "A6": A6, "A7": A7, "A8": A8, "A9": A9, "A10": A10,
            "Age": age, "Sex": sex,
        }
    
        risk = predict_risk(scores)
        high_risk = risk >= 40.0   # Threshold for high risk is 40%
        
        flagged_details = [
            {"code": k, "label": QUESTION_LABELS[k]}
            for k in QUESTION_LABELS if check_flag(k, scores.get(k))
        ]
        total_flags = len(flagged_details)
        
        profile_text = build_profile_text(scores) if high_risk else ""
        app_recs     = recommend_apps(profile_text, top_n) if high_risk else []
        book_recs    = recommend_books(profile_text, top_n) if high_risk else []

        profile_explained = ""
        if state.gemini_client:
            try:
                profile_explained = await explain_profile(
                    age              = age,
                    sex_label        = "Male" if sex == 1 else "Female",
                    risk_probability = round(risk, 1),
                    total_flags      = total_flags,
                    flagged_details  = flagged_details,
                    profile_text     = profile_text,
                    gemini_client    = state.gemini_client,
                )
            except Exception as e:
                log.error("explain_profile crashed: %s", e)
                profile_explained = "We recommend focusing on communication and social engagement activities."
        else:
            profile_explained = "We recommend focusing on communication and social engagement activities."

        # ─────────────────────────────────────────────────────────────
        # RELATIONAL PERSISTENCE: Multi-Child & Screening Assessment
        # ─────────────────────────────────────────────────────────────
        try:
            # 1. Retrieve or auto-create ChildProfile for this user
            child = None
            if child_id:
                try:
                    c_uuid = UUID(str(child_id).strip())
                    child = db.query(ChildProfile).filter(ChildProfile.id == c_uuid, ChildProfile.user_id == user.id).first()
                except Exception:
                    child = None
            
            if not child:
                child = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).first()

            if not child:
                approx_dob = (datetime.now(timezone.utc) - timedelta(days=int(age * 30.44))).date()
                child = ChildProfile(
                    user_id=user.id,
                    first_name=f"{user.username}'s Child",
                    date_of_birth=approx_dob,
                    biological_sex=sex,
                )
                db.add(child)
                db.commit()
                db.refresh(child)
                log.info(f"✓ Provisioned default ChildProfile ({child.id}) for user '{user.username}'")
            else:
                child.biological_sex = sex
                db.commit()

            # 2. Persist ScreeningAssessment
            assessment = ScreeningAssessment(
                child_id=child.id,
                age_months=age,
                a1_score=map_likert_standard(A1),
                a2_score=map_likert_standard(A2),
                a3_score=map_likert_standard(A3),
                a4_score=map_likert_standard(A4),
                a5_score=map_likert_standard(A5),
                a6_score=map_likert_standard(A6),
                a7_score=map_likert_standard(A7),
                a8_score=map_likert_standard(A8),
                a9_score=map_likert_reverse(A9),
                a10_score=map_likert_reverse(A10),
                risk_probability=round(risk, 2),
                is_high_risk=high_risk,
                total_flags=total_flags,
                profile_text=profile_text,
                profile_explained=profile_explained,
            )
            db.add(assessment)
            db.flush()

            # 3. Persist Recommendations
            for rec in app_recs:
                db.add(AssessmentRecommendation(
                    assessment_id=assessment.id,
                    resource_type="app",
                    item_name=rec.app_name,
                    category=rec.category,
                    match_score=rec.match_score,
                    rank=rec.rank,
                ))

            for rec in book_recs:
                db.add(AssessmentRecommendation(
                    assessment_id=assessment.id,
                    resource_type="book",
                    item_name=rec.title,
                    category=rec.category,
                    match_score=rec.match_score,
                    rank=rec.rank,
                ))

            db.commit()
            log.info(f"✓ Successfully persisted ScreeningAssessment {assessment.id} with {len(app_recs)} apps, {len(book_recs)} books")

        except Exception as db_err:
            db.rollback()
            log.error(f"Failed to persist screening record to database: {db_err}")

        screening_context = {
            "age": age,
            "sex_label": "Male" if sex == 1 else "Female",
            "risk_probability": round(risk, 1),
            "total_flags": total_flags,
            "flagged_questions": [f"{d['code']}: {d['label']}" for d in flagged_details],
            "recommended_apps": [r.app_name for r in app_recs],
            "recommended_books": [r.title for r in book_recs],
            "profile_text": profile_text,
            "profile_explained": profile_explained,
        }

        return templates.TemplateResponse(
            request,
            "results.html",
            context={
                "user": user,
                "age": age,
                "sex_label": "Male" if sex == 1 else "Female",
                "risk_probability": round(risk, 1),
                "high_risk": high_risk,
                "total_flags": total_flags,
                "flagged_details": flagged_details,
                "profile_text": profile_text,
                "profile_explained": profile_explained,
                "child": child,
                "child_id": str(child.id) if child else "",
                "recommendations": app_recs,
                "book_recommendations": book_recs,
                "model_card": state.model_card,
                "screening_context": json.dumps(screening_context),
                "chat_available": state.gemini_client is not None,
            },
        )
        
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        full_traceback = "".join(tb_lines)
        log.error(f"SCREEN ENDPOINT FAILED:\n{full_traceback}")
        print(f"\n{'='*80}\nERROR IN /screen:\n{full_traceback}\n{'='*80}\n", file=sys.stderr)
        return HTMLResponse(f"<h1>Screening Error</h1><pre>{str(e)}</pre>", status_code=500)


@app.get("/apps-page", response_class=HTMLResponse, tags=["UI"])
def apps_page(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the apps catalogue page. Restricted to authenticated users."""
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        if state.df_apps.empty:
            apps_list = []
            total = 0
        else:
            apps_list = []
            for _, row in state.df_apps.iterrows():
                apps_list.append({
                    "app_name": row["App_Name"],
                    "category": row.get("Category", "Uncategorized"),
                    "rating": float(row.get("Rating", 0)),
                    "price": row.get("Price", "Free"),
                    "description": row.get("Description", "No description available.")[:200],
                })
            total = len(apps_list)
        
        log.info(f"Rendering apps page with {total} apps")
        
        return templates.TemplateResponse(
            request,
            "all_apps.html",
            context={
                "user": user,
                "apps": apps_list,
                "total_apps": total,
                "chat_available": state.gemini_client is not None,
            },
        )
    except Exception as e:
        log.error(f"Error rendering apps page: {e}")
        return HTMLResponse(f"<h1>Error</h1><pre>{str(e)}</pre>", status_code=500)



# ─────────────────────────────
# PHASE 2: JOURNEY & SCHEDULE UI ROUTES
# ─────────────────────────────

@app.get("/journey", response_class=HTMLResponse, tags=["UI"])
def journey_redirect(
    request: Request,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).first()
    if not child:
        approx_dob = (datetime.now(timezone.utc) - timedelta(days=24 * 30)).date()
        child = ChildProfile(
            user_id=user.id,
            first_name=f"{user.username}'s Child",
            date_of_birth=approx_dob,
            biological_sex=1,
        )
        db.add(child)
        db.commit()
        db.refresh(child)
    return RedirectResponse(url=f"/journey/{child.id}", status_code=303)


@app.get("/journey/{child_id}", response_class=HTMLResponse, tags=["UI"])
def journey_view(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id, ChildProfile.user_id == user.id).first()
    if not child:
        return RedirectResponse(url="/journey", status_code=303)
    
    all_children = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).all()
    trajectory = TrajectoryService.get_trajectory(db, child)
    return templates.TemplateResponse(
        request,
        "journey.html",
        context={
            "user": user,
            "child": child,
            "all_children": all_children,
            "trajectory": trajectory,
            "chat_available": state.gemini_client is not None,
        }
    )


@app.get("/journey/{child_id}/print", response_class=HTMLResponse, tags=["UI"])
def journey_print_view(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id, ChildProfile.user_id == user.id).first()
    if not child:
        return RedirectResponse(url="/journey", status_code=303)

    trajectory = TrajectoryService.get_trajectory(db, child)
    assessments = (
        db.query(ScreeningAssessment)
        .filter(ScreeningAssessment.child_id == child.id)
        .order_by(ScreeningAssessment.completed_at.desc())
        .all()
    )
    context = {
        "child": child,
        "parent_user": user,
        "trajectory": trajectory,
        "assessments": assessments,
        "latest_assessment": assessments[0] if assessments else None,
        "generated_at": datetime.now(timezone.utc).strftime("%B %d, %Y"),
    }
    return templates.TemplateResponse(request, "clinical_report.html", context=context)


@app.get("/schedule", response_class=HTMLResponse, tags=["UI"])
def schedule_redirect(
    request: Request,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).first()
    if not child:
        approx_dob = (datetime.now(timezone.utc) - timedelta(days=24 * 30)).date()
        child = ChildProfile(
            user_id=user.id,
            first_name=f"{user.username}'s Child",
            date_of_birth=approx_dob,
            biological_sex=1,
        )
        db.add(child)
        db.commit()
        db.refresh(child)
    return RedirectResponse(url=f"/schedule/{child.id}", status_code=303)


@app.get("/schedule/{child_id}", response_class=HTMLResponse, tags=["UI"])
def schedule_view(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id, ChildProfile.user_id == user.id).first()
    if not child:
        return RedirectResponse(url="/schedule", status_code=303)
    
    all_children = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).all()
    schedules = ScheduleService.get_child_schedules(db, child.id)
    streak = ScheduleService.calculate_streak(db, child.id)
    return templates.TemplateResponse(
        request,
        "schedules.html",
        context={
            "user": user,
            "child": child,
            "all_children": all_children,
            "schedules": schedules,
            "streak": streak,
            "chat_available": state.gemini_client is not None,
        }
    )



# ─────────────────────────────
# PHASE 3: MELTDOWN EMERGENCY SUITE UI ROUTES
# ─────────────────────────────

@app.get("/meltdown", response_class=HTMLResponse, tags=["UI"])
def meltdown_redirect(
    request: Request,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).first()
    if not child:
        approx_dob = (datetime.now(timezone.utc) - timedelta(days=24 * 30)).date()
        child = ChildProfile(
            user_id=user.id,
            first_name=f"{user.username}'s Child",
            date_of_birth=approx_dob,
            biological_sex=1,
        )
        db.add(child)
        db.commit()
        db.refresh(child)
    return RedirectResponse(url=f"/meltdown/{child.id}/active", status_code=303)


@app.get("/meltdown/{child_id}", response_class=HTMLResponse, tags=["UI"])
def meltdown_child_redirect(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    return RedirectResponse(url=f"/meltdown/{child_id}/active", status_code=303)


@app.get("/meltdown/{child_id}/active", response_class=HTMLResponse, tags=["UI"])
def meltdown_active_view(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id, ChildProfile.user_id == user.id).first()
    if not child:
        return RedirectResponse(url="/meltdown", status_code=303)

    all_children = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).all()
    strategies = MeltdownService.get_quick_calming_strategies(db, child.id)
    return templates.TemplateResponse(
        request,
        "meltdown_active.html",
        context={
            "user": user,
            "child": child,
            "all_children": all_children,
            "strategies": strategies,
            "chat_available": state.gemini_client is not None,
        }
    )


@app.get("/meltdown/{child_id}/analytics", response_class=HTMLResponse, tags=["UI"])
def meltdown_analytics_view(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id, ChildProfile.user_id == user.id).first()
    if not child:
        return RedirectResponse(url="/meltdown", status_code=303)

    all_children = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).order_by(ChildProfile.created_at.asc()).all()
    analytics = MeltdownService.compute_environmental_patterns(db, child)
    incidents = MeltdownService.get_child_incidents(db, child.id)
    return templates.TemplateResponse(
        request,
        "meltdown_analytics.html",
        context={
            "user": user,
            "child": child,
            "all_children": all_children,
            "analytics": analytics,
            "incidents": incidents,
            "chat_available": state.gemini_client is not None,
        }
    )


# ?????????????????????????????
# PHASE 4: CLINICIAN PORTAL & RAG EVIDENCE UI ROUTES
# ?????????????????????????????

@app.get("/clinician/dashboard", response_class=HTMLResponse, tags=["UI"])
def clinician_dashboard_view(
    request: Request,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if user.role != "clinician":
        return RedirectResponse(url="/", status_code=303)

    roster = ClinicianService.get_clinician_roster(db, user.id)
    total_patients = len(roster)
    urgent_count = sum(1 for p in roster if p.triage_priority == "urgent")
    monitor_count = sum(1 for p in roster if p.triage_priority == "monitor")
    stable_count = sum(1 for p in roster if p.triage_priority == "stable")

    all_children = db.query(ChildProfile).order_by(ChildProfile.first_name.asc()).all()

    return templates.TemplateResponse(
        request,
        "clinician_dashboard.html",
        context={
            "user": user,
            "roster": roster,
            "total_patients": total_patients,
            "urgent_count": urgent_count,
            "monitor_count": monitor_count,
            "stable_count": stable_count,
            "all_children": all_children,
            "chat_available": state.gemini_client is not None,
        }
    )


@app.get("/clinician/patients/{child_id}", response_class=HTMLResponse, tags=["UI"])
def clinician_patient_detail_view(
    request: Request,
    child_id: UUID,
    user: Optional[User] = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if user.role != "clinician":
        return RedirectResponse(url="/", status_code=303)

    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        return RedirectResponse(url="/clinician/dashboard", status_code=303)

    parent = db.query(User).filter(User.id == child.user_id).first()
    trajectory = TrajectoryService.get_trajectory(db, child)
    meltdown_analytics = MeltdownService.compute_environmental_patterns(db, child)
    schedules = ScheduleService.get_child_schedules(db, child.id)
    notes = ClinicianService.get_patient_clinical_notes(db, child.id)

    return templates.TemplateResponse(
        request,
        "clinician_patient_detail.html",
        context={
            "user": user,
            "child": child,
            "parent": parent,
            "trajectory": trajectory,
            "meltdown_analytics": meltdown_analytics,
            "schedules": schedules,
            "notes": notes,
            "chat_available": state.gemini_client is not None,
        }
    )


@app.get("/evidence", response_class=HTMLResponse, tags=["UI"])
def evidence_explorer_view(
    request: Request,
    user: Optional[User] = Depends(get_current_user_from_cookie)
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    evidence_items = EvidenceRAGService.search_evidence(query="", top_k=20)
    return templates.TemplateResponse(
        request,
        "evidence_explorer.html",
        context={
            "user": user,
            "evidence_items": evidence_items,
            "chat_available": state.gemini_client is not None,
        }
    )


if __name__ == "__main__":


    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
