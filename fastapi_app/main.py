# main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from dotenv import load_dotenv
load_dotenv(override=True)

from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
import time 
import traceback
import os
from typing import Optional
from sqlalchemy.orm import Session

from database import get_db
from services.auth_service import AuthService
from repositories.user_repository import UserRepository
from models.auth_models import User
from core import (
    state, _load_model, _load_app_cache, _load_book_cache, _init_gemini, 
    predict_risk, recommend_apps, recommend_books,
    map_likert_standard, map_likert_reverse,
    build_profile_text, explain_profile, QUESTION_LABELS, log
)
from routers import api_router
import models.auth_models
from database import engine, Base


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
        
        # Better API Key logging (without exposing the key)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            log.info(f"✓ GEMINI_API_KEY loaded successfully")
        else:
            log.warning("⚠️ GEMINI_API_KEY is missing — Chat feature will be disabled")

    except Exception as e:
        log.error(f"❌ Critical startup error: {e}")
        startup_success = False

    log.info(f"Startup completed in {time.time() - state.startup_time:.2f}s")
    log.info(f"Chat available: {state.gemini_client is not None}")
    log.info(f"Overall startup success: {startup_success}")

    yield
    log.info("Shutting down.")


# ──────────────────────────
# FASTAPI APP
# ─────────────────────────────
app = FastAPI(
    title="Neuro-Adaptive ASD Learning Recommender",
    description="ASD early-screening microservice for toddlers...",
    version="2.0.0",
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


@app.get("/login", response_class=HTMLResponse, tags=["UI"])
def login_page(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the login page. If already authenticated, redirect to screening form."""
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html")


@app.get("/register", response_class=HTMLResponse, tags=["UI"])
def register_page(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the registration page. If already authenticated, redirect to screening form."""
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "register.html")


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def index(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the screening form. Restricted to authenticated users."""
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "user"          : user,
            "model_card"    : state.model_card,
            "apps_count"    : len(state.df_apps),
            "books_count"   : len(state.df_books),
            "chat_available": state.gemini_client is not None,  
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
    top_n: int = Form(3),
    user: Optional[User] = Depends(get_current_user_from_cookie),
):
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        log.info(f"Processing screen request: age={age}, sex={sex}, top_n={top_n}")
        """Process the form, run inference, render results page."""
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

        screening_context = {
            "age": age,
            "sex_label": "Male" if sex == 1 else "Female",
            "risk_probability": round(risk, 1),
            "total_flags": total_flags,
            "flagged_questions": [f"{d['code']}: {d['label']}" for d in flagged_details],
            "recommended_apps": [r.app_name for r in app_recs],
            "recommended_books": [r.title for r in book_recs],
            "profile_text": profile_text,
            "profile_explained": profile_explained, # human-readable — shown in UI
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
                "recommendations": app_recs,
                "book_recommendations": book_recs,
                "model_card": state.model_card,
                "screening_context": json.dumps(screening_context),
                "chat_available": state.gemini_client is not None,
            },
        )
        
    except Exception as e:
        # Get full traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        full_traceback = "".join(tb_lines)
        
        # Log to both file and console
        log.error(f"SCREEN ENDPOINT FAILED:\n{full_traceback}")
        
        # Print to stderr as well (uvicorn will capture this)
        print(f"\n{'='*80}\nERROR IN /screen:\n{full_traceback}\n{'='*80}\n", 
              file=sys.stderr)
        

@app.get("/apps-page", response_class=HTMLResponse, tags=["UI"])
def apps_page(request: Request, user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Render the apps catalogue page. Restricted to authenticated users."""
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        # Get the apps data
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
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"<h1>Error</h1><pre>{str(e)}</pre>", status_code=500)



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)







# load_dotenv()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     log.info("Starting up…")
#     state.startup_time = time.time()
    
#     try:
#         _load_model()
#         log.info("✓ Model loaded")
        
#         # _download_if_missing() 
#         # log.info("✓ files downloaded loaded")

#         _load_app_cache()
#         log.info("✓ App cache loaded")
        
#         _init_gemini()
#         log.info(f"✓ Gemini client initialized: {state.gemini_client is not None}")
        
#         # Verify API key is loaded
#         load_dotenv()
#         api_key = os.getenv("GEMINI_API_KEY")
#         log.info(f"GEMINI_API_KEY from env: {'✓ Present' if api_key else '✗ Missing'}")
        
#     except Exception as e:
#         log.error(f"Startup error: {e}")
#         # Don't raise - allow app to start but with limited functionality
    
#     log.info(f"Startup complete in {time.time() - state.startup_time:.2f}s")
#     log.info(f"Chat available: {state.gemini_client is not None}")
    
#     yield
    
#     log.info("Shutting down.")

