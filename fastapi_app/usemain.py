"""
Neuro-Adaptive ASD Recommender — FastAPI Microservice
======================================================
Authors : Taiye Janet Fagbolade: Principal Investigator, Machine Learning Engineering (XGBoost/NLP), Cloud Deployment, and UI/UX. | Iyanu Arowosola: Technical Contributor (FastAPI integration).
Hackathon: Bluechips × Data Science Nigeria

Endpoints
---------
GET  /              → Web UI (screening form)
GET  /health        → Liveness + readiness check
GET  /apps          → Full cached app list
GET  /books         → Full cached book list
POST /predict       → ASD risk probability from behavioral profile
POST /recommend     → ASD risk + ranked app AND book recommendations
POST /chat          → Stateless contextual chat with Gemini
GET  /results       → Results page (rendered after form POST)
POST /screen        → Form submit handler → renders results.html
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import (
    AppItem,
    AppsResponse,
    BookItem,
    BookRecommendation,
    BooksResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendedApp,
    ScreeningContext,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
MODEL_PATH      = BASE_DIR / "asd_model.pkl"
MODEL_CARD_PATH = BASE_DIR / "model_card.json"
APP_CACHE_PATH  = BASE_DIR / "app_cache.json"
BOOK_CACHE_PATH = BASE_DIR / "book_cache.json"
TEMPLATES_DIR   = BASE_DIR / "templates"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NEED_MAPPING: dict[tuple, str] = {
    ("A1", "A7"):       "speech delay non-verbal communication words language talk",
    ("A2", "A6", "A8"): "eye contact joint attention gesture social awareness",
    ("A3", "A4", "A5"): "social interaction play cognitive learning pretend",
    ("A9", "A10"):      "sensory processing meltdowns routine calm repetitive behaviour visual",
}

QUESTION_LABELS: dict[str, str] = {
    "A1":  "Responds to name",
    "A2":  "Eye contact",
    "A3":  "Points to indicate wants",
    "A4":  "Points to share interest",
    "A5":  "Pretend play",
    "A6":  "Follows gaze / pointing",
    "A7":  "Uses basic words / speech",
    "A8":  "Understands simple gestures",
    "A9":  "Unusual sensory reactions",
    "A10": "Repetitive or unusual behaviours",
}

GEMINI_MODEL = "models/gemini-2.5-flash"

CHAT_SYSTEM_PROMPT = """\
You are a warm, knowledgeable special education consultant named Nora.
A parent has just received an ASD screening result for their toddler
and wants to understand more about autism and early intervention.

Child's screening context:
- Age        : {age} months
- Sex        : {sex_label}
- ASD Risk   : {risk_probability:.1f}%
- Flags      : {total_flags}/10 milestones flagged
- Flagged    : {flagged_questions}
- Needs      : {profile_text}
- Apps rec.  : {recommended_apps}
- Books rec. : {recommended_books}

Guidelines:
1. Always be compassionate and encouraging — the parent may be distressed.
2. Reference the child's specific flags and recommendations when relevant.
3. Never provide a medical diagnosis or replace professional advice.
4. Keep responses concise (under 200 words) and jargon-free.
5. If asked about a specific app or book in the recommendations, explain
   how it addresses this child's specific developmental areas.
6. Gently remind parents to seek a developmental paediatrician when
   the topic warrants it — but always frame it positively.
"""

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE
# Loaded once at startup. TF-IDF matrices pre-fitted so every request
# only runs transform() + cosine_similarity() — not fit() again.
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    model            = None
    model_card       : dict       = {}
    feature_cols     : list[str]  = []

    df_apps          : pd.DataFrame = pd.DataFrame()
    app_tfidf        : Optional[TfidfVectorizer] = None
    app_matrix       = None           # sparse TF-IDF matrix (apps)

    df_books         : pd.DataFrame = pd.DataFrame()
    book_tfidf       : Optional[TfidfVectorizer] = None
    book_matrix      = None           # sparse TF-IDF matrix (books)

    gemini_client    = None
    startup_time     : float = 0.0


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not MODEL_CARD_PATH.exists():
        raise FileNotFoundError(f"Model card not found: {MODEL_CARD_PATH}")

    state.model = joblib.load(MODEL_PATH)
    with open(MODEL_CARD_PATH) as f:
        state.model_card = json.load(f)
    state.feature_cols = state.model_card["feature_columns"]
    log.info("Model loaded — features: %s", state.feature_cols)


def _fit_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, object]:
    """Fit a TF-IDF vectoriser on a list of texts and return (vectoriser, matrix)."""
    vec    = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)
    return vec, matrix


def _load_app_cache() -> None:
    if not APP_CACHE_PATH.exists():
        log.warning("app_cache.json not found — app recommendations unavailable.")
        return
    with open(APP_CACHE_PATH) as f:
        data = json.load(f)
    state.df_apps = pd.DataFrame(data)
    texts = state.df_apps["Description"].fillna("").tolist()
    state.app_tfidf, state.app_matrix = _fit_tfidf(texts)
    log.info("App cache loaded and TF-IDF fitted — %d apps.", len(state.df_apps))


def _load_book_cache() -> None:
    if not BOOK_CACHE_PATH.exists():
        log.warning("book_cache.json not found — book recommendations unavailable.")
        return
    with open(BOOK_CACHE_PATH) as f:
        data = json.load(f)
    state.df_books = pd.DataFrame(data)
    # Match against the rich asd_themes field + description combined
    texts = (
        state.df_books["asd_themes"].fillna("") + " " +
        state.df_books["description"].fillna("")
    ).tolist()
    state.book_tfidf, state.book_matrix = _fit_tfidf(texts)
    log.info("Book cache loaded and TF-IDF fitted — %d books.", len(state.df_books))


def _init_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.warning("GEMINI_API_KEY not set — /chat endpoint will be unavailable.")
        return
    try:
        from google import genai
        state.gemini_client = genai.Client(api_key=api_key)
        log.info("Gemini client initialised.")
    except Exception as e:
        log.warning("Gemini init failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up…")
    state.startup_time = time.time()
    _load_model()
    _load_app_cache()
    _load_book_cache()
    _init_gemini()
    log.info("Startup complete in %.2fs.", time.time() - state.startup_time)
    yield
    log.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Neuro-Adaptive ASD Learning Recommender",
    description=(
        "ASD early-screening microservice for toddlers (12–36 months). "
        "Returns risk probability, personalised app recommendations, "
        "book downloads, and a contextual chat interface."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# SHARED INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_profile_text(scores: dict[str, int]) -> str:
    needs = [
        phrase
        for questions, phrase in NEED_MAPPING.items()
        if any(scores.get(q, 0) == 1 for q in questions)
    ]
    return " ".join(needs) if needs else "autism special education cognitive skills"


def _predict_risk(scores: dict[str, int]) -> float:
    input_df = pd.DataFrame([{col: scores[col] for col in state.feature_cols}])
    return float(state.model.predict_proba(input_df)[0][1] * 100)


def _recommend_apps(profile_text: str, top_n: int) -> list[RecommendedApp]:
    """Uses pre-fitted TF-IDF matrix — only transform() called per request."""
    if state.df_apps.empty or state.app_tfidf is None:
        return []
    qvec   = state.app_tfidf.transform([profile_text])
    scores = (cosine_similarity(qvec, state.app_matrix).flatten() * 100).round(1)
    df     = state.df_apps.copy()
    df["match_score"] = scores
    top = df.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    return [
        RecommendedApp(
            rank        = i + 1,
            app_name    = row["App_Name"],
            category    = row.get("Category", ""),
            rating      = float(row.get("Rating", 0)),
            price       = row.get("Price", ""),
            description = str(row.get("Description", ""))[:200],
            match_score = float(row["match_score"]),
        )
        for i, row in top.iterrows()
    ]


def _recommend_books(profile_text: str, top_n: int) -> list[BookRecommendation]:
    """Uses pre-fitted book TF-IDF matrix — only transform() called per request."""
    if state.df_books.empty or state.book_tfidf is None:
        return []
    qvec   = state.book_tfidf.transform([profile_text])
    scores = (cosine_similarity(qvec, state.book_matrix).flatten() * 100).round(1)
    df     = state.df_books.copy()
    df["match_score"] = scores
    top = df.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    return [
        BookRecommendation(
            rank        = i + 1,
            title       = row["title"],
            author      = row["author"],
            category    = row.get("category", ""),
            age_range   = row.get("age_range", ""),
            description = str(row.get("description", ""))[:250],
            access      = row.get("access", "paid"),
            free_url    = row.get("free_url") or None,
            paid_url    = row.get("paid_url") or None,
            cover_emoji = row.get("cover_emoji", "📖"),
            match_score = float(row["match_score"]),
        )
        for i, row in top.iterrows()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS — SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status         = "ok",
        model_loaded   = state.model is not None,
        model_type     = state.model_card.get("model_type", "unknown"),
        test_accuracy  = state.model_card.get("test_accuracy", 0),
        test_roc_auc   = state.model_card.get("test_roc_auc", 0),
        feature_cols   = state.feature_cols,
        apps_in_cache  = len(state.df_apps),
        books_in_cache = len(state.df_books),
        uptime_seconds = round(time.time() - state.startup_time, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS — CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/apps", response_model=AppsResponse, tags=["Catalogue"])
def list_apps():
    if state.df_apps.empty:
        raise HTTPException(status_code=503, detail="App cache not loaded.")
    apps = [
        AppItem(
            app_name    = row["App_Name"],
            category    = row.get("Category", ""),
            rating      = float(row.get("Rating", 0)),
            price       = row.get("Price", ""),
            description = str(row.get("Description", ""))[:200],
        )
        for _, row in state.df_apps.iterrows()
    ]
    return AppsResponse(total=len(apps), apps=apps)


@app.get("/books", response_model=BooksResponse, tags=["Catalogue"])
def list_books():
    if state.df_books.empty:
        raise HTTPException(status_code=503, detail="Book cache not loaded.")
    books = [
        BookItem(
            title       = row["title"],
            author      = row["author"],
            category    = row.get("category", ""),
            age_range   = row.get("age_range", ""),
            description = str(row.get("description", ""))[:250],
            access      = row.get("access", "paid"),
            free_url    = row.get("free_url") or None,
            paid_url    = row.get("paid_url") or None,
            cover_emoji = row.get("cover_emoji", "📖"),
        )
        for _, row in state.df_books.iterrows()
    ]
    return BooksResponse(total=len(books), books=books)


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS — INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    t0     = time.time()
    scores = request.model_dump()
    risk   = _predict_risk(scores)
    flagged = [
        f"{k}: {QUESTION_LABELS[k]}"
        for k in ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10"]
        if scores.get(k, 0) == 1
    ]
    return PredictResponse(
        risk_probability  = round(risk, 2),
        high_risk         = risk >= 50.0,
        total_flags       = len(flagged),
        flagged_questions = flagged,
        latency_ms        = round((time.time() - t0) * 1000, 1),
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Inference"])
def recommend(request: RecommendRequest):
    t0     = time.time()
    scores = request.model_dump(exclude={"top_n"})
    risk   = _predict_risk(scores)
    total_flags = sum(v for k, v in scores.items() if k.startswith("A"))

    if risk < 50.0:
        return RecommendResponse(
            risk_probability     = round(risk, 2),
            high_risk            = False,
            total_flags          = total_flags,
            profile_text         = "",
            recommendations      = [],
            book_recommendations = [],
            message              = "Low likelihood of ASD traits. Standard monitoring recommended.",
            latency_ms           = round((time.time() - t0) * 1000, 1),
        )

    profile_text = _build_profile_text(scores)
    app_recs     = _recommend_apps(profile_text,  top_n=request.top_n)
    book_recs    = _recommend_books(profile_text, top_n=request.top_n)

    return RecommendResponse(
        risk_probability     = round(risk, 2),
        high_risk            = True,
        total_flags          = total_flags,
        profile_text         = profile_text,
        recommendations      = app_recs,
        book_recommendations = book_recs,
        message              = (
            f"High likelihood of ASD traits detected ({risk:.1f}%). "
            "Early intervention is recommended. Please consult a developmental paediatrician."
        ),
        latency_ms           = round((time.time() - t0) * 1000, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS — CHAT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Stateless contextual chat endpoint.
    The client sends the full screening context + conversation history
    on every turn. Gemini is given the child's profile as a system prompt
    so every reply is personalised, never generic.
    """
    if state.gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Chat unavailable — GEMINI_API_KEY not configured.",
        )

    t0  = time.time()
    ctx = request.screening_context

    # Build the personalised system prompt
    system_prompt = CHAT_SYSTEM_PROMPT.format(
        age               = ctx.age,
        sex_label         = ctx.sex_label,
        risk_probability  = ctx.risk_probability,
        total_flags       = ctx.total_flags,
        flagged_questions = ", ".join(ctx.flagged_questions) or "none",
        profile_text      = ctx.profile_text or "general autism support",
        recommended_apps  = ", ".join(ctx.recommended_apps)  or "none yet",
        recommended_books = ", ".join(ctx.recommended_books) or "none yet",
    )

    # Build the message list for the API:
    # system prompt as first user turn (Gemini 1.5/2.x pattern),
    # then the conversation history, then the new user message.
    # Cap history at last 20 turns to stay within context limits.
    history_turns = request.history[-20:]
    messages = (
        [{"role": "user", "parts": [{"text": system_prompt}]},
         {"role": "model", "parts": [{"text": (
             "Understood. I'm here to help you understand your child's screening "
             "results and early intervention options. What would you like to know?"
         )}]}]
        + [{"role": m.role if m.role == "model" else "user",
            "parts": [{"text": m.content}]}
           for m in history_turns]
        + [{"role": "user", "parts": [{"text": request.message}]}]
    )

    try:
        response = state.gemini_client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = messages,
        )
        reply = response.text
    except Exception as e:
        log.error("Gemini chat error: %s", e)
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")

    return ChatResponse(
        reply      = reply,
        latency_ms = round((time.time() - t0) * 1000, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WEB UI ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "model_card" : state.model_card,
            "apps_count" : len(state.df_apps),
            "books_count": len(state.df_books),
        },
    )


@app.post("/screen", response_class=HTMLResponse, tags=["UI"])
async def screen(
    request : Request,
    age     : int = Form(...),
    sex     : int = Form(...),
    A1      : int = Form(...),
    A2      : int = Form(...),
    A3      : int = Form(...),
    A4      : int = Form(...),
    A5      : int = Form(...),
    A6      : int = Form(...),
    A7      : int = Form(...),
    A8      : int = Form(...),
    A9      : int = Form(...),
    A10     : int = Form(...),
    top_n   : int = Form(3),
):
    scores = {
        "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
        "A6": A6, "A7": A7, "A8": A8, "A9": A9, "A10": A10,
        "Sex": sex,
    }

    risk        = _predict_risk(scores)
    high_risk   = risk >= 50.0
    total_flags = sum(v for k, v in scores.items() if k.startswith("A"))
    profile_text = _build_profile_text(scores) if high_risk else ""

    app_recs  = _recommend_apps(profile_text,  top_n=top_n) if high_risk else []
    book_recs = _recommend_books(profile_text, top_n=top_n) if high_risk else []

    flagged_details = [
        {"code": k, "label": QUESTION_LABELS[k]}
        for k in ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10"]
        if scores.get(k, 0) == 1
    ]

    # Build the screening context for the chat panel (serialised to JSON for JS)
    screening_context = {
        "age"              : age,
        "sex_label"        : "Male" if sex == 1 else "Female",
        "risk_probability" : round(risk, 1),
        "total_flags"      : total_flags,
        "flagged_questions": [f"{d['code']}: {d['label']}" for d in flagged_details],
        "recommended_apps" : [r.app_name for r in app_recs],
        "recommended_books": [r.title    for r in book_recs],
        "profile_text"     : profile_text,
    }

    return templates.TemplateResponse(
        request,
        "results.html",
        context={
            "age"               : age,
            "sex_label"         : "Male" if sex == 1 else "Female",
            "risk_probability"  : round(risk, 1),
            "high_risk"         : high_risk,
            "total_flags"       : total_flags,
            "flagged_details"   : flagged_details,
            "profile_text"      : profile_text,
            "recommendations"   : app_recs,
            "book_recommendations": book_recs,
            "model_card"        : state.model_card,
            "screening_context" : json.dumps(screening_context),
            "chat_available"    : state.gemini_client is not None,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    