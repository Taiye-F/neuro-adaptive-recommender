# routers/recommend_router.py
import time
from fastapi import APIRouter, HTTPException, Depends
from schemas.recommend import (
    PredictRequest, PredictResponse,
    RecommendRequest, RecommendResponse,
    HealthResponse, AppsResponse, AppItem,
    ChatRequest, ChatResponse,
)
from core import (
    state, predict_risk, recommend_apps, recommend_books,
    map_likert_standard, map_likert_reverse,
    build_profile_text, QUESTION_LABELS, CHAT_SYSTEM_PROMPT, GEMINI_MODEL, log
)
from services.auth_service import get_current_user, RoleChecker
from models.auth_models import User

recommend_router = APIRouter()


def check_flag(k: str, val: str) -> bool:
    if k in ["A9", "A10"]:
        return map_likert_reverse(val) >= 3
    else:
        return map_likert_standard(val) >= 3


@recommend_router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status        = "ok",
        model_loaded  = state.model is not None,
        model_type    = state.model_card.get("model_type", "unknown"),
        test_accuracy = state.model_card.get("test_accuracy", 0),
        test_roc_auc  = state.model_card.get("test_roc_auc", 0),
        feature_cols  = ['qchat6recode', 'qchat2recode', 'qchat5recode', 'qchat4recode',
                         'qchat10recode', 'qchat15recode', 'qchat25recode', 'qchat11recode',
                         'qchat1recode', 'qchat17recode', 'age', 'sex'],
        apps_in_cache = len(state.df_apps),
        books_in_cache= len(state.df_books),
        uptime_seconds= round(time.time() - state.startup_time, 1),
    )


@recommend_router.get("/apps", response_model=AppsResponse, tags=["Apps"])
def list_apps(current_user: User = Depends(RoleChecker(["parent", "clinician"]))):
    if state.df_apps.empty:
        raise HTTPException(status_code=503, detail="App cache not loaded.")
    apps = [
        AppItem(
            app_name    = row["App_Name"],
            category    = row.get("Category", ""),
            rating      = float(row.get("Rating", 0)),
            price       = row.get("Price", ""),
            description = row.get("Description", "")[:200],
        )
        for _, row in state.df_apps.iterrows()
    ]
    return AppsResponse(total=len(apps), apps=apps)


@recommend_router.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(
    request: PredictRequest,
    current_user: User = Depends(RoleChecker(["parent", "clinician"]))
):
    t0     = time.time()
    scores = request.model_dump()
    risk   = predict_risk(scores)
    flagged = [
        f"{k}: {QUESTION_LABELS[k]}"
        for k in QUESTION_LABELS
        if check_flag(k, scores.get(k))
    ]
    return PredictResponse(
        risk_probability  = round(risk, 2),
        high_risk         = risk >= 40.0,
        total_flags       = len(flagged),
        flagged_questions = flagged,
        latency_ms        = round((time.time() - t0) * 1000, 1),
    )


@recommend_router.post("/recommend", response_model=RecommendResponse, tags=["Inference"])
def recommend(
    request: RecommendRequest,
    current_user: User = Depends(RoleChecker(["parent", "clinician"]))
):
    t0          = time.time()
    scores      = request.model_dump(exclude={"top_n"})
    risk        = predict_risk(scores)
    flagged = [
        f"{k}: {QUESTION_LABELS[k]}"
        for k in QUESTION_LABELS
        if check_flag(k, scores.get(k))
    ]
    total_flags = len(flagged)

    if risk < 40.0:
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

    profile_text = build_profile_text(scores)
    app_recs     = recommend_apps(profile_text, request.top_n)
    book_recs    = recommend_books(profile_text, request.top_n)

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


@recommend_router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(
    request: ChatRequest,
    current_user: User = Depends(RoleChecker(["parent", "clinician"]))
):
    if state.gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Chat unavailable.."
        )

    t0  = time.time()
    ctx = request.screening_context

    system_prompt = CHAT_SYSTEM_PROMPT.format(
        age               = ctx.age,
        sex_label         = ctx.sex_label,
        risk_probability  = ctx.risk_probability,
        total_flags       = ctx.total_flags,
        flagged_questions = ", ".join(ctx.flagged_questions) or "none",
        profile_text      = ctx.profile_text or "general autism support",
        recommended_apps  = ", ".join(ctx.recommended_apps) or "none yet",
        recommended_books = ", ".join(ctx.recommended_books) or "none yet",
    )

    history_turns = request.history[-20:]
    messages = (
        [
            {"role": "user",  "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": (
                "Understood. I'm here to help you understand your child's screening "
                "results and early intervention options. What would you like to know?"
            )}]},
        ]
        + [
            {"role": m.role if m.role == "model" else "user",
             "parts": [{"text": m.content}]}
            for m in history_turns
        ]
        + [{"role": "user", "parts": [{"text": request.message}]}]
    )

    reply = "I'm sorry, I'm having trouble responding right now. Please try again in a moment."

    try:
        response = state.gemini_client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = messages,
        )
        reply = response.text
    except Exception as e:
        log.error("Gemini chat error: %s", e)

    return ChatResponse(
        reply      = reply,
        latency_ms = round((time.time() - t0) * 1000, 1),
    )
