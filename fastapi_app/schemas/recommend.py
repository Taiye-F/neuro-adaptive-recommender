    

"""
schemas.py — Pydantic models for all API request/response bodies.

Having schemas in a separate file means:
  - main.py stays readable
  - FastAPI auto-generates accurate OpenAPI docs from these types
  - The Streamlit app can import them too for type-safe calls
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

LikertScale = Literal["Always", "Usually", "Sometimes", "Rarely", "Never"]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def _binary(v: int) -> int:
    if v not in (0, 1):
        raise ValueError("Must be 0 or 1.")
    return v


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralProfile(BaseModel):
    """
    Q-CHAT-10 behavioral screening profile for a toddler.
    Accepts 5-point Likert scale responses: Always, Usually, Sometimes, Rarely, Never.
    Age: Child's age in months (12-48).
    Sex: 1 = Male, 0 = Female.
    """

    A1  : LikertScale = Field(..., description="Looks at you when called?")
    A2  : LikertScale = Field(..., description="Makes eye contact easily?")
    A3  : LikertScale = Field(..., description="Points to indicate wants?")
    A4  : LikertScale = Field(..., description="Points to share interest?")
    A5  : LikertScale = Field(..., description="Engages in pretend play?")
    A6  : LikertScale = Field(..., description="Follows where you look?")
    A7  : LikertScale = Field(..., description="Speaks basic words?")
    A8  : LikertScale = Field(..., description="Understands simple gestures?")
    A9  : LikertScale = Field(..., description="Unusual sensory reactions (e.g. staring at nothing)?")
    A10 : LikertScale = Field(..., description="Repetitive behaviors (e.g. hand flapping)?")
    Age : int = Field(..., ge=12, le=48, description="Child's age in months (12–48)")
    Sex : int = Field(..., ge=0, le=1, description="Biological sex: 1 = Male, 0 = Female")

    model_config = {
        "json_schema_extra": {
            "example": {
                "A1": "Usually", "A2": "Usually", "A3": "Usually", "A4": "Usually", "A5": "Usually",
                "A6": "Usually", "A7": "Usually", "A8": "Usually", "A9": "Never", "A10": "Never",
                "Age": 24, "Sex": 1,
            }
        }
    }


class PredictRequest(BehavioralProfile):
    """Request body for POST /predict."""
    pass


class RecommendRequest(BehavioralProfile):
    """Request body for POST /recommend."""
    top_n: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of app recommendations to return (1–10).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "A1": "Usually", "A2": "Usually", "A3": "Usually", "A4": "Usually", "A5": "Usually",
                "A6": "Usually", "A7": "Usually", "A8": "Usually", "A9": "Never", "A10": "Never",
                "Age": 24, "Sex": 1,
                "top_n": 3,
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Response body for POST /predict."""
    risk_probability  : float = Field(..., description="ASD trait probability 0–100%")
    high_risk         : bool  = Field(..., description="True if probability ≥ 50%")
    total_flags       : int   = Field(..., description="Number of Q-CHAT questions flagged")
    flagged_questions : list[str] = Field(..., description="Human-readable list of flagged milestones")
    latency_ms        : float = Field(..., description="Inference latency in milliseconds")


class RecommendedApp(BaseModel):
    """A single ranked app recommendation."""
    rank        : int
    app_name    : str
    category    : str
    rating      : float
    price       : str
    description : str
    url         : Optional[str] = None
    match_score : float = Field(..., description="TF-IDF cosine similarity score 0–100")


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""
    risk_probability : float
    high_risk        : bool
    total_flags      : int
    profile_text     : str  = Field(..., description="Semantic need-query derived from behavioral profile")
    recommendations  : list[RecommendedApp]
    message          : str
    latency_ms       : float


class AppItem(BaseModel):
    """A single app from the cache."""
    app_name    : str
    category    : str
    rating      : float
    price       : str
    description : str


class AppsResponse(BaseModel):
    """Response body for GET /apps."""
    total : int
    apps  : list[AppItem]


class BookItem(BaseModel):
    """A single book from book_cache.json."""
    title       : str
    author      : str
    category    : str
    age_range   : str
    description : str
    access      : str = Field(..., description="'free' or 'paid'")
    free_url    : Optional[str] = None
    paid_url    : Optional[str] = None
    cover_emoji : str = "📖"


class BookRecommendation(BookItem):
    """A ranked book recommendation with match score."""
    rank        : int
    match_score : float = Field(..., description="TF-IDF cosine similarity 0–100")


class BooksResponse(BaseModel):
    """Response body for GET /books."""
    total : int
    books : list[BookItem]


class RecommendedApp(BaseModel):
    """A single ranked app recommendation."""
    rank        : int
    app_name    : str
    category    : str
    rating      : float
    price       : str
    description : str
    url         : Optional[str] = None
    match_score : float = Field(..., description="TF-IDF cosine similarity score 0–100")


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""
    risk_probability    : float
    high_risk           : bool
    total_flags         : int
    profile_text        : str  = Field(..., description="Semantic need-query derived from behavioral profile")
    recommendations     : list[RecommendedApp]
    book_recommendations: list[BookRecommendation]
    message             : str
    latency_ms          : float


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status         : str
    model_loaded   : bool
    model_type     : str
    test_accuracy  : float
    test_roc_auc   : float
    feature_cols   : list[str]
    apps_in_cache  : int
    books_in_cache : int
    uptime_seconds : float


# ─────────────────────────────────────────────────────────────────────────────
# CHAT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ScreeningContext(BaseModel):
    """
    Snapshot of the child's screening result — sent with every chat message
    so Gemini always has personalised context without server-side session state.
    """
    age               : int
    sex_label         : str
    risk_probability  : float
    total_flags       : int
    flagged_questions : list[str]
    recommended_apps  : list[str]   = Field(default_factory=list)
    recommended_books : list[str]   = Field(default_factory=list)
    profile_text      : str         = ""


class ChatMessage(BaseModel):
    role    : str = Field(..., description="'user' or 'assistant'")
    content : str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    screening_context : ScreeningContext
    history           : list[ChatMessage] = Field(
        default_factory=list,
        description="Full conversation history (all prior turns). Max 20 turns.",
    )
    message           : str = Field(..., min_length=1, max_length=1000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "screening_context": {
                    "age": 24,
                    "sex_label": "Male",
                    "risk_probability": 87.4,
                    "total_flags": 7,
                    "flagged_questions": ["A1: Responds to name", "A7: Basic speech"],
                    "recommended_apps": ["Otsimo Special Education", "Proloquo2Go"],
                    "recommended_books": ["More Than Words", "An Early Start for Your Child with Autism"],
                    "profile_text": "speech delay non-verbal communication words language"
                },
                "history": [],
                "message": "What does the speech delay score mean for my child's development?"
            }
        }
    }


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply      : str
    latency_ms : float


# class ChatResponse(BaseModel):
#     """Response body for POST /chat."""
#     reply: str = Field(..., description="AI assistant's response")
#     latency_ms: float = Field(..., description="Response time in milliseconds")
    
#     # Optional fields for better debugging and UX
#     error: Optional[str] = Field(None, description="Error message if any")
#     success: bool = Field(True, description="Whether the request was successful")
