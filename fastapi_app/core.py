import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Optional, Any
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google_play_scraper import search, app as play_store_app

from schemas.recommend import RecommendedApp, BookRecommendation

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
BASE_DIR          = Path(__file__).parent
MODEL_PATH        = BASE_DIR / "files/asd_ordinal_top10_model.pkl"
MODEL_CARD_PATH   = BASE_DIR / "files/model_card1.json"
APP_CACHE_PATH    = BASE_DIR / "files/app_cache.json"
BOOK_CACHE_PATH   = BASE_DIR / "files/book_cache.json"
EVIDENCE_CSV_PATH = BASE_DIR / "files/evidence_based_apps_2.csv"
TEMPLATES_DIR     = BASE_DIR / "templates"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & METADATA
# ─────────────────────────────────────────────────────────────────────────────
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

EVIDENCE_KEYWORDS = [
    "evidence-based", "evidence based", "clinically proven", "peer-reviewed",
    "randomized control trial", "randomized controlled trial", "rct",
    "developed with therapists", "developed with speech-language pathologists",
    "developed with slps", "aba-based", "aba based", "clinically validated",
    "developed by psychologists", "developed by clinicians", "research-backed",
    "backed by research", "clinical trial", "published study"
]

GEMINI_MODEL = "models/gemini-3.5-flash"

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

GENERAL_CHAT_SYSTEM_PROMPT = """\
You are a warm, knowledgeable special education consultant named Nora.
You are chatting with a parent who is seeking general information about child development, autism spectrum traits, and early intervention.

IMPORTANT CONTEXT:
- No screening has been conducted yet for this session.
- DO NOT invent, assume, or reference any specific child age, risk percentage, flag count, or assessment results.
- DO NOT say the baby is newborn, 0 months old, or has 0.0% risk.
- Focus on answering their questions about autism, developmental milestones (12–48 months), sensory regulation, and when to seek developmental screening or pediatric guidance.

Guidelines:
1. Always be compassionate, encouraging, and supportive.
2. Never provide a medical diagnosis or replace professional advice.
3. Keep responses concise (under 200 words) and jargon-free.
4. If a parent is concerned about their toddler (aged 12–48 months), gently encourage them to complete the NeuroAdapt early screening questionnaire on the platform.
5. Remind parents to consult a developmental paediatrician or early childhood specialist if they have persistent concerns.
"""


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    model            = None
    model_card       : dict = {}
    
    # Apps (Scraped + Local Fallback)
    df_apps          : pd.DataFrame = pd.DataFrame()
    app_tfidf        : Optional[TfidfVectorizer] = None
    app_matrix       = None
    
    # Books
    df_books         : pd.DataFrame = pd.DataFrame()
    book_tfidf       : Optional[TfidfVectorizer] = None
    book_matrix      = None
    
    # Evidence CSV lookup
    evidence_lookup  : dict[str, dict] = {}
    
    gemini_client    = None
    startup_time     : float = 0.0

state = AppState()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_html(text: str) -> str:
    return re.sub('<[^<]+?>', '', text)

def clean_nan(val: Any) -> Optional[str]:
    if pd.isna(val) or val is None or val == "":
        return None
    return str(val)

def map_likert_standard(val: str) -> int:
    mapping = {"Always": 0, "Usually": 1, "Sometimes": 2, "Rarely": 3, "Never": 4}
    return mapping.get(val, 1)

def map_likert_reverse(val: str) -> int:
    mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Usually": 3, "Always": 4}
    return mapping.get(val, 0)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    state.model = joblib.load(MODEL_PATH)
    
    if MODEL_CARD_PATH.exists():
        with open(MODEL_CARD_PATH, encoding="utf-8") as f:
            state.model_card = json.load(f)
    else:
        state.model_card = {
            "model_type": "XGBoost Classifier (Lean Ordinal)",
            "test_accuracy": 0.88,
            "test_roc_auc": 0.94,
        }
    log.info("Model loaded successfully")


def _load_evidence_lookup() -> None:
    if not EVIDENCE_CSV_PATH.exists():
        log.warning("evidence_based_apps_2.csv not found. Evidence tagging will be heuristic-only.")
        return
    try:
        df = pd.read_csv(EVIDENCE_CSV_PATH)
        state.evidence_lookup = df.set_index('app_name')[['evidence_tier', 'evidence_note']].to_dict('index')
        log.info("Evidence lookup loaded with %d apps.", len(state.evidence_lookup))
    except Exception as e:
        log.error("Failed to load evidence based CSV: %s", e)


def _fit_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)
    return vec, matrix


def _load_book_cache() -> None:
    if not BOOK_CACHE_PATH.exists():
        log.warning("book_cache.json not found — book recommendations unavailable.")
        return
    try:
        with open(BOOK_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        state.df_books = pd.DataFrame(data)
        texts = (
            state.df_books["asd_themes"].fillna("") + " " +
            state.df_books["description"].fillna("")
        ).tolist()
        state.book_tfidf, state.book_matrix = _fit_tfidf(texts)
        log.info("Book cache loaded — %d books.", len(state.df_books))
    except Exception as e:
        log.error("Failed to load book cache: %s", e)


def _tag_evidence(row: pd.Series) -> pd.Series:
    app_name = row['App_Name']
    if app_name in state.evidence_lookup:
        info = state.evidence_lookup[app_name]
        return pd.Series([True, info['evidence_tier'], info['evidence_note']])
    
    desc = str(row['Description']).lower()
    if any(kw in desc for kw in EVIDENCE_KEYWORDS):
        return pd.Series([True, 'self_reported', 'App description claims evidence-based approach (unverified)'])
    return pd.Series([False, 'unverified', ''])


def _load_app_cache() -> None:
    """
    Loads educational apps directly from local app_cache.json fallback.
    Live scraping is decoupled from startup to ensure instant readiness.
    Use scripts/refresh_app_cache.py to periodically update the cache.
    """
    _load_evidence_lookup()

    if APP_CACHE_PATH.exists():
        try:
            with open(APP_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            state.df_apps = pd.DataFrame(data)
            log.info("? Loaded %d apps from local app_cache.json", len(state.df_apps))
        except Exception as e:
            log.error("Failed to load app_cache.json: %s", e)
            state.df_apps = pd.DataFrame()
    else:
        log.warning("app_cache.json not found. App recommendations will be unavailable.")
        state.df_apps = pd.DataFrame()

    if not state.df_apps.empty:
        # Tag evidence
        state.df_apps[['Evidence_Based', 'Evidence_Tier', 'Evidence_Note']] = state.df_apps.apply(_tag_evidence, axis=1)
        # Fit TF-IDF
        texts = state.df_apps["Description"].fillna("").tolist()
        state.app_tfidf, state.app_matrix = _fit_tfidf(texts)


def _init_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.warning("GEMINI_API_KEY not set.")
        return
    try:
        state.gemini_client = genai.Client(api_key=api_key)
        log.info("Gemini client initialized successfully.")
    except Exception as e:
        log.warning("Gemini init failed: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
# ML WORKFLOWS
# ─────────────────────────────────────────────────────────────────────────────
def build_profile_text(scores: dict[str, Any]) -> str:
    a1 = map_likert_standard(scores.get("A1", "Usually"))
    a2 = map_likert_standard(scores.get("A2", "Usually"))
    a3 = map_likert_standard(scores.get("A3", "Usually"))
    a4 = map_likert_standard(scores.get("A4", "Usually"))
    a5 = map_likert_standard(scores.get("A5", "Usually"))
    a6 = map_likert_standard(scores.get("A6", "Usually"))
    a7 = map_likert_standard(scores.get("A7", "Usually"))
    a8 = map_likert_standard(scores.get("A8", "Usually"))
    a9 = map_likert_reverse(scores.get("A9", "Never"))
    a10 = map_likert_reverse(scores.get("A10", "Never"))

    toddler_needs = []
    # Speech & Language: A1 (responds to name) & A7 (uses words/speech)
    if a1 >= 3 or a7 >= 3:
        toddler_needs.append("speech delay non-verbal communication words language articulation speech therapy")
    elif a1 == 2 or a7 == 2:
        toddler_needs.append("early speech communication language enrichment words")

    # Joint attention & eye contact: A2 (eye contact) & A6 (follows gaze/pointing)
    if a2 >= 3 or a6 >= 3:
        toddler_needs.append("eye contact joint attention social gaze engagement interactive connection")
    elif a2 == 2 or a6 == 2:
        toddler_needs.append("shared attention eye contact interactive engagement")

    # Gestures & Shared Interest: A3 (points to wants) & A4 (points to share) & A8 (simple gestures)
    if a3 >= 3 or a4 >= 3 or a8 >= 3:
        toddler_needs.append("pointing gestures nonverbal communication social interaction sharing interest")
    elif a3 == 2 or a4 == 2 or a8 == 2:
        toddler_needs.append("gestures pointing nonverbal communication")

    # Pretend play: A5 (pretend play / symbolic play)
    if a5 >= 3:
        toddler_needs.append("pretend play imaginative play symbolic play toys imitation social play")
    elif a5 == 2:
        toddler_needs.append("creative play toys imitation social engagement")

    # Sensory processing & routines: A9 (sensory reactions) & A10 (repetitive behaviors)
    if a9 >= 3 or a10 >= 3:
        toddler_needs.append("sensory processing meltdowns routine calm visual schedule emotional regulation behavior ADHD sensory diet")
    elif a9 == 2 or a10 == 2:
        toddler_needs.append("sensory regulation routine calm visual structure")

    if not toddler_needs:
        toddler_needs.append("early childhood developmental milestones social communication play curiosity language enrichment")
        
    return " ".join(toddler_needs)


async def explain_profile(
    age: int,
    sex_label: str,
    risk_probability: float,
    total_flags: int,
    flagged_details: list,
    profile_text: str,
    gemini_client,
) -> str:
    """
    Uses Gemini to generate a warm, parent-friendly summary of the screening
    result — for both high and low risk outcomes, with a deterministic milestone fallback.
    """
    flagged_labels = ", ".join(
        f"{d['code']} ({d['label']})" for d in flagged_details
    ) or "none"

    def _fallback_explanation() -> str:
        if flagged_details:
            top_areas = ", ".join([d["label"].lower() for d in flagged_details[:3]])
            if risk_probability >= 40.0:
                return (
                    f"Based on the developmental screening, your toddler shows areas where extra support can make a big difference, "
                    f"particularly around {top_areas}. Early childhood intervention, guided play routines, and structured communication "
                    f"tools provide proven positive scaffolding for your child's ongoing development."
                )
            else:
                return (
                    f"Your toddler's screening shows an encouraging developmental profile, with emerging focus areas around {top_areas}. "
                    f"Continuing positive daily play, shared reading, and responsive interactions will support their natural milestone progression."
                )
        return (
            "Your toddler is meeting typical developmental milestones across all screened areas. "
            "Continuing positive daily interactive play, language exploration, and social routines will keep fostering their growth."
        )

    if not gemini_client:
        return _fallback_explanation()

    if risk_probability >= 40.0:
        prompt = f"""
            A {age}-month-old {sex_label} toddler was screened using the Q-CHAT-10 developmental 
            checklist. The result shows a {risk_probability:.1f}% likelihood of ASD traits, 
            with {total_flags} out of 10 milestones flagged: {flagged_labels}.
            Their developmental needs profile is: "{profile_text}".

            Write 2–3 warm, encouraging sentences for the parent explaining what this result 
            means in plain everyday language. Focus on what the child may be finding 
            challenging right now and why early support matters.
            Do not use clinical labels or the word "autism". Be compassionate and hopeful.
        """.strip()
    else:
        prompt = f"""
            A {age}-month-old {sex_label} toddler was screened using the Q-CHAT-10 developmental 
            checklist. The result shows a {risk_probability:.1f}% likelihood of ASD traits — 
            a low-risk result — with {total_flags} out of 10 milestones flagged: {flagged_labels or "none"}.

            Write 2–3 warm, reassuring sentences for the parent explaining what this result 
            means in plain everyday language. Acknowledge any milestones that were flagged 
            if any exist, but frame the overall result positively. 
            Recommend routine monitoring and continued engagement.
            Do not use clinical labels. Be warm and encouraging.
        """.strip()

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return response.text.strip() or _fallback_explanation()
    except Exception as e:
        log.warning(f"Profile Explanation failed: {e}")
        return _fallback_explanation()


def predict_risk(scores: dict[str, Any]) -> float:
    """Maps Likert inputs to 0-4 values and runs prediction on the lean XGBoost model."""
    a1 = map_likert_standard(scores.get("A1", "Usually"))
    a2 = map_likert_standard(scores.get("A2", "Usually"))
    a3 = map_likert_standard(scores.get("A3", "Usually"))
    a4 = map_likert_standard(scores.get("A4", "Usually"))
    a5 = map_likert_standard(scores.get("A5", "Usually"))
    a6 = map_likert_standard(scores.get("A6", "Usually"))
    a7 = map_likert_standard(scores.get("A7", "Usually"))
    a8 = map_likert_standard(scores.get("A8", "Usually"))
    a9 = map_likert_reverse(scores.get("A9", "Never"))
    a10 = map_likert_reverse(scores.get("A10", "Never"))
    
    age = int(scores.get("Age", 24))
    sex = int(scores.get("Sex", 1))

    # Compile input exactly in the feature order of the lean XGBoost model
    input_df = pd.DataFrame([{
        'qchat6recode': a6,
        'qchat2recode': a2,
        'qchat5recode': a5,
        'qchat4recode': a4,
        'qchat10recode': a10,
        'qchat15recode': a3,
        'qchat25recode': a9,
        'qchat11recode': a7,
        'qchat1recode': a1,
        'qchat17recode': a8,
        'age': age,
        'sex': sex
    }])
    
    return float(state.model.predict_proba(input_df)[0][1] * 100)


def recommend_apps(profile_text: str, top_n: int = 3) -> list[RecommendedApp]:
    """Applies TF-IDF Cosine Similarity and Evidence-Based prioritization to rank apps.
    Completely eliminates 0.0% matches and attaches valid store links."""
    if state.df_apps.empty or state.app_tfidf is None:
        return []
    
    import urllib.parse
    qvec = state.app_tfidf.transform([profile_text])
    scores = (cosine_similarity(qvec, state.app_matrix).flatten() * 100).round(1)
    
    df = state.df_apps.copy()
    df["match_score"] = scores

    # CRITICAL: Filter out 0.0% matches completely!
    df_positive = df[df["match_score"] > 0].copy()
    if df_positive.empty:
        return []

    # Priority 1: Strict high matches (>= 20%) that are Evidence-Based
    strict_matches = df_positive[
        (df_positive['match_score'] >= 20) & (df_positive['Evidence_Based'] == True)
    ].sort_values(by='match_score', ascending=False)

    if len(strict_matches) > 0:
        remaining = df_positive[~df_positive.index.isin(strict_matches.index)].sort_values(
            by=['Evidence_Based', 'match_score', 'Rating'], ascending=[False, False, False]
        )
        ranked_apps = pd.concat([strict_matches, remaining])
    else:
        # Priority 2: Rank by Evidence-Based, match score, rating
        ranked_apps = df_positive.sort_values(
            by=['Evidence_Based', 'match_score', 'Rating'],
            ascending=[False, False, False]
        )

    # Double check that zero matches are never returned
    ranked_apps = ranked_apps[ranked_apps['match_score'] > 0]
    if ranked_apps.empty:
        return []

    top = ranked_apps.head(top_n).reset_index(drop=True)
    
    results = []
    for i, row in top.iterrows():
        app_name = clean_nan(row.get("App_Name")) or ""
        link = clean_nan(row.get("App_Link")) or clean_nan(row.get("app_url")) or clean_nan(row.get("url"))
        if not link:
            link = f"https://play.google.com/store/search?q={urllib.parse.quote(app_name)}&c=apps"

        results.append(
            RecommendedApp(
                rank=i + 1,
                app_name=app_name,
                category=clean_nan(row.get("Category")) or "",
                rating=float(row.get("Rating", 0) if not pd.isna(row.get("Rating")) else 0),
                price=clean_nan(row.get("Price")) or "Free",
                description=str(clean_nan(row.get("Description")) or "")[:200],
                match_score=float(row["match_score"]),
                app_url=str(link),
                url=str(link),
            )
        )
    return results


def recommend_books(profile_text: str, top_n: int = 3) -> list[BookRecommendation]:
    """Uses TF-IDF similarity to recommend relevant educational/parental guidance books."""
    if state.df_books.empty or state.book_tfidf is None:
        return []
    
    import urllib.parse
    qvec = state.book_tfidf.transform([profile_text])
    scores = (cosine_similarity(qvec, state.book_matrix).flatten() * 100).round(1)
    
    df = state.df_books.copy()
    df["match_score"] = scores
    
    df_positive = df[df["match_score"] > 0]
    if not df_positive.empty:
        top = df_positive.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    else:
        top = df.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    
    results = []
    for i, row in top.iterrows():
        title = clean_nan(row.get("title")) or ""
        free_url = clean_nan(row.get("free_url"))
        paid_url = clean_nan(row.get("paid_url"))
        if not free_url and not paid_url:
            paid_url = f"https://www.google.com/search?tbm=bks&q={urllib.parse.quote(title)}"

        results.append(
            BookRecommendation(
                rank=i + 1,
                title=title,
                author=clean_nan(row.get("author")) or "",
                category=clean_nan(row.get("category")) or "",
                age_range=clean_nan(row.get("age_range")) or "",
                description=str(clean_nan(row.get("description")) or "")[:200],
                access=clean_nan(row.get("access")) or "free",
                free_url=free_url,
                paid_url=paid_url,
                cover_emoji=clean_nan(row.get("cover_emoji")) or "📖",
                match_score=float(row["match_score"]),
            )
        )
    return results
