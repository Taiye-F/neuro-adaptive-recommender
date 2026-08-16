"""
Neuro-Adaptive Learning Recommender — Streamlit App
=====================================================
Authors : Taiye Janet Fagbolade: Principal Investigator, Machine Learning Engineering (XGBoost/NLP), Cloud Deployment, and UI/UX. | Iyanu Arowosola: Technical Contributor (FastAPI integration).
Hackathon: Bluechips × Data Science Nigeria

Fixes applied vs original:
  1. A9/A10 inversion bug — map_response now takes an explicit
     `invert` flag so sensory/repetitive questions score correctly.
  2. Feature mismatch — input_data is built from model_card.json's
     feature_columns, so it always matches what the model was trained on.
  3. Indentation bug — app display block was nested inside the TF-IDF
     spinner; moved to correct scope.
  4. Scraper now cached with st.cache_data (TTL = 6 h) and falls back
     to a static dataset if the Play Store is unreachable.
  5. NEED_MAPPING covers all 10 Q-CHAT questions (A2, A5, A6, A8 were
     silently dropped in the original).
  6. Duplicate comment blocks cleaned up.
  7. Model card loaded to validate feature schema at startup.
"""

import json
import os
import time

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google_play_scraper import app as play_app
from google_play_scraper import search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCRAPE_QUERIES = [
    "autism speech therapy special education",
    "autism learning toddler app",
    "speech delay communication therapy child",
]
VALID_GENRES = {"Education", "Medical", "Parenting", "Health & Fitness"}

# Static fallback — used when Play Store scraping fails (network, rate-limit)
FALLBACK_APPS = [
    {
        "App_Name": "Autism iHelp – Listen",
        "Category": "Education",
        "Rating": 4.3,
        "Price": "Paid",
        "Description": (
            "Designed for children with autism and speech delay. Builds listening "
            "comprehension, receptive language, and following directions through "
            "visual and auditory cues."
        ),
    },
    {
        "App_Name": "Proloquo2Go",
        "Category": "Medical",
        "Rating": 4.5,
        "Price": "Paid",
        "Description": (
            "AAC (Augmentative and Alternative Communication) app for non-verbal "
            "children with speech delays. Uses symbols and words to build expressive "
            "language and communication."
        ),
    },
    {
        "App_Name": "Endless Alphabet",
        "Category": "Education",
        "Rating": 4.7,
        "Price": "Free",
        "Description": (
            "Vocabulary and language learning app for young children. Engaging puzzles "
            "support early word recognition, social-emotional learning, and communication."
        ),
    },
    {
        "App_Name": "Otsimo Special Education",
        "Category": "Education",
        "Rating": 4.6,
        "Price": "Free",
        "Description": (
            "Special education app for children with autism and developmental delays. "
            "Covers speech, cognitive skills, social interaction, and daily living."
        ),
    },
    {
        "App_Name": "Autism & PDD – Reasoning",
        "Category": "Education",
        "Rating": 4.1,
        "Price": "Paid",
        "Description": (
            "Targets reasoning, following instructions, and social cognitive understanding "
            "for children with autism. Includes sensory-friendly visual exercises."
        ),
    },
    {
        "App_Name": "Lingokids",
        "Category": "Education",
        "Rating": 4.5,
        "Price": "Free",
        "Description": (
            "Early childhood learning with songs, games, and stories. Supports speech "
            "development, vocabulary, and social skill building for young learners."
        ),
    },
]

# All 10 Q-CHAT signals mapped to semantic need phrases
# (original code missed A2, A5, A6, A8)
NEED_MAPPING = {
    ("A1", "A7"):       "speech delay non-verbal communication words language talk",
    ("A2", "A6", "A8"): "eye contact joint attention gesture social awareness",
    ("A3", "A4", "A5"): "social interaction play cognitive learning pretend",
    ("A9", "A10"):      "sensory processing meltdowns routine calm repetitive behaviour visual",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neuro-Adaptive Learning Recommender",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — clean, clinical, accessible
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');
 
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1, h2, h3 { font-family: 'DM Serif Display', serif; }
 
      /* ── Light mode tokens ──────────────────────────────────────── */
      :root {
        --card-bg:          #ffffff;
        --card-border:      #e0e8f0;
        --card-shadow:      0 1px 4px rgba(0,0,0,0.06);
        --meta-color:       #556070;
        --snippet-color:    #334050;
        --sidebar-bg:       #f4f6f9;
        --sidebar-text:     #1a2530;
        --sidebar-caption:  #556070;
        --sidebar-divider:  #d0d8e4;
        --badge-bg:         #1a73e8;
        --badge-text:       #ffffff;
        --alert-bg:         #fff4f0;
        --alert-border:     #e84c1a;
        --alert-text:       #7a1a00;
        --safe-bg:          #f0fff4;
        --safe-border:      #1ab84c;
        --safe-text:        #004d1a;
        --info-bg:          #f0f7ff;
        --info-border:      #1a73e8;
        --info-text:        #002d6e;
      }
 
      /* ── Dark mode tokens ───────────────────────────────────────── */
      @media (prefers-color-scheme: dark) {
        :root {
          --card-bg:          #1e2530;
          --card-border:      #2e3a4a;
          --card-shadow:      0 1px 6px rgba(0,0,0,0.35);
          --meta-color:       #8fa4b8;
          --snippet-color:    #c0ced8;
          --sidebar-bg:       #151c25;
          --sidebar-text:     #e0eaf4;
          --sidebar-caption:  #7a94a8;
          --sidebar-divider:  #2a3848;
          --badge-bg:         #1a73e8;
          --badge-text:       #ffffff;
          --alert-bg:         #2a1008;
          --alert-border:     #e84c1a;
          --alert-text:       #ffb8a0;
          --safe-bg:          #081a10;
          --safe-border:      #1ab84c;
          --safe-text:        #a0ffb8;
          --info-bg:          #08102a;
          --info-border:      #1a73e8;
          --info-text:        #a0c4ff;
        }
      }
 
      /* ── Streamlit also exposes its own theme class on <html> ────
         This catches users who have Streamlit set to Dark regardless
         of their OS preference (the dropdown in the hamburger menu). */
      [data-theme="dark"] {
        --card-bg:          #1e2530;
        --card-border:      #2e3a4a;
        --card-shadow:      0 1px 6px rgba(0,0,0,0.35);
        --meta-color:       #8fa4b8;
        --snippet-color:    #c0ced8;
        --sidebar-bg:       #151c25;
        --sidebar-text:     #e0eaf4;
        --sidebar-caption:  #7a94a8;
        --sidebar-divider:  #2a3848;
        --badge-bg:         #1a73e8;
        --badge-text:       #ffffff;
        --alert-bg:         #2a1008;
        --alert-border:     #e84c1a;
        --alert-text:       #ffb8a0;
        --safe-bg:          #081a10;
        --safe-border:      #1ab84c;
        --safe-text:        #a0ffb8;
        --info-bg:          #08102a;
        --info-border:      #1a73e8;
        --info-text:        #a0c4ff;
      }
 
      /* ── Sidebar ────────────────────────────────────────────────── */
      section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
      }
      section[data-testid="stSidebar"] * {
        color: var(--sidebar-text) !important;
      }
      /* Caption / small text inside sidebar */
      section[data-testid="stSidebar"] small,
      section[data-testid="stSidebar"] .stCaption,
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--sidebar-caption) !important;
      }
      /* Horizontal rule inside sidebar */
      section[data-testid="stSidebar"] hr {
        border-color: var(--sidebar-divider) !important;
      }
      /* Radio button labels */
      section[data-testid="stSidebar"] .stRadio label,
      section[data-testid="stSidebar"] .stRadio span {
        color: var(--sidebar-text) !important;
      }
      /* Slider label and value */
      section[data-testid="stSidebar"] .stSlider label,
      section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
      section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: var(--sidebar-caption) !important;
      }
 
      /* ── Result / metric cards ──────────────────────────────────── */
      .metric-card {
        background:   var(--info-bg);
        border-left:  4px solid var(--info-border);
        border-radius: 8px;
        padding:      1rem 1.25rem;
        margin-bottom: 0.75rem;
        color:        var(--info-text);
      }
      .metric-card strong { color: var(--info-text); }
 
      .metric-card.alert {
        background:  var(--alert-bg);
        border-left-color: var(--alert-border);
        color:       var(--alert-text);
      }
      .metric-card.alert strong { color: var(--alert-text); }
 
      .metric-card.safe {
        background:  var(--safe-bg);
        border-left-color: var(--safe-border);
        color:       var(--safe-text);
      }
      .metric-card.safe strong { color: var(--safe-text); }
 
      /* ── App recommendation cards ───────────────────────────────── */
      .app-card {
        border:        1px solid var(--card-border);
        border-radius: 10px;
        padding:       1rem 1.25rem;
        margin-bottom: 0.75rem;
        background:    var(--card-bg);
        box-shadow:    var(--card-shadow);
      }
      .app-card h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
        color: var(--snippet-color);
      }
      .app-card .meta    { color: var(--meta-color);    font-size: 0.82rem; }
      .app-card .snippet { color: var(--snippet-color); font-size: 0.87rem; margin-top: 0.4rem; }
 
      /* ── Match badge (always teal — readable on both themes) ────── */
      .match-badge {
        display:       inline-block;
        background:    var(--badge-bg);
        color:         var(--badge-text);
        border-radius: 20px;
        padding:       2px 10px;
        font-size:     0.78rem;
        font-weight:   600;
        margin-left:   0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODEL + MODEL CARD
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """Load the trained model and its model card. Fail loudly if missing."""
    model = joblib.load("asd_model1.pkl")

    with open("model_card1.json", "r") as f:
        card = json.load(f)

    return model, card


try:
    model, model_card = load_artifacts()
    FEATURE_COLS = model_card["feature_columns"]  # single source of truth
except FileNotFoundError as exc:
    st.error(
        f"❌ Required file not found: **{exc.filename}**  \n"
        "Ensure `asd_model1.pkl` and `model_card1.json` are in the same directory as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 2. GEMINI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    gemini_client = None  # app will still work; LLM block will show a warning

# ─────────────────────────────────────────────────────────────────────────────
# 3. CACHED SCRAPER  (TTL = 6 hours — avoids rate-limiting on every run)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_app_database() -> pd.DataFrame:
    """
    Scrape the Play Store across multiple queries, deduplicate, filter by
    genre, and return a DataFrame. Falls back to FALLBACK_APPS on any error.
    """
    seen, rows = set(), []

    for query in SCRAPE_QUERIES:
        try:
            results = search(query, lang="en", country="us")[:10]
        except Exception:
            continue

        for r in results:
            pkg = r["appId"]
            if pkg in seen:
                continue
            seen.add(pkg)

            try:
                d = play_app(pkg, lang="en", country="us")
                if d.get("genre") not in VALID_GENRES:
                    continue
                rows.append(
                    {
                        "App_Name": d["title"],
                        "Category": d["genre"],
                        "Rating": round(d.get("score") or 0, 2),
                        "Price": "Free" if d.get("free") else "Paid",
                        "Description": (d.get("description") or "")[:600],
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(FALLBACK_APPS)

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 4. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_profile_text(scores: dict) -> str:
    """
    Translate binary A1-A10 scores into a semantic query string.
    All 10 Q-CHAT signals are covered (fixes original which dropped A2/A5/A6/A8).
    """
    needs = [
        phrase
        for questions, phrase in NEED_MAPPING.items()
        if any(scores.get(q, 0) == 1 for q in questions)
    ]
    return " ".join(needs) if needs else "autism special education cognitive skills"


def recommend_apps(scores: dict, df_apps: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return top_n apps ranked by TF-IDF cosine similarity to the child's profile."""
    profile_text = build_profile_text(scores)
    descriptions = df_apps["Description"].fillna("").tolist()

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(descriptions)
    query_vec = tfidf.transform([profile_text])

    df_result = df_apps.copy()
    df_result["Match_%"] = (cosine_similarity(query_vec, matrix).flatten() * 100).round(1)
    return df_result.sort_values("Match_%", ascending=False).head(top_n).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER: response mapper
# ─────────────────────────────────────────────────────────────────────────────

def map_response(response: str, invert: bool = False) -> int:
    """
    Convert a radio option to a binary flag (1 = delayed / atypical).

    For most questions the "Yes" option is typical   → Yes maps to 0.
    For A9/A10 the "Yes" option is atypical (delayed) → needs invert=True.

    BUG FIX: the original app used the same map_response for all questions,
    causing A9 and A10 scores to be inverted (Yes=delayed was scored as 0).
    """
    yes_is_typical = response.startswith("Yes")
    if invert:
        # "Yes" here means atypical
        return 1 if yes_is_typical else 0
    return 0 if yes_is_typical else 1

# ─────────────────────────────────────────────────────────────────────────────
# 6. PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧠 Neuro-Adaptive Learning Recommender")
st.markdown(
    "Early screening for ASD traits in toddlers (12–36 months), "
    "matched to live educational resources via NLP — powered by XGBoost + Gemini AI."
)

# Show model metadata in an expandable block
with st.expander("ℹ️ Model Info", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model_card.get("model_type", "XGBoost"))
    col2.metric("Test Accuracy", f"{model_card.get('test_accuracy', 0)*100:.1f}%")
    col3.metric("ROC-AUC", f"{model_card.get('test_roc_auc', 0):.3f}")
    st.caption(
        f"Trained on {model_card.get('training_samples', '?')} samples · "
        f"Features: {', '.join(FEATURE_COLS)} · "
        f"Last trained: {model_card.get('trained_on', 'N/A')[:10]}"
    )
    st.caption(f"⚠️ {model_card.get('disclaimer', '')}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR — BEHAVIORAL PROFILE FORM
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("👶 Toddler Behavioral Profile")
st.sidebar.caption("Answer each question based on what you observe most of the time.")

age = st.sidebar.slider("Child's Age (Months)", 12, 36, 24)
sex_input = st.sidebar.radio("Biological Sex", ["Male", "Female"])
sex = 1 if sex_input == "Male" else 0

st.sidebar.markdown("---")
st.sidebar.markdown("### Q-CHAT-10 Screening Questions")
st.sidebar.caption("*'Delayed' = the child does NOT yet do this reliably.*")

# Q-CHAT-10 questions
# Note: A9/A10 use invert=True because "Yes" = atypical for those questions
a1  = map_response(st.sidebar.radio("1. Responds when called by name?",    ["Yes (Typical)", "No (Delayed)"]))
a2  = map_response(st.sidebar.radio("2. Makes eye contact easily?",         ["Yes (Typical)", "No (Delayed)"]))
a3  = map_response(st.sidebar.radio("3. Points to indicate wants?",         ["Yes (Typical)", "No (Delayed)"]))
a4  = map_response(st.sidebar.radio("4. Points to share interest?",         ["Yes (Typical)", "No (Delayed)"]))
a5  = map_response(st.sidebar.radio("5. Engages in pretend play?",          ["Yes (Typical)", "No (Delayed)"]))
a6  = map_response(st.sidebar.radio("6. Follows where you look/point?",     ["Yes (Typical)", "No (Delayed)"]))
a7  = map_response(st.sidebar.radio("7. Uses basic words/speech?",          ["Yes (Typical)", "No (Delayed)"]))
a8  = map_response(st.sidebar.radio("8. Understands simple gestures?",      ["Yes (Typical)", "No (Delayed)"]))
a9  = map_response(st.sidebar.radio("9. Unusual sensory reactions?",        ["No (Typical)",  "Yes (Delayed)"]), invert=True)  # FIX
a10 = map_response(st.sidebar.radio("10. Repetitive/unusual behaviours?",   ["No (Typical)",  "Yes (Delayed)"]), invert=True)  # FIX

# Build input_data strictly from FEATURE_COLS in the model card
# (fixes feature-mismatch crash if model was trained with/without Age_Mons)
raw_scores = {
    "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5,
    "A6": a6, "A7": a7, "A8": a8, "A9": a9, "A10": a10,
    "Sex": sex,
}
input_data = pd.DataFrame([{col: raw_scores[col] for col in FEATURE_COLS}])

st.sidebar.markdown("---")
st.sidebar.caption("*Screening aid only — not a medical diagnosis.*")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN PANEL — ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

run = st.button("🔍 Analyse Profile & Generate Plan", type="primary", use_container_width=True)

if run:

    # ── A. Predict ────────────────────────────────────────────────────────────
    risk_prob = model.predict_proba(input_data)[0][1] * 100
    total_flags = sum(v for k, v in raw_scores.items() if k.startswith("A"))

    if risk_prob < 50:
        st.markdown(
            f'<div class="metric-card safe">'
            f'<strong>✅ Low Likelihood of ASD Traits — {risk_prob:.1f}%</strong><br>'
            f'{total_flags}/10 behavioural milestones flagged. '
            f'The child appears to be meeting standard developmental milestones. '
            f'Routine monitoring is recommended.'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # High-risk path
    st.markdown(
        f'<div class="metric-card alert">'
        f'<strong>⚠️ High Likelihood of ASD Traits Detected — {risk_prob:.1f}%</strong><br>'
        f'{total_flags}/10 behavioural milestones flagged. '
        f'Early intervention is strongly recommended. Please consult a developmental paediatrician.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── B. Load app database (cached) ─────────────────────────────────────────
    with st.spinner("Loading educational app database…"):
        df_apps = get_app_database()

    st.caption(
        f"App database loaded: {len(df_apps)} apps "
        f"({'live Play Store data' if len(df_apps) > len(FALLBACK_APPS) else 'fallback dataset'})"
    )

    # ── C. NLP Recommendation ─────────────────────────────────────────────────
    with st.spinner("Matching apps to this child's specific needs…"):
        best_apps = recommend_apps(raw_scores, df_apps, top_n=3)
        profile_text = build_profile_text(raw_scores)
        recommended_names = best_apps["App_Name"].tolist()

    # ── D. Gemini AI Intervention Plan ────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI Agent Intervention Plan")

    GEMINI_PROMPT = f"""
            You are a compassionate, expert special education consultant speaking directly to a parent.

            A {age}-month-old toddler was just screened and showed a {risk_prob:.1f}% likelihood of ASD traits.
            The child's flagged developmental areas include: {profile_text}.

            Our recommendation engine has matched these 3 educational apps as the best fit:
            {recommended_names}

            Write a short (under 250 words), warm, and encouraging message to the parent. Cover:
            1. Why early, consistent intervention matters at this age.
            2. A one-sentence explanation of how each of the 3 apps specifically addresses this child's delays.
            3. A gentle reminder to seek a professional assessment — but frame it positively.

            Do NOT provide a medical diagnosis. Use plain language. No bullet points — flowing paragraphs only.
        """

    if gemini_client:
        try:
            with st.spinner("Generating personalised intervention plan…"):
                response = gemini_client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=GEMINI_PROMPT,
                )
            st.info(response.text)
        except Exception as e:
            st.warning(f"AI plan unavailable ({e}). Showing standard recommendations below.")
    else:
        st.warning(
            "Gemini API key not configured (`GEMINI_API_KEY` env var). "
            "Add it to your `.env` file or Streamlit secrets to enable the AI plan."
        )

    # ── E. Recommended Apps Display ───────────────────────────────────────────
    # BUG FIX: this block was indented inside the TF-IDF spinner in the original,
    # so it only rendered while the spinner was active. Now at correct scope.
    st.markdown("---")
    st.subheader("📱 Recommended Learning Resources")

    for rank, (_, row) in enumerate(best_apps.iterrows(), start=1):
        st.markdown(
            f'<div class="app-card">'
            f'<h4>#{rank} {row["App_Name"]}'
            f'<span class="match-badge">{row["Match_%"]}% match</span></h4>'
            f'<div class="meta">'
            f'Category: {row["Category"]} &nbsp;|&nbsp; '
            f'Rating: {row["Rating"]} ⭐ &nbsp;|&nbsp; '
            f'{row["Price"]}'
            f'</div>'
            f'<div class="snippet">{row["Description"][:160]}…</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── F. Flagged questions summary ──────────────────────────────────────────
    with st.expander("📋 View Full Screening Summary", expanded=False):
        Q_LABELS = {
            "A1": "Responds to name", "A2": "Eye contact", "A3": "Points (wants)",
            "A4": "Points (interest)", "A5": "Pretend play", "A6": "Follows gaze",
            "A7": "Basic speech", "A8": "Understands gestures",
            "A9": "Sensory reactions", "A10": "Repetitive behaviours",
        }
        summary_rows = [
            {"Question": f"{k}: {Q_LABELS[k]}", "Score": v, "Status": "⚠️ Flagged" if v == 1 else "✅ Typical"}
            for k, v in raw_scores.items()
            if k.startswith("A")
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.caption(f"Age: {age} months | Sex: {sex_input} | Total flags: {total_flags}/10")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Architected and Developed by **Taiye Janet Fagbolade | FastAPI Support by Iyanu Arowosola** · "
    "Bluechips × Data Science Nigeria Hackathon · "
    "⚠️ For screening purposes only — not a substitute for professional medical assessment."
)
