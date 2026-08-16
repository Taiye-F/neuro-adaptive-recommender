# Neuro-Adaptive ASD Recommender (NALR)

**Bridging the Diagnostic Gap: AI-Powered Early Intervention for Toddlers**

An intelligent screening and recommendation system that helps parents and caregivers identify Autism Spectrum Disorder (ASD) traits in toddlers (12–48 months) and receive **personalized educational app recommendations** with **contextual AI guidance**.

---

## Project Overview

The Neuro-Adaptive ASD Recommender combines machine learning, semantic matching, and generative AI to deliver:

* Accurate *  *ASD risk prediction*  * using the validated *  *Q-CHAT-10* * screening tool.
* *  *Personalized app recommendations* * based on the child's specific behavioral profile.
* *  *Contextual AI Chat* * powered by Google Gemini for parent-friendly explanations and guidance.

---

## Key Features

* Interactive Q-CHAT-10 behavioral screening
* Real-time ASD risk prediction (XGBoost)
* Semantic app recommendations (TF-IDF + Cosine Similarity)
* *  *Google Gemini-powered contextual chat* * — explains recommendations and answers parent questions
* Modern *  *FastAPI* * backend with RESTful endpoints
* Responsive *  *Streamlit* * frontend with dark mode support
* Clean separation of concerns (FastAPI microservice + UI)

---

## Tech Stack

* *  *Backend* *: FastAPI, Uvicorn
* *  *ML* *: XGBoost, scikit-learn, pandas, TF-IDF + Cosine Similarity
* *  *Generative AI* *: Google Gemini
* *  *Frontend* *: Streamlit (alternative UI)
* *  *Others* *: Pydantic, Jinja2, Joblib

---

## Project Structure

```bash
neuro-adaptive-recommender/
├── fastapi_app/                  # Main FastAPI backend (recommended for production/hackathon submission)
│   ├── files/                    # Contains data files
│   ├── templates/                # Jinja2 HTML templates (index.html, results.html, etc.)
│   ├── core.py                   # Core logic: model loading, helpers, TF-IDF, Gemini functions
│   ├── main.py                   # FastAPI app entry point + lifespan + root routes
│   ├── schemas.py                # Pydantic models (requests & responses)
│   └── routers.py                # All API endpoints (health, predict, recommend, result, chat)
│
├── requirements.txt              # Dependencies for FastAPI backend
├── notebook/                    # Jupyter notebooks for model training & EDA
├── streamlit_app/                # Alternative frontend (Streamlit version)
│   ├── autism_app.py
│   └── autism_streamlit_app.py   # Improved version with better dark mode & UX
├── visualizations/               # Charts, screenshots, and model evaluation images
├── .gitignore
└── README.md                     # Project documentation (should be updated)
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Taiye-F/neuro-adaptive-recommender.git
cd neuro-adaptive-recommender
```

### 2. Create a virtual environemnt and activate

```bash
python -m venv .venv
```
On windows
```
.venv\scripts\activate
```
On Mac
```
source .venv\bin\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root:
```env
GEMINI_API_KEY=

```

### 4. Run the FastAPI Backend

```bash
uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:
* *  *Web UI* *: http://localhost:8000
* *  *Swagger Docs* *: http://localhost:8000/docs

### 5. (Optional) Run Streamlit Frontend

```bash
streamlit run autism_streamlit_app.py
```

---

## API Endpoints

| Method   | Path                      | Description                       |
| -------- | ------------------------- | --------------------------------- |
| GET      | `/` | Screening form (Web UI)           |
| POST     | `/screen` | Submit form → render results      |
| GET      | `/health` | System health check               |
| GET      | `/apps` | List all cached apps              |
| POST     | `/predict` | ASD risk probability only         |
| POST     | `/recommend` | Risk + ranked app recommendations |
| POST     | `/chat` | Contextual chat with Gemini       |

---

## Major Contributions & Updates

* Migrated core logic to *  *FastAPI* * microservice architecture
* Added *  *Gemini-powered contextual chat* * for personalized parent support
* Improved dark mode support in Streamlit UI
* Refactored into clean layered structure ( `main `, ` core `, ` routers` )
* Better error handling and UTF-8 support for data files
* Enhanced model transparency with `model_card.json`

---

## Authors & Collaborators

* *  *Taiye Janet Fagbolade* * — Principal Investigator, Machine Learning Engineering (XGBoost/ NLP), Cloud Deployment, and UI/UX.
* *  *Iyanu Arowosola* * — Technical Contributor (FastAPI integration)
 
Disclaimer: NALR is a digital triage and recommendation tool, not a substitute for a formal medical diagnosis. Always consult a licensed pediatric neurologist or healthcare professional.
