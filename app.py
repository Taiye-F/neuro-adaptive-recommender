import re
import os

def clean_html(text):
    return re.sub('<[^<]+?>', '', text)

import streamlit as st
import pandas as pd
import joblib
from google_play_scraper import search, app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import time
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# 1. PAGE CONFIGURATION & API SETUP
# =========================================================
st.set_page_config(page_title="Neuro-Adapt Learning Recommender", page_icon="🧠", layout="wide")

try:
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("Google API Key not found. Please configure Streamlit Secrets.")
    st.stop()

# =========================================================
# 2. LOAD THE MACHINE LEARNING MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load('asd_ordinal_top10_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'asd_ordinal_top10_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# =========================================================
# 3. APP UI HEADER
# =========================================================
st.title("🧠 Neuro-Adapt Learning Recommender")
st.write("""
This tool screens toddlers for ASD traits, scrapes live educational apps, and uses an AI Agent to deliver a personalized, empathetic intervention plan.
""")
st.markdown("---")

# =========================================================
# 4. USER INPUT (Sidebar Screening Tool - 5 Point Scale)
# =========================================================
st.sidebar.header("Toddler Behavioral Profile")
st.sidebar.write("Answer the questions below based on the child's typical behavior:")

# The 5-point Likert scale options
options_standard = ["Always", "Usually", "Sometimes", "Rarely", "Never"]
options_reverse = ["Never", "Rarely", "Sometimes", "Usually", "Always"]

# Helper functions to convert words to 0-4 numbers for the XGBoost model
def map_standard(response):
    # For positive behaviors (e.g., eye contact): Always=0 (Safe), Never=4 (High Risk)
    mapping = {"Always": 0, "Usually": 1, "Sometimes": 2, "Rarely": 3, "Never": 4}
    return mapping[response]

def map_reverse(response):
    # For negative behaviors (e.g., hand flapping): Never=0 (Safe), Always=4 (High Risk)
    mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Usually": 3, "Always": 4}
    return mapping[response]

age = st.sidebar.slider("Child's Age (Months)", 12, 48, 24)
sex_input = st.sidebar.radio("Biological Sex", ["Male", "Female"])
sex = 1 if sex_input == "Male" else 0

st.sidebar.markdown("### Behavioral Milestones")

# Standard Questions (We want the child to do these)
a1 = map_standard(st.sidebar.selectbox("1. Looks at you when called?", options_standard, index=1))
a2 = map_standard(st.sidebar.selectbox("2. Makes eye contact easily?", options_standard, index=1))
a3 = map_standard(st.sidebar.selectbox("3. Points to indicate wants?", options_standard, index=1))
a4 = map_standard(st.sidebar.selectbox("4. Points to share interest?", options_standard, index=1))
a5 = map_standard(st.sidebar.selectbox("5. Engages in pretend play?", options_standard, index=1))
a6 = map_standard(st.sidebar.selectbox("6. Follows where you look?", options_standard, index=1))
a7 = map_standard(st.sidebar.selectbox("7. Speaks basic words?", options_standard, index=1))
a8 = map_standard(st.sidebar.selectbox("8. Understands simple gestures?", options_standard, index=1))

# Reverse Questions (We don't want the child to do these often)
a9 = map_reverse(st.sidebar.selectbox("9. Unusual sensory reactions (e.g., staring at nothing)?", options_reverse, index=0))
a10 = map_reverse(st.sidebar.selectbox("10. Repetitive behaviors (e.g., hand flapping)?", options_reverse, index=0))

# Compile inputs exactly as the ordinal model expects them
input_data = pd.DataFrame({
    'qchat6recode': [a6], 'qchat2recode': [a2], 'qchat5recode': [a5], 
    'qchat4recode': [a4], 'qchat10recode': [a10], 'qchat15recode': [a1], 
    'qchat25recode': [a9], 'qchat11recode': [a7], 'qchat1recode': [a1], 
    'qchat17recode': [a8], 'age': [age], 'sex': [sex]
})

# =========================================================
# 5. EXECUTION: PREDICT -> SCRAPE -> RECOMMEND -> LLM AGENT
# =========================================================
try:
    risk_prob_raw = model.predict_proba(input_data)[0][1]
    risk_prob = risk_prob_raw * 100
except ValueError as e:
    st.error(f"Prediction Error: Feature mismatch. Please ensure input columns match model training. {e}")
    st.stop()

CHOSEN_THRESHOLD = 0.40  # tuned via precision-recall analysis; improves recall to 92.6% without losing precision

if risk_prob_raw < CHOSEN_THRESHOLD:
    st.success(f"✅ **Low Likelihood of ASD Traits ({risk_prob:.1f}%)**")
    st.write("The child is currently meeting standard developmental milestones. Routine monitoring is recommended.")
else:
    st.error(f"⚠️ **High Likelihood of ASD Traits Detected ({risk_prob:.1f}%)**")

    # --- B. Live Web Scraping (Finding Apps) ---
    with st.spinner("Scraping live educational apps from the Play Store..."):
        search_results = search("autism speech therapy special education", lang="en", country="us")
        top_apps = search_results[:50]  # increased from 12 -> wider candidate pool before filtering

        app_data = []
        for result in top_apps:
            try:
                app_details = app(result['appId'], lang='en', country='us')
                if app_details['genre'] in ['Education', 'Medical', 'Parenting']:
                    app_data.append({
                        'App_Name': app_details['title'],
                        'Category': app_details['genre'],
                        'Rating': round(app_details.get('score', 0), 2),
                        'Price': "Free" if app_details.get('free') else "Paid",
                        'Description': clean_html(app_details['description'])[:600],
                        'App_Link': app_details['url']
                    })
            except:
                continue
        df_apps = pd.DataFrame(app_data)

    # --- C. NLP Recommendation (TF-IDF Matching) ---
    with st.spinner("Matching apps to the child's specific behavioral needs..."):
        toddler_needs = []

        if a1 >= 3 or a7 >= 3: toddler_needs.append("speech delay non-verbal communication talk words articulation")
        if a3 >= 3 or a4 >= 3 or a6 >= 3: toddler_needs.append("social interaction play cognitive learning pointing joint attention")
        if a9 >= 3 or a10 >= 3: toddler_needs.append("sensory meltdowns routine calm visual behavior ADHD")
        if not toddler_needs: toddler_needs.append("autism special education cognitive skills")

        toddler_profile_text = " ".join(toddler_needs)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_apps['Description'].fillna("").tolist())
        toddler_vector = tfidf.transform([toddler_profile_text])

        similarity_scores = cosine_similarity(toddler_vector, tfidf_matrix).flatten()
        df_apps['Match_Score'] = (similarity_scores * 100).round(1)

        # --- Evidence-based tagging (CSV-driven, replaces old hardcoded set) ---
        EVIDENCE_DF = pd.read_csv('evidence_based_apps_2.csv')  # place this file next to your app.py
        EVIDENCE_LOOKUP = EVIDENCE_DF.set_index('app_name')[['evidence_tier', 'evidence_note']].to_dict('index')

        EVIDENCE_KEYWORDS = [
                "evidence-based", "evidence based", "clinically proven", "peer-reviewed",
                "randomized control trial", "randomized controlled trial", "rct",
                "developed with therapists", "developed with speech-language pathologists",
                "developed with slps", "aba-based", "aba based", "clinically validated",
                "developed by psychologists", "developed by clinicians", "research-backed",
                "backed by research", "clinical trial", "published study"
]

        def tag_evidence(row):
            if row['App_Name'] in EVIDENCE_LOOKUP:
                info = EVIDENCE_LOOKUP[row['App_Name']]
                return pd.Series([True, info['evidence_tier'], info['evidence_note']])
            desc = row['Description'].lower()
            if any(kw in desc for kw in EVIDENCE_KEYWORDS):
                return pd.Series([True, 'self_reported', 'App description claims evidence-based approach (unverified)'])
            return pd.Series([False, 'unverified', ''])

        df_apps[['Evidence_Based', 'Evidence_Tier', 'Evidence_Note']] = df_apps.apply(tag_evidence, axis=1)

        # --- Fallback tiering logic ---
        strict_matches = df_apps[
            (df_apps['Match_Score'] >= 30) & (df_apps['Evidence_Based'] == True)
        ].sort_values(by='Match_Score', ascending=False)

        if len(strict_matches) > 0:
            ranked_apps = strict_matches
            match_note = "✅ Strong, evidence-based matches for your child's profile."
        else:
            fallback_apps = df_apps[
                df_apps['Evidence_Based'] == True
            ].sort_values(by='Match_Score', ascending=False)

            if len(fallback_apps) > 0:
                ranked_apps = fallback_apps
                match_note = "⚠️ No apps cleared our 50% match threshold — showing the closest evidence-based matches instead."
            else:
                ranked_apps = df_apps.sort_values(by='Match_Score', ascending=False)
                match_note = "⚠️ No verified evidence-based apps found in this scrape — showing closest matches by relevance only. Please verify clinical backing independently before use."

        recommended_app_names = ranked_apps['App_Name'].head(3).tolist()

    # --- D. The LLM Agent (Personalizing the Output) ---
    with st.spinner("Generating empathetic intervention plan via LLM Agent..."):
        system_prompt = f"""
        You are a compassionate, expert special education consultant. 
        A parent's {age}-month-old toddler was just screened with a {risk_prob:.1f}% likelihood of ASD traits.
        The child struggles with: {toddler_profile_text}.
        Our system recommends these apps: {recommended_app_names}.
        
        Write a highly empathetic, encouraging message to the parent. Explain gently why early intervention matters, and briefly explain how those apps will help their child's specific needs. Do not provide medical diagnosis. Maximum 200 words.
        """

        try:
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=system_prompt
            )
            st.markdown("### 🤖 Your Personalized Agent Analysis")
            st.info(response.text)
        except Exception as e:
            st.warning(f"LLM Agent unavailable: {e}")

    # --- E. Display the Recommended Apps UI (With Top 1/3/5 toggle) ---
    st.markdown("### 📱 Recommended Live Resources")
    #st.caption(match_note)

    #view_choice = st.radio(
       # "How many recommendations would you like to see?",
        #options=["Top 1", "Top 3", "Top 5"],
        #index=1,
        #horizontal=True
    #)
    #n_results = {"Top 1": 1, "Top 3": 3, "Top 5": 5}[view_choice]

    display_apps = ranked_apps.head(3)

    tier_labels = {
        'evidence_based_method': '🧪 Built on evidence-based method',
        'self_reported': 'ℹ️ Claims evidence-based (unverified)',
        'unverified': '❓ Not yet verified'
    }

    for _, row in display_apps.iterrows():
        with st.container():
            evidence_tag = tier_labels.get(row['Evidence_Tier'], '❓ Not yet verified')
            st.markdown(f"#### ⭐ [{row['App_Name']}]({row['App_Link']}) ")
                        #(Match: {row['Match_Score']}%)
            st.caption(f"{evidence_tag} | Category: {row['Category']} | Rating: {row['Rating']}⭐ | Price: {row['Price']}")
            st.write(st.write(f"_{clean_html(row['Description'])[:150]}..._"))

st.markdown("---")
st.caption("Architected and Deployed by Taiye Janet Fagbolade | FastAPI Support by Iyanu Arowosola")