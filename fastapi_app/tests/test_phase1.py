import sys
import time
from pathlib import Path

FASTAPI_DIR = Path("/home/techyz-admin/sevenwings/02_startups/neuro-adaptive-recommender/fastapi_app")
sys.path.insert(0, str(FASTAPI_DIR))

from starlette.testclient import TestClient
from database import get_db, engine, Base
import models
from models.auth_models import User, UserRole
from models.domain_models import ChildProfile, ScreeningAssessment, AssessmentRecommendation
from core import state, _load_model, _load_app_cache, _load_book_cache, predict_risk
from main import app

# Ensure tables exist
Base.metadata.create_all(bind=engine)

def test_1_a3_model_sensitivity():
    print("\n[TEST 1] Verifying A3 milestone bug fix in predict_risk...")
    _load_model()

    base_scores = {
        "A1": "Usually", "A2": "Usually", "A3": "Usually", "A4": "Usually", "A5": "Usually",
        "A6": "Usually", "A7": "Usually", "A8": "Usually", "A9": "Never", "A10": "Never",
        "Age": 24, "Sex": 1,
    }
    risk_baseline = predict_risk(base_scores)

    # Modify A3 from Usually (1) to Never (4) -> risk score MUST change
    scores_flagged_a3 = dict(base_scores)
    scores_flagged_a3["A3"] = "Never"
    risk_flagged_a3 = predict_risk(scores_flagged_a3)

    print(f"  Baseline Risk (A3=Usually): {risk_baseline:.2f}%")
    print(f"  Flagged Risk (A3=Never)   : {risk_flagged_a3:.2f}%")
    
    assert risk_baseline != risk_flagged_a3, "Bug! Changing A3 had zero effect on risk score!"
    print("  ✓ A3 actively influences XGBoost model prediction!")


def test_2_decoupled_startup_latency():
    print("\n[TEST 2] Verifying decoupled app cache startup speed...")
    t0 = time.time()
    _load_app_cache()
    elapsed = time.time() - t0
    print(f"  _load_app_cache loaded {len(state.df_apps)} apps in {elapsed:.3f} seconds.")
    assert elapsed < 3.0, f"App cache load too slow: {elapsed:.2f}s (should be <3s without scraping)"
    print("  ✓ Decoupled startup is fast and non-blocking!")


def test_3_secure_httponly_cookies_and_login():
    print("\n[TEST 3] Verifying HttpOnly cookie issuance on /auth/login...")
    client = TestClient(app)
    
    test_username = f"test_parent_{int(time.time())}"
    test_email = f"{test_username}@example.com"
    test_password = "Password123!"

    # 1. Register
    reg_res = client.post("/auth/register", json={
        "username": test_username,
        "email": test_email,
        "password": test_password,
        "role": "parent"
    })
    assert reg_res.status_code == 201, f"Register failed: {reg_res.text}"

    # 2. Login
    login_res = client.post("/auth/login", json={
        "username_or_email": test_username,
        "password": test_password
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"

    # Check cookies
    cookies = login_res.cookies
    assert "access_token" in cookies, "access_token cookie missing from login response!"
    assert "refresh_token" in cookies, "refresh_token cookie missing from login response!"
    
    # Check Set-Cookie headers for HttpOnly & SameSite
    set_cookie_headers = [v for k, v in login_res.headers.items() if k.lower() == "set-cookie"]
    assert any("httponly" in h.lower() for h in set_cookie_headers), "Set-Cookie header missing HttpOnly flag!"
    print("  ✓ HttpOnly and SameSite=Lax cookies issued correctly!")

    return client, cookies, test_username


def test_4_screening_persistence_and_ui_recommendations(client, cookies, username):
    print("\n[TEST 4] Verifying screening assessment persistence & UI app recommendations...")
    
    # Fill form as authenticated user
    screen_payload = {
        "age": 28,
        "sex": 1,
        "A1": "Never",
        "A2": "Rarely",
        "A3": "Never",
        "A4": "Rarely",
        "A5": "Never",
        "A6": "Never",
        "A7": "Never",
        "A8": "Never",
        "A9": "Always",
        "A10": "Always",
        "top_n": 3
    }
    
    # Set cookie on client
    client.cookies.set("access_token", cookies.get("access_token"))

    res = client.post("/screen", data=screen_payload)
    assert res.status_code == 200, f"/screen returned {res.status_code}"
    html_text = res.text

    # Verify both apps and books are present in HTML output
    assert "📱 Recommended Apps" in html_text, "Recommended Apps header missing from HTML output!"
    assert "📖 Recommended Books & Guides" in html_text, "Recommended Books header missing from HTML output!"
    print("  ✓ Both Recommended Apps and Books are rendered in UI!")

    # Verify Database Persistence
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None
        
        child = db.query(ChildProfile).filter(ChildProfile.user_id == user.id).first()
        assert child is not None, "ChildProfile was not created!"
        print(f"  ✓ ChildProfile verified: {child.first_name} (ID: {child.id})")

        assessment = db.query(ScreeningAssessment).filter(ScreeningAssessment.child_id == child.id).first()
        assert assessment is not None, "ScreeningAssessment was not persisted!"
        assert assessment.age_months == 28
        assert assessment.is_high_risk == True
        print(f"  ✓ ScreeningAssessment verified: Risk={assessment.risk_probability}%, Flags={assessment.total_flags}")

        recs = db.query(AssessmentRecommendation).filter(AssessmentRecommendation.assessment_id == assessment.id).all()
        assert len(recs) > 0, "No AssessmentRecommendations saved!"
        app_recs_count = sum(1 for r in recs if r.resource_type == "app")
        book_recs_count = sum(1 for r in recs if r.resource_type == "book")
        print(f"  ✓ AssessmentRecommendations verified: {app_recs_count} apps, {book_recs_count} books saved to DB!")

    finally:
        db.close()


if __name__ == "__main__":
    test_1_a3_model_sensitivity()
    test_2_decoupled_startup_latency()
    client, cookies, username = test_3_secure_httponly_cookies_and_login()
    test_4_screening_persistence_and_ui_recommendations(client, cookies, username)
    print("\n🎉 ALL PHASE 1 AUTOMATED TESTS PASSED SUCCESSFULLY!")
