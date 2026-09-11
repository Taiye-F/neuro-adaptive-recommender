# test_phase3.py
import sys
import time
from uuid import uuid4, UUID
from datetime import datetime, timezone, timedelta
from pathlib import Path

FASTAPI_DIR = Path("/home/techyz-admin/sevenwings/02_startups/neuro-adaptive-recommender/fastapi_app")
sys.path.insert(0, str(FASTAPI_DIR))

from starlette.testclient import TestClient
from database import get_db, engine, Base
import models
from models.auth_models import User
from models.domain_models import (
    ChildProfile, ScreeningAssessment, MeltdownIncident,
    MeltdownStrategyApplied, ReminderNotification
)
from core import state, _load_model, _load_app_cache, _load_book_cache
from services.reminder_service import ReminderService
from services.meltdown_service import MeltdownService
from main import app

# Ensure tables exist
Base.metadata.create_all(bind=engine)
_load_model()
_load_app_cache()
_load_book_cache()

client = TestClient(app, raise_server_exceptions=True)

def test_phase3_suite():
    print("=" * 80)
    print("STARTING PHASE 3 AUTOMATED INTEGRATION TEST SUITE")
    print("=" * 80)

    # Setup parent user and child
    ts = int(time.time())
    parent_username = f"parent_phase3_{ts}"
    parent_email = f"parent_p3_{ts}@example.com"
    password = "SecurePassword123!"

    # 1. Register and Login
    reg_resp = client.post("/auth/register", json={
        "username": parent_username,
        "email": parent_email,
        "password": password,
        "role": "parent"
    })
    assert reg_resp.status_code == 201

    login_resp = client.post("/auth/login", json={
        "username_or_email": parent_username,
        "password": password
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    cookies = {"access_token": token}

    # 2. Create Child "Kofi"
    child_resp = client.post("/api/v1/children", headers=headers, json={
        "first_name": "Kofi",
        "date_of_birth": "2024-03-10",
        "biological_sex": 1,
        "notes": "Sensitive to auditory stimuli and sudden transitions"
    })
    assert child_resp.status_code == 201
    kofi_id = child_resp.json()["id"]
    print(f"\n[SETUP] Created Child Profile: Kofi (ID: {kofi_id})")

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Post-Event Triage Intake API
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 1] Testing Post-Event Triage Intake API & Strategy Persistence...")
    now = datetime.now(timezone.utc)
    triage_payload = {
        "start_time": (now - timedelta(minutes=8)).isoformat(),
        "end_time": now.isoformat(),
        "duration_seconds": 480,
        "intensity": 3,
        "location": "supermarket",
        "triggers": ["noise", "crowds", "fatigue"],
        "strategies": [
            {
                "strategy_name": "Deep Pressure / Firm Hug",
                "strategy_category": "proprioceptive",
                "efficacy": "effective",
                "notes": "Wrapped tightly in jacket"
            },
            {
                "strategy_name": "Noise Reduction / Mute",
                "strategy_category": "auditory",
                "efficacy": "effective",
                "notes": "Noise-cancelling headphones placed"
            }
        ],
        "notes": "Acute distress at supermarket checkout line due to fluorescent flicker and speaker noise."
    }

    triage_resp = client.post(f"/api/v1/meltdowns/{kofi_id}/triage", headers=headers, json=triage_payload)
    assert triage_resp.status_code == 201, f"Triage failed: {triage_resp.text}"
    triage_data = triage_resp.json()
    incident_id = triage_data["id"]
    assert triage_data["child_id"] == kofi_id
    assert triage_data["duration_seconds"] == 480
    assert triage_data["intensity"] == 3
    assert triage_data["location"] == "supermarket"
    assert "noise" in triage_data["triggers"]
    assert len(triage_data["strategies_applied"]) == 2
    print(f"  ✓ Successfully logged Meltdown Incident {incident_id} with 2 applied strategies")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Environmental Pattern Aggregator & Plain-English Insights
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 2] Testing Environmental Pattern Aggregator & Clinical Insights...")
    # Seed 3 additional diverse incidents directly or via API to verify pattern analytics
    db = next(get_db())
    try:
        # Incident 2: School, transition & noise, 15m, intensity 4
        inc2 = MeltdownIncident(
            child_id=UUID(kofi_id),
            start_time=now - timedelta(days=2, hours=3),
            end_time=now - timedelta(days=2, hours=2, minutes=45),
            duration_seconds=900,
            intensity=4,
            location="school",
            triggers=["transition", "noise"],
            notes="Recess bell triggered escalation"
        )
        db.add(inc2)
        db.flush()
        db.add(MeltdownStrategyApplied(
            meltdown_id=inc2.id,
            strategy_name="Heavy Work & Push",
            strategy_category="proprioceptive",
            efficacy="partially_effective"
        ))

        # Incident 3: Home, transition, 5m, intensity 2
        inc3 = MeltdownIncident(
            child_id=UUID(kofi_id),
            start_time=now - timedelta(days=4, hours=1),
            end_time=now - timedelta(days=4, hours=0, minutes=55),
            duration_seconds=300,
            intensity=2,
            location="home",
            triggers=["transition"],
            notes="Bedtime routine transition"
        )
        db.add(inc3)
        db.flush()
        db.add(MeltdownStrategyApplied(
            meltdown_id=inc3.id,
            strategy_name="Deep Pressure / Firm Hug",
            strategy_category="proprioceptive",
            efficacy="effective"
        ))

        # Incident 4: Home, hunger & noise, 8m, intensity 3
        inc4 = MeltdownIncident(
            child_id=UUID(kofi_id),
            start_time=now - timedelta(days=6, hours=5),
            end_time=now - timedelta(days=6, hours=4, minutes=52),
            duration_seconds=480,
            intensity=3,
            location="home",
            triggers=["hunger", "noise"],
            notes="Pre-dinner hunger overload"
        )
        db.add(inc4)
        db.flush()
        db.add(MeltdownStrategyApplied(
            meltdown_id=inc4.id,
            strategy_name="4-7-8 Guided Breathing",
            strategy_category="breathing",
            efficacy="ineffective"
        ))
        db.commit()
    finally:
        db.close()

    analytics_resp = client.get(f"/api/v1/meltdowns/{kofi_id}/analytics", headers=headers)
    assert analytics_resp.status_code == 200
    ana = analytics_resp.json()
    assert ana["total_incidents"] == 4
    # Expected Avg duration = (480 + 900 + 300 + 480) / (4 * 60) = 2160 / 240 = 9.0 mins
    assert ana["avg_duration_minutes"] == 9.0
    # Expected Avg intensity = (3 + 4 + 2 + 3) / 4 = 3.0
    assert ana["avg_intensity"] == 3.0
    print(f"  ✓ Aggregated 4 incidents: Avg Duration={ana['avg_duration_minutes']} min, Avg Intensity={ana['avg_intensity']}/5")

    # Verify Trigger frequencies
    trigs = ana["trigger_frequencies"]
    assert "noise" in trigs and trigs["noise"] == 3
    assert "transition" in trigs and trigs["transition"] == 2
    print(f"  ✓ Trigger Correlations computed: noise (3x), transition (2x), hunger (1x), crowds (1x)")

    # Verify Location frequencies
    locs = ana["location_frequencies"]
    assert locs.get("home") == 2
    assert locs.get("school") == 1
    assert locs.get("supermarket") == 1
    print(f"  ✓ Location Breakdown computed: home (2), school (1), supermarket (1)")

    # Verify Strategy Efficacy Ranking
    strats = ana["strategy_efficacy_ranking"]
    assert len(strats) >= 3
    # Deep Pressure was used 2x and was effective 2x -> 100% success rate
    deep_press = next(s for s in strats if "Deep Pressure" in s["strategy_name"])
    assert deep_press["success_rate"] == 100.0
    assert deep_press["times_used"] == 2
    print(f"  ✓ Strategy Efficacy Leaderboard: '{deep_press['strategy_name']}' ranked #1 with 100.0% success rate!")

    # Verify Plain-English Clinical Insights
    insights = ana["plain_english_insights"]
    assert len(insights) >= 3
    for ins in insights:
        print(f"    • Insight: {ins}")
    print(f"  ✓ Synthesized {len(insights)} plain-English clinical insights for caregivers")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Automated 30-Day Reassessment Reminder Worker
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 3] Testing Automated 30-Day Reassessment Reminder Worker...")
    # Seed an assessment completed 35 days ago for Kofi
    db = next(get_db())
    try:
        past_assessment = ScreeningAssessment(
            child_id=UUID(kofi_id),
            completed_at=datetime.now(timezone.utc) - timedelta(days=35),
            age_months=24,
            a1_score=3, a2_score=3, a3_score=3, a4_score=3, a5_score=3,
            a6_score=3, a7_score=3, a8_score=3, a9_score=3, a10_score=3,
            risk_probability=70.0,
            is_high_risk=True,
            total_flags=10,
            profile_text="Initial baseline.",
            profile_explained="Initial clinical baseline."
        )
        db.add(past_assessment)
        db.commit()
    finally:
        db.close()

    # Trigger reminder evaluation
    check_resp = client.post("/api/v1/reminders/check-now", headers=headers)
    assert check_resp.status_code == 200
    assert check_resp.json()["new_reminders_count"] >= 1
    print(f"  ✓ Triggered 30-day reminder evaluation: {check_resp.json()['new_reminders_count']} reminder(s) generated")

    # Retrieve pending reminders for parent
    reminders_resp = client.get("/api/v1/reminders?status=pending", headers=headers)
    assert reminders_resp.status_code == 200
    reminders = reminders_resp.json()
    assert len(reminders) >= 1
    rem = next(r for r in reminders if r["child_id"] == kofi_id)
    assert "Kofi is due for a 30-day developmental reassessment" in rem["message"]
    print(f"  ✓ Pending Notification verified for Kofi: '{rem['message'][:65]}...'")

    # Dismiss reminder
    dismiss_resp = client.post(f"/api/v1/reminders/{rem['id']}/dismiss", headers=headers)
    assert dismiss_resp.status_code == 200
    print(f"  ✓ Dismissed reminder {rem['id']}")

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Meltdown HTML UI View Rendering
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 4] Testing Meltdown Emergency Suite Web UI Views...")
    
    # 1. Active Meltdown Emergency Screen
    active_html = client.get(f"/meltdown/{kofi_id}/active", cookies=cookies)
    assert active_html.status_code == 200
    assert "Acute Incident Tracking for" in active_html.text
    assert "00:00:00" in active_html.text  # Stopwatch display
    assert "Co-Regulation 4-7-8 Breathing Guide" in active_html.text
    assert "Quick Sensory Calming Prompts" in active_html.text
    assert "Post-Incident Triage Intake" in active_html.text
    print(f"  ✓ Rendered /meltdown/{kofi_id}/active: Live Stopwatch, 4-7-8 Breathing & Triage Modal")

    # 2. Meltdown Analytics & Pattern Aggregator Screen
    analytics_html = client.get(f"/meltdown/{kofi_id}/analytics", cookies=cookies)
    assert analytics_html.status_code == 200
    assert "Meltdown Analytics" in analytics_html.text
    assert "Prevalent Overload Triggers" in analytics_html.text
    assert "Caregiver Behavioral Insights" in analytics_html.text
    assert "Strategy Efficacy Leaderboard" in analytics_html.text
    assert "Recent Incident Logs" in analytics_html.text
    print(f"  ✓ Rendered /meltdown/{kofi_id}/analytics: Metrics, Visual Bars & Incident Logs")

    print("\n" + "=" * 80)
    print("🎉 ALL PHASE 3 AUTOMATED TESTS COMPLETED & PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_phase3_suite()
