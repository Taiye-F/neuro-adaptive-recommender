# test_phase2.py
import sys
import time
from uuid import uuid4, UUID
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

FASTAPI_DIR = Path("/home/techyz-admin/sevenwings/02_startups/neuro-adaptive-recommender/fastapi_app")
sys.path.insert(0, str(FASTAPI_DIR))

from starlette.testclient import TestClient
from database import get_db, engine, Base
import models
from models.auth_models import User, UserRole
from models.domain_models import (
    ChildProfile, ScreeningAssessment, AssessmentRecommendation,
    VisualSchedule, ScheduleTask, TaskCompletion
)
from core import state, _load_model, _load_app_cache, _load_book_cache
from services.schedule_service import ScheduleService
from main import app

# Ensure tables exist
Base.metadata.create_all(bind=engine)
_load_model()
_load_app_cache()
_load_book_cache()

client = TestClient(app, raise_server_exceptions=True)

def test_phase2_suite():
    print("=" * 80)
    print("STARTING PHASE 2 AUTOMATED INTEGRATION TEST SUITE")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Child Profile Multi-Child CRUD & Authorization
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 1] Testing Multi-Child Profile CRUD & RBAC Security...")
    ts = int(time.time())
    parent_username = f"parent_phase2_{ts}"
    parent_email = f"parent_{ts}@example.com"

    reg_resp = client.post("/auth/register", json={
        "username": parent_username,
        "email": parent_email,
        "password": "SecurePassword123!",
        "role": "parent"
    })
    assert reg_resp.status_code == 201, f"Register failed: {reg_resp.text}"

    login_resp = client.post("/auth/login", json={
        "username_or_email": parent_username,
        "password": "SecurePassword123!"
    })
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    cookies = {"access_token": token}

    # Create Child 1: Maya
    create_maya = client.post("/api/v1/children", headers=headers, json={
        "first_name": "Maya",
        "date_of_birth": "2024-06-15",
        "biological_sex": 0,
        "notes": "Loves music and tactile toys"
    })
    assert create_maya.status_code == 201, f"Failed creating Maya: {create_maya.text}"
    maya_data = create_maya.json()
    maya_id = maya_data["id"]
    assert maya_data["first_name"] == "Maya"
    assert maya_data["biological_sex"] == 0
    print(f"  ✓ Created Child Profile: Maya (ID: {maya_id})")

    # Create Child 2: Leo
    create_leo = client.post("/api/v1/children", headers=headers, json={
        "first_name": "Leo",
        "date_of_birth": "2023-11-20",
        "biological_sex": 1,
        "notes": "Fascinated by moving gears"
    })
    assert create_leo.status_code == 201
    leo_id = create_leo.json()["id"]
    print(f"  ✓ Created Child Profile: Leo (ID: {leo_id})")

    # Retrieve all children
    list_resp = client.get("/api/v1/children", headers=headers)
    assert list_resp.status_code == 200
    children = list_resp.json()
    assert len(children) >= 2
    child_ids = [c["id"] for c in children]
    assert maya_id in child_ids and leo_id in child_ids
    print(f"  ✓ Listed {len(children)} child profiles for {parent_username}")

    # Test RBAC / Data Isolation: Stranger cannot access Maya
    stranger_reg = client.post("/auth/register", json={
        "username": f"stranger_{ts}",
        "email": f"stranger_{ts}@example.com",
        "password": "SecurePassword123!",
        "role": "parent"
    })
    stranger_login = client.post("/auth/login", json={
        "username_or_email": f"stranger_{ts}",
        "password": "SecurePassword123!"
    })
    stranger_token = stranger_login.json()["access_token"]
    stranger_headers = {"Authorization": f"Bearer {stranger_token}"}

    forbidden_resp = client.get(f"/api/v1/children/{maya_id}", headers=stranger_headers)
    assert forbidden_resp.status_code == 403, f"Expected 403 Forbidden, got {forbidden_resp.status_code}"
    print(f"  ✓ Verified Cross-User Isolation: Stranger access blocked with 403 Forbidden")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Trajectory Engine API & Milestone Transition Shifts
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 2] Testing Trajectory Engine API & Milestone Transitions...")
    
    # 0 assessments state
    traj_0 = client.get(f"/api/v1/journey/{maya_id}/trajectory", headers=headers)
    assert traj_0.status_code == 200
    assert traj_0.json()["total_assessments"] == 0
    assert traj_0.json()["trend_direction"] == "insufficient_data"
    print(f"  ✓ 0-Assessment empty state verified (trend_direction: insufficient_data)")

    # Seed Assessment T1 (Baseline: high risk)
    db = next(get_db())
    try:
        t1 = ScreeningAssessment(
            child_id=UUID(maya_id),
            completed_at=datetime.now(timezone.utc) - timedelta(days=60),
            age_months=24,
            a1_score=4, a2_score=4, a3_score=4, a4_score=4, a5_score=4,
            a6_score=4, a7_score=4, a8_score=4, a9_score=4, a10_score=4,
            risk_probability=85.5,
            is_high_risk=True,
            total_flags=10,
            profile_text="High risk baseline screening.",
            profile_explained="Focus on eye contact and pointing."
        )
        db.add(t1)
        db.flush()
        db.add(AssessmentRecommendation(
            assessment_id=t1.id,
            resource_type="app",
            item_name="Otsimo Special Education",
            category="Communication",
            match_score=94.5,
            rank=1
        ))
        db.commit()
    finally:
        db.close()

    traj_1 = client.get(f"/api/v1/journey/{maya_id}/trajectory", headers=headers)
    assert traj_1.status_code == 200
    d1 = traj_1.json()
    assert d1["total_assessments"] == 1
    assert d1["baseline_risk"] == 85.5
    assert d1["current_risk"] == 85.5
    assert d1["risk_delta"] == 0.0
    assert d1["trend_direction"] == "stable"
    assert len(d1["assessments_timeline"]) == 1
    print(f"  ✓ 1-Assessment baseline verified (Baseline Risk: 85.5%, Delta: 0.0%)")

    # Seed Assessment T2 (Follow-up: 30 days later, improved!)
    db = next(get_db())
    try:
        t2 = ScreeningAssessment(
            child_id=UUID(maya_id),
            completed_at=datetime.now(timezone.utc) - timedelta(days=1),
            age_months=25,
            # A1, A2, A3 improved to score 0 (typical)
            a1_score=0, a2_score=0, a3_score=0, a4_score=4, a5_score=4,
            a6_score=4, a7_score=4, a8_score=4, a9_score=0, a10_score=0,
            risk_probability=35.0,
            is_high_risk=False,
            total_flags=5,
            profile_text="Follow-up screening showing significant gains in joint attention.",
            profile_explained="Great progress in social interaction."
        )
        db.add(t2)
        db.flush()
        db.add(AssessmentRecommendation(
            assessment_id=t2.id,
            resource_type="app",
            item_name="Proloquo2Go",
            category="Speech & Language",
            match_score=90.0,
            rank=1
        ))
        db.commit()
    finally:
        db.close()

    traj_2 = client.get(f"/api/v1/journey/{maya_id}/trajectory", headers=headers)
    assert traj_2.status_code == 200
    d2 = traj_2.json()
    assert d2["total_assessments"] == 2
    assert d2["baseline_risk"] == 85.5
    assert d2["current_risk"] == 35.0
    # Expected Risk Delta = 35.0 - 85.5 = -50.5
    assert d2["risk_delta"] == -50.5
    assert d2["trend_direction"] == "improving"
    print(f"  ✓ Trajectory Delta calculated correctly: Delta = {d2['risk_delta']}% (improving)")

    # Verify Milestone Transitions
    shifts = {item["code"]: item["status"] for item in d2["milestone_shifts"]}
    assert shifts["A1"] == "Resolved", f"Expected A1 to be Resolved, got {shifts['A1']}"
    assert shifts["A2"] == "Resolved", f"Expected A2 to be Resolved, got {shifts['A2']}"
    assert shifts["A3"] == "Resolved", f"Expected A3 to be Resolved, got {shifts['A3']}"
    assert shifts["A4"] == "Ongoing Focus Area", f"Expected A4 to be Ongoing Focus Area, got {shifts['A4']}"
    print(f"  ✓ Milestone Transition Shifts verified: A1, A2, A3 = 'Resolved', A4 = 'Ongoing Focus Area'")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Visual Schedules & Daily Streak Engine
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 3] Testing Visual Schedules, Task Completions & Streaks...")
    
    # Get child schedules (auto-provisions default morning routine)
    sched_resp = client.get(f"/api/v1/schedules/{maya_id}", headers=headers)
    assert sched_resp.status_code == 200
    schedules = sched_resp.json()
    assert len(schedules) >= 1
    morning_sched = schedules[0]
    assert morning_sched["title"] == "Daily Morning Routine"
    assert len(morning_sched["tasks"]) == 5
    task1_id = morning_sched["tasks"][0]["id"]
    print(f"  ✓ Default schedule auto-provisioned with {len(morning_sched['tasks'])} tasks")

    # Add a custom task
    add_task_resp = client.post(
        f"/api/v1/schedules/{morning_sched['id']}/tasks",
        headers=headers,
        json={"title": "Sensory Calming Corner", "icon_key": "sensory"}
    )
    assert add_task_resp.status_code == 201
    new_task = add_task_resp.json()
    new_task_id = new_task["id"]
    print(f"  ✓ Added custom task: '{new_task['title']}' (ID: {new_task_id})")

    # Toggle task completion (check off)
    toggle_on = client.post(f"/api/v1/schedules/tasks/{new_task_id}/toggle", headers=headers)
    assert toggle_on.status_code == 200
    res_on = toggle_on.json()
    assert res_on["completed_today"] is True
    assert res_on["reward_animation"] == "stars_confetti"
    assert res_on["streak_days"] >= 1
    print(f"  ✓ Checked off task: completed_today=True, reward={res_on['reward_animation']}, streak={res_on['streak_days']}")

    # Toggle task off (uncheck)
    toggle_off = client.post(f"/api/v1/schedules/tasks/{new_task_id}/toggle", headers=headers)
    assert toggle_off.status_code == 200
    res_off = toggle_off.json()
    assert res_off["completed_today"] is False
    print(f"  ✓ Unchecked task: completed_today=False")

    # Test Streak Multi-Day Calculation
    db = next(get_db())
    try:
        today = datetime.now(timezone.utc).date()
        # Seed completions for today, yesterday, and 2 days ago
        for days_back in [0, 1, 2]:
            db.add(TaskCompletion(
                task_id=UUID(task1_id),
                completion_date=today - timedelta(days=days_back),
                rewarded=True
            ))
        db.commit()
    finally:
        db.close()

    calculated_streak = ScheduleService.calculate_streak(db, UUID(maya_id))
    assert calculated_streak >= 3, f"Expected streak >= 3, got {calculated_streak}"
    print(f"  ✓ Consecutive Streak Calculator verified: {calculated_streak} active consecutive days!")

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Clinical PDF Engine (WeasyPrint)
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 4] Testing Clinical WeasyPrint PDF Generation & Streaming...")
    pdf_resp = client.get(f"/api/v1/journey/{maya_id}/report.pdf", headers=headers)
    assert pdf_resp.status_code == 200, f"PDF generation failed: {pdf_resp.status_code} {pdf_resp.text}"
    assert pdf_resp.headers["Content-Type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-"), "Invalid PDF binary header!"
    assert len(pdf_resp.content) > 10000, f"PDF file too small ({len(pdf_resp.content)} bytes)"
    print(f"  ✓ Generated valid clinical pediatric PDF report ({len(pdf_resp.content):,} bytes) with '%PDF-' magic bytes!")

    # Also test PDF download using cookie auth
    cookie_pdf_resp = client.get(f"/api/v1/journey/{maya_id}/report.pdf", cookies=cookies)
    assert cookie_pdf_resp.status_code == 200
    assert cookie_pdf_resp.content.startswith(b"%PDF-")
    print(f"  ✓ Verified Cookie-based browser PDF download without Authorization header!")

    # ─────────────────────────────────────────────────────────────
    # TEST 5: Phase 2 HTML Template Rendering
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 5] Testing Phase 2 HTML UI Views & SVG Timeline...")
    
    # 1. Journey Loop HTML
    journey_html = client.get(f"/journey/{maya_id}", cookies=cookies)
    assert journey_html.status_code == 200
    assert "Maya's Journey Loop" in journey_html.text
    assert "<svg" in journey_html.text  # SVG Timeline component
    assert "Resolved" in journey_html.text
    print(f"  ✓ Rendered /journey/{maya_id} with interactive SVG timeline component")

    # 2. Visual Schedule HTML
    schedule_html = client.get(f"/schedule/{maya_id}", cookies=cookies)
    assert schedule_html.status_code == 200
    assert "Maya's Visual Schedule" in schedule_html.text
    assert "Daily Morning Routine" in schedule_html.text
    assert "Streak" in schedule_html.text
    print(f"  ✓ Rendered /schedule/{maya_id} with routine cards & daily streak counters")

    # 3. Clinical Report Browser Print View HTML
    print_html = client.get(f"/journey/{maya_id}/print", cookies=cookies)
    assert print_html.status_code == 200
    assert "Neuro-Adaptive Pediatric ASD Summary" in print_html.text or "Clinical Developmental Monitoring Report" in print_html.text
    assert "Maya" in print_html.text
    print(f"  ✓ Rendered /journey/{maya_id}/print printable browser preview")

    print("\n" + "=" * 80)
    print("🎉 ALL PHASE 2 AUTOMATED TESTS COMPLETED & PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_phase2_suite()
