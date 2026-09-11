# test_phase4.py
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
    ClinicianPatientAssignment, ClinicalNote
)
from core import state, _load_model, _load_app_cache, _load_book_cache
from services.clinician_service import ClinicianService
from services.rag_evidence_service import EvidenceRAGService
from main import app

# Ensure database tables exist
Base.metadata.create_all(bind=engine)
_load_model()
_load_app_cache()
_load_book_cache()

client = TestClient(app, raise_server_exceptions=True)


def test_phase4_suite():
    print("=" * 80)
    print("STARTING PHASE 4 AUTOMATED INTEGRATION TEST SUITE")
    print("=" * 80)

    ts = int(time.time())

    # ?????????????????????????????????????????????????????????????
    # TEST 1: Clinician & Parent Registration and Authentication
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 1] Testing Clinician & Parent Registration & JWT Role Claims...")
    clinician_username = f"dr_chen_{ts}"
    clinician_email = f"dr_chen_{ts}@hospital.org"
    password = "ClinicianSecurePass123!"

    # 1a. Register Clinician
    reg_clinician = client.post("/auth/register", json={
        "username": clinician_username,
        "email": clinician_email,
        "password": password,
        "role": "clinician"
    })
    assert reg_clinician.status_code == 201, f"Clinician registration failed: {reg_clinician.text}"
    assert reg_clinician.json()["role"] == "clinician"

    # 1b. Login Clinician
    login_clinician = client.post("/auth/login", json={
        "username_or_email": clinician_username,
        "password": password
    })
    assert login_clinician.status_code == 200
    clinician_token = login_clinician.json()["access_token"]
    assert login_clinician.json().get("role") == "clinician"
    clinician_headers = {"Authorization": f"Bearer {clinician_token}"}
    clinician_cookies = {"access_token": clinician_token}
    print(f"? Registered and authenticated clinician '{clinician_username}' with role='clinician'")

    # 1c. Register Parent
    parent_username = f"parent_sarah_{ts}"
    parent_email = f"sarah_p4_{ts}@example.com"
    reg_parent = client.post("/auth/register", json={
        "username": parent_username,
        "email": parent_email,
        "password": password,
        "role": "parent"
    })
    assert reg_parent.status_code == 201

    login_parent = client.post("/auth/login", json={
        "username_or_email": parent_username,
        "password": password
    })
    assert login_parent.status_code == 200
    parent_token = login_parent.json()["access_token"]
    assert login_parent.json().get("role") == "parent"
    parent_headers = {"Authorization": f"Bearer {parent_token}"}
    parent_cookies = {"access_token": parent_token}
    print(f"? Registered and authenticated parent '{parent_username}' with role='parent'")

    # Create a child for parent
    child_resp = client.post("/api/v1/children", headers=parent_headers, json={
        "first_name": "Leo",
        "date_of_birth": "2023-11-15",
        "biological_sex": 1,
        "notes": "Speech delay and tactile hypersensitivity"
    })
    assert child_resp.status_code == 201
    leo_id = child_resp.json()["id"]
    print(f"? Created child patient Leo (ID: {leo_id})")

    # ?????????????????????????????????????????????????????????????
    # TEST 2: RBAC Security Enforcement (Clinician vs Parent)
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 2] Testing RBAC Security Isolation...")
    # Parent tries to access clinician caseload -> MUST be 403 Forbidden
    forbidden_resp = client.get("/api/v1/clinician/patients", headers=parent_headers)
    assert forbidden_resp.status_code == 403, f"Expected 403 Forbidden for parent, got {forbidden_resp.status_code}"
    print("? Confirmed: Parent user is rejected with HTTP 403 Forbidden from /api/v1/clinician/patients")

    # Parent tries to assign patient -> MUST be 403 Forbidden
    forbidden_assign = client.post("/api/v1/clinician/patients/assign", headers=parent_headers, json={
        "child_id": leo_id,
        "access_level": "full_clinical"
    })
    assert forbidden_assign.status_code == 403
    print("? Confirmed: Parent user is rejected with HTTP 403 Forbidden from /api/v1/clinician/patients/assign")

    # Unauthenticated request (no session cookies, no bearer header) -> MUST be 401 Unauthorized
    unauth_client = TestClient(app)
    unauth_resp = unauth_client.get("/api/v1/clinician/patients")
    assert unauth_resp.status_code == 401
    print("? Confirmed: Unauthenticated request rejected with HTTP 401 Unauthorized")

    # ?????????????????????????????????????????????????????????????
    # TEST 3: Clinician Patient Assignment & Caseload Retrieval
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 3] Testing Patient Caseload Assignment & Roster Retrieval...")
    assign_resp = client.post("/api/v1/clinician/patients/assign", headers=clinician_headers, json={
        "child_id": leo_id,
        "access_level": "full_clinical"
    })
    assert assign_resp.status_code == 201, f"Assignment failed: {assign_resp.text}"
    assert assign_resp.json()["status"] == "success"
    print(f"? Assigned patient Leo to Dr. Chen's caseload: {assign_resp.json()['message']}")

    # Clinician retrieves roster
    roster_resp = client.get("/api/v1/clinician/patients", headers=clinician_headers)
    assert roster_resp.status_code == 200
    roster = roster_resp.json()
    assert len(roster) >= 1
    leo_roster = next((p for p in roster if p["child_id"] == leo_id), None)
    assert leo_roster is not None
    assert leo_roster["child_name"] == "Leo"
    assert leo_roster["parent_name"] == parent_username
    print(f"? Roster retrieved: Found {len(roster)} patients, including Leo with initial triage='{leo_roster['triage_priority']}'")

    # ?????????????????????????????????????????????????????????????
    # TEST 4: Triage Priority Rules & Escalation
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 4] Testing Clinical Triage Priority Classification Engine...")
    db = next(get_db())
    now = datetime.now(timezone.utc)

    # 4a. Add a High-Risk Assessment (78%) for Leo -> Should escalate to "urgent"
    high_risk_assessment = ScreeningAssessment(
        child_id=UUID(leo_id),
        age_months=24,
        a1_score=3, a2_score=3, a3_score=3, a4_score=3, a5_score=3,
        a6_score=3, a7_score=3, a8_score=3, a9_score=3, a10_score=3,
        risk_probability=78.5,
        is_high_risk=True,
        total_flags=8,
        profile_text="High ASD risk profile with multiple social-communication flags.",
        profile_explained="Immediate clinical follow-up indicated.",
        completed_at=now
    )
    db.add(high_risk_assessment)
    db.commit()

    roster_resp = client.get("/api/v1/clinician/patients", headers=clinician_headers)
    roster = roster_resp.json()
    leo_roster = next(p for p in roster if p["child_id"] == leo_id)
    assert leo_roster["triage_priority"] == "urgent"
    assert any("High ASD Probability" in r for r in leo_roster["triage_reasons"])
    print(f"? Verified Triage Escalation: 78.5% risk triggered priority='urgent' ({leo_roster['triage_reasons']})")

    # 4b. Add sensory meltdowns to verify volatility escalation
    for i in range(3):
        db.add(MeltdownIncident(
            child_id=UUID(leo_id),
            start_time=now - timedelta(days=i, hours=2),
            end_time=now - timedelta(days=i, hours=1, minutes=50),
            duration_seconds=600,
            location="Daycare",
            intensity=4,
            triggers=["Noise", "Transition"]
        ))
    db.commit()

    roster_resp = client.get("/api/v1/clinician/patients", headers=clinician_headers)
    leo_roster = next(p for p in roster_resp.json() if p["child_id"] == leo_id)
    assert leo_roster["total_meltdowns"] >= 3
    assert any("Sensory Volatility" in r for r in leo_roster["triage_reasons"])
    print(f"? Verified Sensory Volatility Triage: 3 meltdowns logged, reason '{leo_roster['triage_reasons']}'")

    # ?????????????????????????????????????????????????????????????
    # TEST 5: Clinical Notes Lifecycle
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 5] Testing Clinical Consultation Notes Logging & Endorsement...")
    note_payload = {
        "note_type": "consultation",
        "content": "Comprehensive 24-month developmental evaluation. Recommend AAC picture exchange system and sensory deep pressure protocol during classroom transitions."
    }
    note_resp = client.post(
        f"/api/v1/clinician/patients/{leo_id}/notes",
        headers=clinician_headers,
        json=note_payload
    )
    assert note_resp.status_code == 201
    created_note = note_resp.json()
    assert created_note["note_type"] == "consultation"
    assert created_note["clinician_username"] == clinician_username
    assert "Recommend AAC" in created_note["content"]
    print(f"? Successfully created ClinicalNote (ID: {created_note['id']}) by Dr. {clinician_username}")

    # Retrieve patient notes
    notes_list_resp = client.get(f"/api/v1/clinician/patients/{leo_id}/notes", headers=clinician_headers)
    assert notes_list_resp.status_code == 200
    notes = notes_list_resp.json()
    assert len(notes) >= 1
    assert notes[0]["id"] == created_note["id"]
    print(f"? Retrieved {len(notes)} clinical notes for patient Leo")

    # ?????????????????????????????????????????????????????????????
    # TEST 6: Patient 360-Degree Longitudinal Overview API
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 6] Testing Patient 360-Degree Longitudinal Profile API...")
    p360_resp = client.get(f"/api/v1/clinician/patients/{leo_id}", headers=clinician_headers)
    assert p360_resp.status_code == 200
    p360_data = p360_resp.json()
    assert p360_data["child"]["first_name"] == "Leo"
    assert "trajectory" in p360_data
    assert "meltdown_analytics" in p360_data
    assert "schedules" in p360_data
    assert "clinical_notes" in p360_data
    print(f"? 360? Profile validated: Trajectory current_risk={p360_data['trajectory']['current_risk']}%, Meltdowns count={p360_data['meltdown_analytics']['total_incidents']}")

    # ?????????????????????????????????????????????????????????????
    # TEST 7: Semantic Hybrid RAG Evidence Engine
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 7] Testing Semantic Hybrid RAG Clinical Evidence Engine...")
    
    # 7a. Search AAC speech
    aac_search = client.get("/api/v1/evidence/search?q=AAC+speech+language&top_k=3")
    assert aac_search.status_code == 200
    aac_data = aac_search.json()
    assert aac_data["total_found"] > 0
    top_intervention = aac_data["results"][0]["intervention_name"]
    assert any(k in top_intervention for k in ["Proloquo2Go", "LAMP", "MITA"])
    print(f"? RAG Search 'AAC speech language': Top result '{top_intervention}' (Relevance: {aac_data['results'][0]['relevance_score']}%)")

    # 7b. Search Deep Pressure sensory
    sensory_search = client.get("/api/v1/evidence/search?q=deep+pressure+sensory+calming&top_k=3")
    assert sensory_search.status_code == 200
    sensory_data = sensory_search.json()
    top_sensory = sensory_data["results"][0]["intervention_name"]
    assert "Deep Pressure" in top_sensory or "Noise Reduction" in top_sensory or "MITA" in top_sensory
    print(f"? RAG Search 'deep pressure sensory': Top result '{top_sensory}'")

    # 7c. Search AAP Guidelines
    surveillance_search = client.get("/api/v1/evidence/search?q=AAP+surveillance+screening&top_k=2")
    assert surveillance_search.status_code == 200
    top_guideline = surveillance_search.json()["results"][0]
    assert "Surveillance" in top_guideline["intervention_name"] or "AAP" in top_guideline["title"]
    print(f"? RAG Search 'AAP surveillance': Found '{top_guideline['intervention_name']}'")

    # 7d. Specific intervention detail lookup
    mita_resp = client.get("/api/v1/evidence/interventions/MITA")
    assert mita_resp.status_code == 200
    mita_data = mita_resp.json()
    assert "MITA" in mita_data["intervention_name"]
    assert mita_data["sample_size"] == 6454
    print(f"? Specific Intervention Detail: MITA (N={mita_data['sample_size']}, Tier={mita_data['evidence_tier']})")

    # 7e. Non-existent intervention 404
    missing_resp = client.get("/api/v1/evidence/interventions/NonExistentTherapyXYZ")
    assert missing_resp.status_code == 404

    # 7f. Clinical context augmentation string generator
    augmented_str = EvidenceRAGService.augment_clinical_context("AAC communication", top_k=2)
    assert "CLINICALLY VALIDATED EVIDENCE" in augmented_str
    print(f"? RAG Prompt Augmentation generator validated (length: {len(augmented_str)} chars)")

    # ?????????????????????????????????????????????????????????????
    # TEST 8: UI Endpoints & Role Redirection
    # ?????????????????????????????????????????????????????????????
    print("\n[TEST 8] Testing UI Endpoints & Redirection...")
    # Clinician dashboard with clinician cookie -> HTTP 200
    dash_resp = client.get("/clinician/dashboard", cookies=clinician_cookies)
    assert dash_resp.status_code == 200
    assert "Triage Roster" in dash_resp.text
    print("? GET /clinician/dashboard renders HTTP 200 with roster view")

    # Clinician dashboard with parent cookie -> Redirects (303) to /
    dash_parent_resp = client.get("/clinician/dashboard", cookies=parent_cookies, follow_redirects=False)
    assert dash_parent_resp.status_code == 303
    assert dash_parent_resp.headers["location"] == "/"
    print("? GET /clinician/dashboard correctly redirects non-clinician parent to '/'")

    # Clinician patient detail view -> HTTP 200
    detail_resp = client.get(f"/clinician/patients/{leo_id}", cookies=clinician_cookies)
    assert detail_resp.status_code == 200
    assert "Leo" in detail_resp.text
    assert "Consultation & Care Notes" in detail_resp.text
    print(f"? GET /clinician/patients/{leo_id} renders HTTP 200 with 360? clinical view")

    # Evidence Library Explorer -> HTTP 200
    evidence_ui_resp = client.get("/evidence", cookies=clinician_cookies)
    assert evidence_ui_resp.status_code == 200
    assert "Evidence Library" in evidence_ui_resp.text
    print("? GET /evidence renders HTTP 200 with full trial explorer")

    print("\n" + "=" * 80)
    print("ALL PHASE 4 INTEGRATION & COMPONENT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_phase4_suite()
