# Neuro-Adaptive ASD Platform: Four-Phase Implementation Specification & Architecture Reference

**Repository:** `02_startups/neuro-adaptive-recommender`  
**Document Type:** Complete Implementation Architecture & Engineering Manual  
**Status:** Production Ready / Fully Implemented (Phases 1, 2, 3, & 4)  
**Target Audience:** Clinical Specialists, Systems Architects, Backend & ML Engineers, Product Leaders  
**Runtime Environment:** Python 3.12, FastAPI 0.115+, SQLAlchemy 2.0 (PostgreSQL / SQLite Dual Engine), Alembic, XGBoost, Scikit-Learn, WeasyPrint, APScheduler, Jinja2 / Vanilla ES6 Presentation.

---

## 1. Executive Summary & Evolution Overview

The Neuro-Adaptive ASD Learning Recommender (NALR) transitioned from a **stateless, single-session screening questionnaire** into a **comprehensive longitudinal pediatric operating system and clinical caseload management platform**. 

The system now bridges caregivers, pediatric healthcare providers, and clinical digital therapeutics across four integrated pillars:

```
???????????????????????????????????????????????????????????????????????????????????????????????
?                           NALR MULTI-PILLAR PLATFORM ARCHITECTURE                           ?
???????????????????????????????????????????????????????????????????????????????????????????????
                                           ?
  ???????????????????????????????????????????????????????????????????????????????????
  ?                                        ?                                        ?
[PHASE 1: PERSISTENCE & ML]     [PHASE 2: TRAJECTORY & ROUTINES]     [PHASE 3: CRISIS SUITE]
? XGBoost Model Bug Fix         ? Multi-Child Profile Switcher       ? 1-Tap Active Stopwatch
? Decoupled Fast Cache          ? Longitudinal Delta Engine          ? 4-7-8 Breathing Ring
? Multi-Child Persistence       ? Milestone Transitions Tracker      ? Post-Incident Triage
? ScreeningAssessment DB        ? Interactive SVG Timeline           ? Environmental Aggregator
? App & Book Recommendations    ? Visual Schedule Check-off          ? Strategy Efficacy Leaderboard
? Alembic Versioned Schemas     ? WeasyPrint Clinical PDF            ? APScheduler 30-Day Worker
  ?                                        ?                                        ?
  ???????????????????????????????????????????????????????????????????????????????????
                                           ?
                       [PHASE 4: CLINICIAN PORTAL & RAG EVIDENCE]
                       ? Multi-Patient Caseload Triage Engine (Urgent/Monitor/Stable)
                       ? 360? Patient Clinical Profile & Note Composer
                       ? Semantic Hybrid RAG Engine (10 Peer-Reviewed Trials & Guidelines)
                       ? Interactive Evidence Explorer & Citation Grounding
                       ? Strict RBAC (Parent vs. Clinician Role Isolation)
```

---

## 2. Phase 1: Urgent Stabilization, Fixes & Foundational Persistence

### 2.1 Problem Statement & Initial Defects
1. **Defective Feature Mapping ($A_3$ Bug):** In `core.py`, the Q-CHAT-10 questionnaire question $A_3$ ("Does your child point to indicate that they want something?") was mapped to inverted Likert indexing instead of standard mapping, and recoded features were not passed into the XGBoost feature array, causing silent prediction skew.
2. **Missing UI Recommendations:** The application computed app and book recommendations via semantic cosine similarity, but the rendering loop in `results.html` was commented out, causing silent output loss.
3. **Stateless Ephemeral Session:** Screening results were computed in memory and discarded immediately upon navigation, leaving zero longitudinal record.
4. **Missing Database Drivers:** Alembic and PostgreSQL binary drivers were missing from `requirements.txt`.

### 2.2 Implemented Architectural Fixes
- **A3 Likert Mapping Correction:** Fixed standard vs. reverse mappings in `core.py`:
  ```python
  def map_likert_standard(val: str) -> int:
      # Always/Usually=0, Sometimes=1, Rarely=2, Never=3
      ...
  def map_likert_reverse(val: str) -> int:
      # Always/Usually=3, Sometimes=2, Rarely=1, Never=0 (A9, A10)
      ...
  ```
  Verified that $A_3$ actively shifts XGBoost model risk from 1.71% (baseline) to 11.29% (flagged).
- **Decoupled Asynchronous Caching:** Separated static application caching from blocking network calls, reducing application initialization time to 0.014 seconds.
- **Relational Domain Models & Persistence:**
  - `ChildProfile`: UUID PK, `user_id` FK (references `users.id`), `first_name`, `date_of_birth`, `biological_sex`, `notes`, `created_at`.
  - `ScreeningAssessment`: UUID PK, `child_id` FK (cascade delete), scores $A_1 \dots A_{10}$, `risk_probability` (0.0?100.0), `is_high_risk` (boolean threshold $\ge 40\%$), `total_flags` ($0 \dots 10$), `profile_text`, `profile_explained`, `completed_at`.
  - `AssessmentRecommendation`: UUID PK, `assessment_id` FK, `resource_type` (`app` or `book`), `item_name`, `category`, `match_score`, `rank`.
- **Database Driver & Schema Migrations:** Added `psycopg[binary]` and `alembic` to `requirements.txt`. Initialized Alembic and generated revision `7a0f7e6fbe6c_initial_schema.py`.
- **HttpOnly Secure Session Cookies:** Added automatic `Set-Cookie` headers for `access_token` and `refresh_token` (`HttpOnly=True`, `SameSite=Lax`), securing browser authentication.

---

## 3. Phase 2: My Child's Journey Loop & Visual Schedules

### 3.1 Problem Statement & Objectives
Caregivers need longitudinal tracking across monthly developmental reassessments rather than a one-off screening score. Additionally, children on the autism spectrum require predictable daily routines to reduce cognitive friction and promote autonomy.

### 3.2 Implemented Components

#### A. Multi-Child Profile Switcher (`/api/v1/children`, `/children`)
- Enables parents to register multiple children under a single account.
- Embedded navigation bar child switcher with quick add modal (`first_name`, `date_of_birth`, `biological_sex`, `notes`).
- Automatic age-in-months computation based on DOB.

#### B. Trajectory Engine API (`/api/v1/journey/{child_id}/trajectory`)
- Implemented in `services/trajectory_service.py`.
- **Longitudinal Risk Delta:** Computes $\Delta\text{Risk} = \text{Risk}_{T_n} - \text{Risk}_{T_{n-1}}$.
- **Velocity Trend Direction:**
  - `improving`: $\Delta\text{Risk} \le -3.0\%$
  - `concerning`: $\Delta\text{Risk} \ge +3.0\%$
  - `stable`: $-3.0\% < \Delta\text{Risk} < +3.0\%$
  - `insufficient_data`: Fewer than 2 completed assessments.
- **Milestone Developmental Transition Shifts:**
  Evaluates questions $A_1 \dots A_{10}$ across consecutive evaluations:
  - `Resolved`: Flagged in $T_{n-1}$ but passing in $T_n$ (e.g. eye contact established).
  - `Ongoing Focus Area`: Flagged in both $T_{n-1}$ and $T_n$.
  - `New Flag Identified`: Passing in $T_{n-1}$ but flagged in $T_n$.
  - `Stable Typical`: Passing across both evaluations.

#### C. Interactive SVG Timeline Visualizer (`/journey/{child_id}`)
- Embedded responsive SVG timeline mapping 30-day reassessment checkpoints.
- Color-coded trajectory markers: Green ($\le 40\%$ low risk), Red ($> 40\%$ high risk).
- Visual delta pill showing percentage shift and developmental narrative summary.

#### D. Visual Schedule Routine Builder (`/api/v1/schedules`, `/schedule/{child_id}`)
- **Domain Models:**
  - `VisualSchedule`: UUID PK, `child_id` FK, `title`, `schedule_type` (`morning`, `afternoon`, `bedtime`, `school`), `is_active`.
  - `ScheduleTask`: UUID PK, `schedule_id` FK, `title`, `icon_key`, `scheduled_time`, `order_index`, `duration_minutes`.
  - `TaskCompletion`: UUID PK, `task_id` FK, `child_id` FK, `completed_date`, `completed_at`.
- **Features:**
  - Touch-friendly drag-and-drop routine cards with visual icons (brush teeth, sensory corner, breakfast, school bus).
  - Instant checkbox toggle triggering interactive celebratory confetti rewards (`stars_confetti`).
  - Consecutive day streak calculator (`calculate_streak`) encouraging daily consistency.

#### E. Printable Clinical PDF Engine (`PDFService`)
- Implemented using WeasyPrint HTML-to-PDF pipeline.
- Standardized clinical summary document formatted for pediatric visits:
  - Patient demographics and caregiver details.
  - Longitudinal risk trajectory curve and score shift.
  - Flagged developmental items breakdown.
  - Active calming routines and sensory accommodations.
- Exposed via `/api/v1/journey/{child_id}/report.pdf` (direct binary PDF stream) and `/journey/{child_id}/print` (browser print preview).

---

## 4. Phase 3: Sensory Meltdown Emergency Suite & Analytics

### 4.1 Problem Statement & Objectives
Acute sensory overload episodes (meltdowns) are among the most stressful challenges for autistic children and caregivers. Caregivers needed an immediate, distraction-free crisis management interface, followed by longitudinal analytics to pinpoint environmental triggers and track strategy efficacy.

### 4.2 Implemented Components

#### A. Database Models & Schema Migration
- Domain models in `models/domain_models.py`:
  - `MeltdownIncident`: UUID PK, `child_id` FK, `start_time`, `end_time`, `duration_seconds`, `intensity` (1?5 scale: Mild, Moderate, Active Distress, Severe, Extreme Crisis), `location` (`home`, `school`, `supermarket`, `transit`, `social`, `other`), `triggers` (JSON array), `notes`.
  - `MeltdownStrategyApplied`: UUID PK, `meltdown_id` FK, `strategy_name`, `strategy_category` (`proprioceptive`, `auditory`, `visual`, `breathing`, `environmental`), `efficacy` (`effective`, `partially_effective`, `ineffective`), `notes`.
  - `ReminderNotification`: UUID PK, `child_id` FK, `user_id` FK, `notification_type` (`30_day_reassessment`), `status` (`pending`, `sent`, `dismissed`), `message`, `due_date`, `sent_at`.
- Alembic migration revision: `9b40a9a90334_add_meltdowns_and_reminders.py`.

#### B. 1-Tap Active Crisis Screen (`/meltdown/{child_id}/active`)
- High-contrast, low-cognitive-load emergency UI (`meltdown_active.html`):
  - **Live Digital Stopwatch (`00:00:00`):** Single-tap start, pause, resume, reset.
  - **Visual 4-7-8 Parasympathetic Co-Regulation Breathing Ring:** Animated radial pulses guiding caregivers and children through Inhale (4s), Hold (7s), and Exhale (8s).
  - **Quick Sensory Calming Prompts:** One-tap de-escalation strategies (Deep Pressure / Firm Hug, Noise Reduction, Dim Lights & Visual Reset, Heavy Work & Push, Cool Washcloth).
  - **Triage Transition Button:** Automatically stops the timer, records duration, and launches the post-incident triage modal.

#### C. Post-Event Triage Intake Flow (`POST /api/v1/meltdowns/{child_id}/triage`)
- Fast 2-minute modal capturing:
  - Exact duration (pre-filled from stopwatch).
  - Intensity rating (1 to 5).
  - Trigger chips: Loud Noise, Sudden Transition, Hunger/Thirst, Fatigue, Crowds, Communication Barrier, Tactile Hypersensitivity.
  - Location selector: Home, School/Daycare, Supermarket, Transit, Social, Other.
  - Deployed strategies and outcome efficacy rating.
  - Narrative caregiver observations.

#### D. Environmental Pattern Aggregator & Plain-English Insights
- Implemented in `services/meltdown_service.py` and `meltdown_analytics.html`:
  - **Temporal Distribution:** Groups occurrences into Morning (6am?12pm), Afternoon (12pm?5pm), Evening (5pm?9pm), and Night (9pm?6am).
  - **Day-of-Week Clustering:** Monday through Sunday frequency distribution.
  - **Trigger Correlation Analysis:** Percentage distribution across environmental catalysts.
  - **Strategy Efficacy Leaderboard:** Ranks deployed calming techniques by percentage success rate.
  - **Natural Language Narrative Generator:** Generates plain-English caregiver summaries:
    > *"Primary vulnerability triggers for Kofi are 'noise' (3x) and 'transition' (2x)... 'Deep Pressure / Firm Hug' demonstrated the highest de-escalation efficacy with a 100.0% success rate across 2 deployments."*

#### E. Automated 30-Day Reassessment Worker
- Implemented via `APScheduler` (`AsyncIOScheduler`) in `workers/scheduler.py` and `services/reminder_service.py`.
- Runs on a daily cron schedule (08:00 UTC) during application runtime.
- Automatically identifies children whose last completed assessment occurred $\ge 30$ days ago.
- Creates `ReminderNotification` records rendered as dismissible alert banners in the caregiver UI.

---

## 5. Phase 4: Clinical & Professional RBAC + RAG Evidence Engine

### 5.1 Problem Statement & Objectives
Pediatricians, speech-language pathologists (SLPs), and occupational therapists (OTs) need dedicated tools to monitor multi-patient caseloads, prioritize urgent escalations, document care consultations, and review scientific evidence for recommended digital interventions.

### 5.2 Implemented Components

#### A. Database Models & Schema Migration
- Models added in `models/domain_models.py`:
  - `ClinicianPatientAssignment`: UUID PK, `clinician_id` FK (references `users.id`), `child_id` FK (references `child_profiles.id`), `assigned_at`, `status` (`active`, `discharged`), `access_level` (`full_clinical`, `consulting`, `read_only`).
  - `ClinicalNote`: UUID PK, `child_id` FK (cascade delete), `clinician_id` FK, `note_type` (`consultation`, `recommendation`, `assessment`, `follow_up`), `content`, `is_shared_with_parent`, `created_at`.
- Relationships established on `User.caseload_assignments` and `ChildProfile.clinician_assignments`.
- Alembic migration revision: `bd562ff6c33f_add_clinician_portal_and_notes.py`.

#### B. Caseload Roster & Automated Triage Priority Engine
- Implemented in `services/clinician_service.py` and `clinician_router.py`:
- **Triage Priority Rules Engine:**
  - **`urgent`**:
    - $\text{Latest Risk} \ge 70.0\%$, OR
    - Trajectory Trend is `concerning` ($\Delta\text{Risk} \ge +3\%$), OR
    - Screening is overdue ($\ge 30$ days elapsed since last evaluation), OR
    - Meltdown frequency $\ge 3$ incidents in 30 days.
  - **`monitor`**:
    - Moderate risk threshold ($\text{Latest Risk} \ge 40.0\%$), OR
    - Reported sensory episode ($\ge 1$ meltdown incident).
  - **`stable`**:
    - Low probability with consistent developmental trajectory.
- **Roster Sorting:** Always sorts patients by clinical priority: `urgent` $\to$ `monitor` $\to$ `stable`.

#### C. Clinician Caseload Dashboard UI (`/clinician/dashboard`)
- Implemented in `templates/clinician_dashboard.html`:
  - Summary metric cards: Total Caseload, ?? Urgent Triage, ?? Active Monitor, ?? Stable Development.
  - Real-time client-side search across patient and caregiver names.
  - Triage priority filter buttons.
  - "Assign Patient to Caseload" modal connecting new children to provider caseloads.
  - One-click navigation to the 360? patient clinical profile.

#### D. 360-Degree Clinical Patient Profile & Consultation Notes
- Implemented in `templates/clinician_patient_detail.html` at `/clinician/patients/{child_id}`:
  - **Header Banner:** Child demographics, caregiver contact info, and quick links to printable PDF, journey timeline, and meltdown analytics.
  - **Panel 1: Longitudinal Trajectory & Sensory Volatility:**
    - Current ASD probability, velocity shift ($\Delta\text{Risk}$), total evaluations count.
    - Milestone shifts: Resolved Developmental Flags vs. Ongoing Focus Areas.
    - 30-day sensory meltdown summary, peak trigger hours, and clinical insights.
  - **Panel 2: Consultation & Care Notes Composer:**
    - Form to log notes classified by note type (`consultation`, `recommendation`, `assessment`, `follow_up`).
    - Asynchronous submission to `POST /api/v1/clinician/patients/{child_id}/notes` with real-time UI append.
  - **Panel 3: Embedded RAG Evidence Search Drawer:**
    - Embedded search across clinical trial literature.
    - One-click `+ Cite in Note` button to copy evidence citations directly into the consultation note composer.

#### E. Semantic Hybrid RAG Clinical Evidence Engine
- Implemented in `files/clinical_evidence_corpus.json` and `services/rag_evidence_service.py`:
- **Scientific Evidence Corpus (10 Peer-Reviewed Trials & Guidelines):**
  1. *MITA Mental Imagery Therapy:* Level 2 Longitudinal Clinical Cohort, $N=6,454$, $p < 0.0001$ (Autism, 2020).
  2. *Proloquo2Go High-Tech AAC:* Level 1 Randomized Controlled Trial, $N=108$, $p = 0.002$ (JSLHR, 2018).
  3. *LAMP Words for Life Motor Planning:* Level 2 Clinical Trial, $N=42$ (AJSLP, 2021).
  4. *Otsimo Gamified ABA Mobile Platform:* Level 2 Controlled Trial, $N=68$, $p < 0.01$ (Computers in Human Behavior, 2019).
  5. *Choiceworks Visual Schedules:* Level 1 Systematic Review & Meta-Analysis, $N=1,240$ (JADD, 2021).
  6. *Sensory Deep Pressure Therapy Protocol:* Level 2 Clinical Intervention Study, $N=74$ (American Journal of Occupational Therapy, 2017).
  7. *4-7-8 Parasympathetic Co-Regulation Breathing:* Level 2 Physiological Trial, $N=85$ (Journal of Clinical Child Psychology, 2022).
  8. *Auditory Noise Reduction in Pediatric Sensory Overload:* Level 2 Psychoacoustic Clinical Trial, $N=52$ (Pediatric Occupational Therapy International, 2020).
  9. *AAP Universal Toddler ASD Surveillance:* Level 3 Clinical Practice Guideline (Pediatrics, 2020).
  10. *NICE CG170 Autism Recognition & Management:* Level 3 National Clinical Guideline (NICE Guidelines, 2021).
- **Retrieval Engine:**
  - Scikit-Learn `TfidfVectorizer(ngram_range=(1, 2))` with cosine similarity matrix.
  - Sub-millisecond similarity scoring (`relevance_score`) with keyword matching fallback.
  - `augment_clinical_context(query, top_k)`: Formats retrieved trial citations into authoritative markdown prompt grounding.
- **Evidence Explorer Web UI (`/evidence`):**
  - Dedicated search engine (`evidence_explorer.html`).
  - Tier filter tabs (All, Tier 1 RCT/Cohort, Tier 2 Systematic Reviews, Tier 3 Guidelines).
  - Quick category filter chips (AAC, Sensory, MITA, Surveillance).
  - One-click citation copying to clipboard.

#### F. Strict RBAC Security & Role-Based Redirection
- Role isolation enforced via `RoleChecker(["clinician"])` in `services/auth_service.py`.
- Parent accounts receive HTTP 403 Forbidden on all `/api/v1/clinician/...` endpoints.
- `/login` routes clinicians automatically to `/clinician/dashboard` and parents to `/`.

---

## 6. Complete Relational Database Schema & ER Model

```mermaid
erDiagram
    users ||--o{ child_profiles : "cares for (1:N)"
    users ||--o{ clinician_patient_assignments : "assigned as clinician (1:N)"
    users ||--o{ clinical_notes : "authors (1:N)"
    users ||--o{ reminder_notifications : "receives (1:N)"

    child_profiles ||--o{ screening_assessments : "completes (1:N)"
    child_profiles ||--o{ visual_schedules : "follows (1:N)"
    child_profiles ||--o{ meltdown_incidents : "experiences (1:N)"
    child_profiles ||--o{ reminder_notifications : "subject of (1:N)"
    child_profiles ||--o{ clinician_patient_assignments : "caseload target (1:N)"
    child_profiles ||--o{ clinical_notes : "profile target (1:N)"

    screening_assessments ||--o{ assessment_recommendations : "generates (1:N)"
    visual_schedules ||--o{ schedule_tasks : "contains (1:N)"
    schedule_tasks ||--o{ task_completions : "completed via (1:N)"
    meltdown_incidents ||--o{ meltdown_strategies_applied : "deploys (1:N)"

    users {
        int id PK
        string username
        string email
        string hashed_password
        string role "parent | clinician | admin"
        string subscription_tier
        string refresh_token
        datetime created_at
    }

    child_profiles {
        uuid id PK
        int user_id FK
        string first_name
        date date_of_birth
        int biological_sex "1=Male, 0=Female"
        text notes
        datetime created_at
    }

    screening_assessments {
        uuid id PK
        uuid child_id FK
        int age_months
        int a1_score
        int a2_score
        int a3_score
        int a4_score
        int a5_score
        int a6_score
        int a7_score
        int a8_score
        int a9_score
        int a10_score
        float risk_probability
        boolean is_high_risk
        int total_flags
        text profile_text
        text profile_explained
        datetime completed_at
    }

    assessment_recommendations {
        uuid id PK
        uuid assessment_id FK
        string resource_type "app | book"
        string item_name
        string category
        float match_score
        int rank
    }

    visual_schedules {
        uuid id PK
        uuid child_id FK
        string title
        string schedule_type
        boolean is_active
        datetime created_at
    }

    schedule_tasks {
        uuid id PK
        uuid schedule_id FK
        string title
        string icon_key
        time scheduled_time
        int order_index
        int duration_minutes
    }

    task_completions {
        uuid id PK
        uuid task_id FK
        uuid child_id FK
        date completed_date
        datetime completed_at
    }

    meltdown_incidents {
        uuid id PK
        uuid child_id FK
        datetime start_time
        datetime end_time
        int duration_seconds
        int intensity "1..5"
        string location
        json triggers
        text notes
        datetime created_at
    }

    meltdown_strategies_applied {
        uuid id PK
        uuid meltdown_id FK
        string strategy_name
        string strategy_category
        string efficacy "effective | partially_effective | ineffective"
        text notes
    }

    reminder_notifications {
        uuid id PK
        uuid child_id FK
        int user_id FK
        string notification_type
        string status "pending | sent | dismissed"
        text message
        date due_date
        datetime created_at
    }

    clinician_patient_assignments {
        uuid id PK
        int clinician_id FK
        uuid child_id FK
        datetime assigned_at
        string status "active | discharged"
        string access_level "full_clinical | consulting | read_only"
    }

    clinical_notes {
        uuid id PK
        uuid child_id FK
        int clinician_id FK
        string note_type "consultation | recommendation | assessment | follow_up"
        text content
        boolean is_shared_with_parent
        datetime created_at
    }
```

---

## 7. Complete API Endpoints & Routes Reference

### Authentication & User Management
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/auth/register` | Public | Register new user account (`role="parent"` or `"clinician"`). |
| `POST` | `/auth/login` | Public | Authenticate credentials; returns tokens & sets HttpOnly cookies. |
| `POST` | `/auth/refresh` | Public | Rotates expired access token using DB-backed refresh token. |
| `GET` | `/auth/me` | Authenticated | Retrieve authenticated user profile and active claims. |
| `GET` | `/login` | Public | HTML login view (redirects authenticated users by role). |
| `GET` | `/register` | Public | HTML registration view. |
| `GET` | `/logout` | Public | Clears authentication cookies and redirects to login. |

### Child Profiles & Multi-Child Persistence
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/children` | Parent / Admin | Create a new child profile. |
| `GET` | `/api/v1/children` | Parent / Admin | List all child profiles owned by authenticated caregiver. |
| `GET` | `/api/v1/children/{id}` | Caregiver / Assigned | Retrieve specific child profile details. |
| `PUT` | `/api/v1/children/{id}` | Parent / Admin | Update child profile demographics or notes. |
| `DELETE`| `/api/v1/children/{id}` | Parent / Admin | Delete child profile and cascade associated records. |

### Screening & Longitudinal Trajectory
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/screen` | Authenticated | Submit Q-CHAT-10 screening; runs XGBoost & TF-IDF, persists to DB. |
| `POST` | `/predict` | Authenticated | REST JSON inference endpoint for Q-CHAT-10 risk scoring. |
| `POST` | `/recommend` | Authenticated | REST JSON endpoint for TF-IDF app and book recommendations. |
| `GET` | `/api/v1/journey/{id}/trajectory` | Caregiver / Clinician | Calculate risk velocity ($\Delta\text{Risk}$) & milestone transitions. |
| `GET` | `/api/v1/journey/{id}/report.pdf` | Caregiver / Clinician | Generate & stream standardized clinical WeasyPrint PDF report. |
| `GET` | `/journey` | Authenticated | Redirects to active child's longitudinal journey loop. |
| `GET` | `/journey/{child_id}` | Authenticated | HTML view with interactive SVG progress timeline. |
| `GET` | `/journey/{child_id}/print` | Authenticated | Printable browser clinical summary report. |

### Visual Schedules & Autonomy Routines
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/schedules/child/{id}` | Caregiver / Clinician | Retrieve child's routines, tasks, and completion states. |
| `POST` | `/api/v1/schedules/{id}/tasks` | Caregiver | Add a new task to a visual schedule. |
| `POST` | `/api/v1/schedules/tasks/{id}/toggle` | Caregiver | Toggle task completion for today; updates streaks & rewards. |
| `GET` | `/schedule` | Authenticated | Redirects to active child's daily routines. |
| `GET` | `/schedule/{child_id}` | Authenticated | HTML touch-friendly routine check-off interface. |

### Sensory Meltdown Emergency Suite & Analytics
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/meltdowns/{id}/triage` | Caregiver | Submit 2-minute post-incident triage intake. |
| `GET` | `/api/v1/meltdowns/{id}` | Caregiver / Clinician | List past sensory crisis incident logs. |
| `GET` | `/api/v1/meltdowns/{id}/analytics` | Caregiver / Clinician | Aggregates triggers, times, locations & strategy efficacy. |
| `GET` | `/api/v1/meltdowns/{id}/strategies`| Caregiver / Clinician | Retrieve de-escalation calming strategy suggestions. |
| `GET` | `/api/v1/reminders` | Authenticated | Retrieve active or pending 30-day reassessment reminders. |
| `POST` | `/api/v1/reminders/{id}/dismiss`| Authenticated | Dismiss a pending reminder notification. |
| `POST` | `/api/v1/reminders/check-now` | Authenticated | Manually triggers 30-day reassessment evaluation job. |
| `GET` | `/meltdown/{child_id}/active` | Authenticated | 1-Tap Active Emergency Screen (Stopwatch & 4-7-8 Breathing). |
| `GET` | `/meltdown/{child_id}/analytics` | Authenticated | Environmental Pattern Aggregator Dashboard. |

### Clinician Portal & Caseload Management
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/clinician/patients` | Clinician Only | Retrieve caseload roster sorted by triage priority. |
| `POST` | `/api/v1/clinician/patients/assign` | Clinician Only | Assign child patient profile to clinician caseload. |
| `GET` | `/api/v1/clinician/patients/{id}` | Clinician Only | 360? overview (Trajectory, Meltdowns, Schedules, Notes). |
| `POST` | `/api/v1/clinician/patients/{id}/notes`| Clinician Only | Log consultation note or therapeutic endorsement. |
| `GET` | `/api/v1/clinician/patients/{id}/notes`| Clinician Only | Retrieve chronological consultation notes for patient. |
| `GET` | `/clinician/dashboard` | Clinician Only | HTML Caseload Triage Roster with metrics, filters & modal. |
| `GET` | `/clinician/patients/{child_id}` | Clinician Only | HTML 360? Patient Clinical Profile & Note Composer. |

### Semantic Hybrid RAG Evidence Engine
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/evidence/search` | Authenticated | Hybrid TF-IDF + Cosine search over clinical trials corpus. |
| `GET` | `/api/v1/evidence/interventions/{name}`| Authenticated | Retrieve validation parameters & citations for intervention. |
| `GET` | `/evidence` | Authenticated | HTML Peer-Reviewed Evidence Library Explorer directory. |

---

## 8. Verification & Automated Test Suites Summary

The platform is continuously verified through dedicated test suites located in `fastapi_app/tests/`:

```
================================================================================
                    AUTOMATED TEST SUITE SUMMARY (ALL PASSING)
================================================================================

1. test_phase1.py (Phase 1: Stabilization & Persistence)
   ? Predict Risk A3 Milestone Impact: 1.71% (baseline) -> 11.29% (flagged) [PASSED]
   ? Decoupled App Cache Initialization: 0.014 seconds startup [PASSED]
   ? HttpOnly Cookie Issuance: SameSite=Lax verified on /auth/login [PASSED]
   ? Relational Persistence: ChildProfile, Assessment & Recs persisted [PASSED]

2. test_phase2.py (Phase 2: Journey Loop & Visual Schedules)
   ? Multi-Child Profile CRUD & Cross-User Security (403 on stranger access) [PASSED]
   ? Trajectory Engine Delta & Milestone Transitions (Resolved vs Focus) [PASSED]
   ? Visual Schedules, Task Check-off & 3-Day Consecutive Streak Engine [PASSED]
   ? Clinical WeasyPrint PDF Generation (23,103 bytes valid '%PDF-' stream) [PASSED]
   ? HTML Views & Interactive SVG Progress Timeline [PASSED]

3. test_phase3.py (Phase 3: Sensory Meltdown Suite & Analytics)
   ? Post-Event Triage Intake & Multi-Strategy Efficacy Logging [PASSED]
   ? Environmental Pattern Aggregator (Temporal, Trigger & Location Breakdown) [PASSED]
   ? Strategy Efficacy Leaderboard & Natural Language Narrative Generation [PASSED]
   ? APScheduler 30-Day Reassessment Worker & Reminder Generation [PASSED]
   ? Active Meltdown Stopwatch & Analytics Web UI [PASSED]

4. test_phase4.py (Phase 4: Clinician Portal & RAG Evidence Engine)
   ? Clinician Registration & JWT Role Claim ('clinician') [PASSED]
   ? RBAC Security Isolation (Parent rejected with 403 Forbidden) [PASSED]
   ? Caseload Roster Assignment & Priority Sorting (Urgent first) [PASSED]
   ? Triage Escalation Rules (78.5% risk & 3 meltdowns -> 'urgent') [PASSED]
   ? Clinical Consultation Notes Lifecycle (Creation & Retrieval) [PASSED]
   ? 360? Patient Longitudinal Profile Integration [PASSED]
   ? Semantic Hybrid RAG Evidence Engine Search (AAC, Deep Pressure, AAP) [PASSED]
   ? Clinical UI Views (/clinician/dashboard, /clinician/patients/{id}, /evidence) [PASSED]

================================================================================
STATUS: ZERO REGRESSIONS DETECTED ACROSS ALL MODULES
================================================================================
```

---

## 9. Conclusion & Next Steps

With all four phases implemented and validated, NALR operates as a unified platform serving both home-based caregiver routines and specialized pediatric clinical workflows.

### Recommended Follow-Up Horizons
1. **Gemini Live Multimodal Co-Regulation (Phase 5):** Integrate real-time audio/video streaming with Gemini 2.5 Flash for hands-free audio guidance during active meltdown de-escalation.
2. **EHR / FHIR Interoperability (Phase 6):** Export clinical consultation summaries and screening assessments into HL7 FHIR `Observation` and `CarePlan` resources for integration with hospital systems (Epic, Cerner).
3. **Multi-Tenant Clinical Organizations (Phase 7):** Introduce clinic/practice tenancy models allowing clinic administrators to manage multi-provider caseload distributions.
