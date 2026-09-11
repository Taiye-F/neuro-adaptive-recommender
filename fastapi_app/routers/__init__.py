# routers/__init__.py
from fastapi import APIRouter
from routers.recommend_router import recommend_router
from routers.auth_router import auth_router
from routers.children_router import children_router
from routers.journey_router import journey_router
from routers.schedules_router import schedules_router
from routers.meltdowns_router import meltdowns_router
from routers.reminders_router import reminders_router
from routers.clinician_router import clinician_router
from routers.evidence_router import evidence_router

api_router = APIRouter()

# Include auth endpoints
api_router.include_router(auth_router)

# Include recommend endpoints
api_router.include_router(recommend_router)

# Include child profile management
api_router.include_router(children_router)

# Include longitudinal journey & trajectory
api_router.include_router(journey_router)

# Include visual routine schedules
api_router.include_router(schedules_router)

# Include meltdown emergency suite & analytics
api_router.include_router(meltdowns_router)

# Include automated reminders & notifications
api_router.include_router(reminders_router)

# Include clinician portal & caseload management
api_router.include_router(clinician_router)

# Include clinical RAG evidence engine
api_router.include_router(evidence_router)
