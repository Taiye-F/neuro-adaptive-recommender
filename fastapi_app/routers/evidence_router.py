# routers/evidence_router.py
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status

from schemas.domain_schemas import EvidenceSearchResponse
from services.rag_evidence_service import EvidenceRAGService

evidence_router = APIRouter(prefix="/api/v1/evidence", tags=["Clinical RAG Evidence Engine"])


@evidence_router.get("/search", response_model=EvidenceSearchResponse)
def search_clinical_evidence(
    q: str = Query("", description="Search term for trial, intervention, or clinical outcome"),
    top_k: int = Query(5, ge=1, le=20)
):
    """Perform semantic hybrid search over peer-reviewed ASD clinical trials, evidence tiers, and guidelines."""
    results = EvidenceRAGService.search_evidence(query=q, top_k=top_k)
    return EvidenceSearchResponse(
        query=q,
        total_found=len(results),
        results=results
    )


@evidence_router.get("/interventions/{name}")
def get_intervention_evidence_detail(name: str):
    """Retrieve clinical validation parameters and trial citations for a specific intervention."""
    item = EvidenceRAGService.get_intervention_evidence(name)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scientific evidence record found for '{name}'."
        )
    return item
