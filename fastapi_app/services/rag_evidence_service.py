# rag_evidence_service.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).resolve().parent.parent / "files" / "clinical_evidence_corpus.json"


class EvidenceRAGService:
    _corpus: List[Dict[str, Any]] = []
    _vectorizer: Optional[TfidfVectorizer] = None
    _tfidf_matrix = None
    _initialized: bool = False

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return

        if not CORPUS_PATH.exists():
            log.warning(f"Clinical evidence corpus not found at {CORPUS_PATH}")
            cls._corpus = []
            cls._initialized = True
            return

        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                cls._corpus = json.load(f)

            # Build searchable text documents
            documents = []
            for item in cls._corpus:
                doc = (
                    f"{item.get('title', '')} "
                    f"{item.get('intervention_name', '')} "
                    f"{item.get('category', '')} "
                    f"{item.get('evidence_tier', '')} "
                    f"{item.get('primary_outcomes', '')} "
                    f"{item.get('clinical_summary', '')} "
                    f"{' '.join(item.get('keywords', []))} "
                    f"{item.get('study_citation', '')}"
                )
                documents.append(doc)

            cls._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            cls._tfidf_matrix = cls._vectorizer.fit_transform(documents)
            cls._initialized = True
            log.info(f"✓ Evidence RAG Engine initialized with {len(cls._corpus)} clinical evidence documents.")
        except Exception as e:
            log.error(f"Failed to initialize Evidence RAG Engine: {e}")
            cls._corpus = []
            cls._initialized = True

    @classmethod
    def search_evidence(cls, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        cls._initialize()
        if not cls._corpus or cls._vectorizer is None or cls._tfidf_matrix is None:
            return []

        query = query.strip()
        if not query:
            return cls._corpus[:top_k]

        query_vec = cls._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, cls._tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            # If similarity is zero, check for keyword presence as fallback
            item = dict(cls._corpus[idx])
            item["relevance_score"] = round(score * 100.0, 1)

            if score > 0.05 or any(w.lower() in item["title"].lower() or w.lower() in item.get("keywords", []) for w in query.split()):
                results.append(item)
                if len(results) >= top_k:
                    break

        # If no results matched threshold, return top items
        if not results:
            for idx in top_indices[:top_k]:
                item = dict(cls._corpus[idx])
                item["relevance_score"] = round(float(similarities[idx]) * 100.0, 1)
                results.append(item)

        return results

    @classmethod
    def get_intervention_evidence(cls, name: str) -> Optional[Dict[str, Any]]:
        cls._initialize()
        name_lower = name.strip().lower()
        for item in cls._corpus:
            if name_lower in item.get("intervention_name", "").lower() or any(name_lower in k.lower() for k in item.get("keywords", [])):
                return item
        return None

    @classmethod
    def augment_clinical_context(cls, query: str, top_k: int = 2) -> str:
        """Formats retrieved clinical evidence into an authoritative grounding block for generative prompts."""
        evidence_items = cls.search_evidence(query, top_k=top_k)
        if not evidence_items:
            return ""

        context_lines = [
            "### CLINICALLY VALIDATED EVIDENCE & RESEARCH CITATIONS:",
        ]
        for i, item in enumerate(evidence_items, 1):
            context_lines.append(
                f"{i}. **{item['intervention_name']}** ({item['evidence_tier']}): "
                f"{item['study_citation']}. "
                f"Sample size: N={item.get('sample_size', 'N/A')}. "
                f"Outcomes: {item['primary_outcomes']}"
            )
        return "\n".join(context_lines)
