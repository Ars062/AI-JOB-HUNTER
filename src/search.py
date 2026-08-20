"""Hybrid search: vector similarity + keyword search over SQLite."""

import re

import numpy as np

from config import TOP_K_DEFAULT
from src.db import get_all_embeddings, keyword_search
from src.embeddings import embed_text

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "for", "of", "to",
    "with", "show", "find", "me", "jobs", "job", "are", "there", "i", "want",
    "looking", "need", "please", "do", "does", "company", "companies", "that",
    "which", "have", "has", "any", "from", "list", "get", "is", "was", "can",
    "you", "your", "my", "resume", "match", "best", "top", "like",
}


def _extract_terms(query: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#._-]{1,}", query.lower())
    terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:8]


def vector_search(query: str, filters: dict | None = None, k: int = TOP_K_DEFAULT) -> list[tuple[dict, float]]:
    matrix, meta = get_all_embeddings()
    if matrix.shape[0] == 0:
        return []
    q = embed_text(query)
    scores = (matrix @ q.T).flatten()

    pairs = list(zip(meta, scores.tolist()))
    pairs = [p for p in pairs if _matches_filters(p[0], filters)]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[: k * 2]


def _matches_filters(job: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    if filters.get("source") and job.get("source") != filters["source"]:
        return False
    if filters.get("company") and filters["company"].lower() not in (job.get("company") or "").lower():
        return False
    if filters.get("location") and filters["location"].lower() not in (job.get("location") or "").lower():
        return False
    return True


def search(query: str, filters: dict | None = None, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    """Hybrid search. Returns job dicts augmented with a 'score' (0-1)."""
    terms = _extract_terms(query)
    kw = keyword_search(terms, filters=filters, limit=top_k * 4)
    vec = vector_search(query, filters=filters, k=top_k)

    merged: dict[int, dict] = {}
    for job in kw:
        merged[job["id"]] = job

    # keyword hits get a base score, vector hits refine it
    for job, score in vec:
        if job["id"] in merged:
            merged[job["id"]]["score"] = max(merged[job["id"]].get("score", 0.0), score)
        else:
            entry = dict(job)
            entry["score"] = score
            merged[entry["id"]] = entry

    results = sorted(merged.values(), key=lambda j: j.get("score", 0.0), reverse=True)
    return results[:top_k]


def format_jobs_for_llm(jobs: list[dict]) -> str:
    lines = []
    for j in jobs:
        search_score = j.get("score")
        ats = j.get("ats_score")
        score_txt = f" (search {search_score:.0%})" if search_score is not None else ""
        ats_txt = f" (ATS {ats}/100)" if isinstance(ats, int) else ""
        missing = j.get("missing_keywords")
        missing_txt = f"\n  Missing keywords: {', '.join(missing[:8])}" if missing else ""
        lines.append(
            f"- {j.get('title')} @ {j.get('company')} | {j.get('location')} | {j.get('source')}"
            f"{score_txt}{ats_txt}{missing_txt}\n  URL: {j.get('url')}\n  "
            f"Snippet: {(j.get('description') or '')[:300].strip()}"
        )
    return "\n".join(lines)
