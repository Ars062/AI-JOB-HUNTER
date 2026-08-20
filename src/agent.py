"""Orchestrates: question -> scrape fresh from Apify -> ATS score -> answer."""

import logging

from src.db import add_chat_message, get_chat_history
from src.llm import extract_search_params
from src.scraper import run_job_search
from src.normalize import normalize_items

logger = logging.getLogger(__name__)


def ask(
    question: str,
    top_k: int = 20,
    resume_text: str | None = None,
    resume_skills: str | None = None,
    ui_filters: dict | None = None,
) -> dict:
    """Answer a user question by scraping fresh jobs from Apify.

    Always scrapes fresh from LinkedIn via Apify.
    Returns {"answer": str, "jobs": [job dicts with ats_score]}.
    """
    add_chat_message("user", question)

    # Extract params from question
    params = extract_search_params(question)

    # Build search query: UI filter > question keywords > resume skills > raw question
    _generic = {"jobs", "job", "find", "show", "get", "list", "search",
                "looking", "for", "me", "please", "recommend", "openings", "roles",
                "matching", "profile", "based", "on", "my", "cv", "resume", "here"}
    if ui_filters and ui_filters.get("query"):
        query = ui_filters["query"]
    else:
        keywords = [k for k in ((params or {}).get("keywords") or []) if k.lower() not in _generic]
        if keywords:
            query = " ".join(keywords)
        elif resume_skills:
            query = " ".join(
                s.strip() for s in resume_skills.split(",") if s.strip()
            )[:80]
        else:
            query = question

    location = ""
    max_age_days = 0
    max_items = 20

    if ui_filters:
        if ui_filters.get("location"):
            location = ui_filters["location"]
        if ui_filters.get("freshness"):
            max_age_days = {"Anytime": 0, "24 hours": 1, "7 days": 7, "30 days": 30}.get(
                ui_filters["freshness"], 0
            )
        if ui_filters.get("max_items"):
            max_items = ui_filters["max_items"]

    # Question-extracted location overrides if no UI location
    if not location and params and params.get("location"):
        location = params["location"]

    # If still no location, try CV
    if not location and resume_text:
        import re
        # Look for "open to relocate" + location
        relocate_match = re.search(
            r'open to relocate\s+(?:in\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+)*)',
            resume_text, re.IGNORECASE
        )
        if relocate_match:
            location = relocate_match.group(1).strip()
        else:
            # Fallback: look for city, country pattern
            match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', resume_text)
            if match:
                location = match.group(0)

    # Scrape fresh from Apify
    jobs = []
    try:
        raw_items = run_job_search(
            query=query,
            location=location,
            max_items=max_items,
            max_age_days=max_age_days,
        )
        if raw_items:
            jobs = normalize_items(raw_items)
            # Deduplicate by URL
            seen_urls = set()
            unique_jobs = []
            for job in jobs:
                url = job.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_jobs.append(job)
                elif not url:
                    unique_jobs.append(job)
            jobs = unique_jobs
    except Exception as e:
        logger.warning("Apify scrape failed: %s", e)

    # ATS scoring
    if resume_text and jobs:
        from src.ats_scorer import score_job
        skills_list = [s.strip() for s in resume_skills.split(",")] if resume_skills else None
        for job in jobs:
            ats_result = score_job(resume_text, job, resume_skills=skills_list)
            job["ats_score"] = ats_result["score"]
            job["matched_keywords"] = ats_result.get("matched_keywords", [])
            job["missing_keywords"] = ats_result.get("missing_keywords", [])
        # Drop low-scoring jobs (below 20), then sort by ATS score (highest first)
        jobs = [j for j in jobs if j.get("ats_score", 0) >= 20]
        jobs.sort(key=lambda j: j.get("ats_score", 0), reverse=True)

    # Build answer deterministically (no LLM hallucination: scores come from the engine)
    answer = _build_answer(jobs, resume_skills)

    add_chat_message("assistant", answer)
    return {"answer": answer, "jobs": jobs}


def _build_answer(jobs: list[dict], resume_skills: str | None = None) -> str:
    """Simple, deterministic summary built from the scored job list."""
    if not jobs:
        return (
            "No matching jobs found for your profile. "
            "Try adjusting your filters (location, country) and search again."
        )
    lines = []
    if resume_skills:
        lines.append(f"Found {len(jobs)} matching jobs based on your CV skills.")
    else:
        lines.append(f"Found {len(jobs)} matching jobs.")

    for j in jobs[:5]:
        ats = j.get("ats_score")
        ats_txt = f" — ATS {ats}/100" if isinstance(ats, int) else ""
        title = j.get("title") or "Untitled"
        company = j.get("company") or ""
        location = j.get("location") or ""
        meta = " | ".join(x for x in [company, location] if x)
        lines.append(f"▪ **{title}**{ats_txt}{' — ' + meta if meta else ''}")

    if len(jobs) > 5:
        lines.append(f"... and {len(jobs) - 5} more below, sorted by ATS score.")
    lines.append("Results shown below.")
    return "\n".join(lines)
