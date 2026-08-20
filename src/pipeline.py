"""Fetch jobs from Apify, normalize, embed, and store in SQLite."""

import logging

from src.db import init_db, insert_job
from src.embeddings import embed_texts
from src.normalize import normalize_items
from src.scraper import run_job_search

logger = logging.getLogger(__name__)


def scrape_and_store(
    query: str = "software engineer",
    location: str = "",
    country: str = "US",
    max_items: int = 30,
    max_age_days: int = 30,
    **kwargs,
) -> dict:
    """Run Apify scrape -> normalize -> embed -> insert. Returns summary stats."""
    init_db()

    raw_items = run_job_search(
        query=query,
        location=location,
        country=country,
        max_items=max_items,
        max_age_days=max_age_days,
        **kwargs,
    )
    if not raw_items:
        return {"fetched": 0, "new": 0, "duplicates": 0}

    jobs = normalize_items(raw_items)
    texts = [
        f"{j['title']}. {j['company']}. {j['location']}. {j['description'][:2000]} {j['skills']}"
        for j in jobs
    ]
    embeddings = embed_texts(texts)

    new = 0
    dup = 0
    for job, emb in zip(jobs, embeddings):
        if insert_job(job, embedding=emb):
            new += 1
        else:
            dup += 1

    return {"fetched": len(raw_items), "new": new, "duplicates": dup}
