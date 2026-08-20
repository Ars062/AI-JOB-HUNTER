"""Resume parsing and skill extraction — used by the ATS scorer and app UI."""

import re

from pypdf import PdfReader

from src.ats_scorer import SKILLS_DB


def parse_resume(path: str) -> str:
    """Extract text from a resume file (PDF or TXT)."""
    text = ""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text.strip()


def extract_skills(resume_text: str, keywords: list[str] | None = None) -> list[str]:
    """Heuristic skill extraction from a resume (for display + ATS matching).
    Uses the shared SKILLS_DB so the ATS gate and search use the same vocabulary.
    """
    if not keywords:
        keywords = sorted(SKILLS_DB)
    text_l = resume_text.lower()
    found = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_l)]
    return found
