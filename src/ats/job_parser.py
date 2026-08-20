"""Extract structured data from job description text using LLM."""
import logging

from src.ats.llm_client import complete_json
from src.ats.models import JDSkill, JobData

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract structured information from job descriptions. "
    "Return ONLY a JSON object matching the schema provided. "
    "Separate required vs preferred skills carefully. "
    "Do NOT treat every word as equally important."
)

_PROMPT_TEMPLATE = """Extract structured data from this job description. Return ONLY a JSON object with these EXACT keys:
{{
    "job_title": "string",
    "company": "string",
    "location": "string",
    "min_years_experience": 0.0,
    "required_education": "string or empty",
    "required_field": "string or empty",
    "required_skills": [
        {{"name": "string", "importance": "required"}}
    ],
    "preferred_skills": [
        {{"name": "string", "importance": "preferred"}}
    ],
    "required_technologies": ["string", ...],
    "preferred_technologies": ["string", ...],
    "certifications": ["string", ...],
    "domain": "string",
    "work_authorization_required": false,
    "work_authorization_detail": "string or empty"
}}

Rules:
- Do NOT add any keys not listed above
- Keep lists short (max 8 items). Only include what is explicitly stated.
- Distinguish REQUIRED from PREFERRED/NICE-TO-HAVE
- min_years_experience: extract the minimum years mentioned (e.g. "3+ years" → 3.0)
- If no experience requirement, use 0
- If no education requirement, leave required_education empty
- required_skills: hard skills explicitly required
- preferred_skills: skills listed as nice-to-have, bonus, etc.
- Do NOT guess. Only extract what is explicitly stated.

Job description:
---
{jd_text}
---
"""


def parse_job(jd_text: str) -> JobData:
    """Extract structured data from job description using LLM.

    Returns a JobData model with required/preferred skills separated.
    Falls back to basic extraction if LLM fails.
    """
    if not jd_text.strip():
        return JobData()

    # Truncate if very long (keep total prompt under the 4096-token model window).
    # First ~1500 chars of a JD hold title + requirements + skills.
    text = jd_text[:1500]

    try:
        data = complete_json(
            _PROMPT_TEMPLATE.format(jd_text=text),
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=4096,
        )
        return _to_model(data, jd_text)
    except Exception as e:
        logger.warning("LLM JD parsing failed, using fallback: %s", e)
        return _fallback_parse(jd_text)


def _to_model(data: dict, raw_text: str) -> JobData:
    """Convert raw LLM dict to JobData model."""
    try:
        required = [JDSkill(**s) for s in data.get("required_skills", [])]
        preferred = [JDSkill(**s) for s in data.get("preferred_skills", [])]

        return JobData(
            job_title=data.get("job_title", ""),
            company=data.get("company", ""),
            location=data.get("location", ""),
            min_years_experience=float(data.get("min_years_experience", 0)),
            required_education=data.get("required_education", ""),
            required_field=data.get("required_field", ""),
            required_skills=required,
            preferred_skills=preferred,
            required_technologies=data.get("required_technologies", []),
            preferred_technologies=data.get("preferred_technologies", []),
            certifications=data.get("certifications", []),
            domain=data.get("domain", ""),
            work_authorization_required=data.get("work_authorization_required", False),
            work_authorization_detail=data.get("work_authorization_detail", ""),
            raw_text=raw_text,
        )
    except Exception as e:
        logger.warning("Failed to parse JD data: %s", e)
        return _fallback_parse(raw_text)


def _fallback_parse(text: str) -> JobData:
    """Basic fallback when LLM fails."""
    import re

    lines = text.strip().split("\n")
    title = lines[0].strip() if lines else ""

    # Extract years requirement
    years = 0.0
    year_match = re.search(r"(\d+)\+?\s*years?", text.lower())
    if year_match:
        years = float(year_match.group(1))

    # Basic tech detection
    tech_keywords = [
        "python", "java", "javascript", "typescript", "c++", "go", "rust",
        "pytorch", "tensorflow", "keras", "opencv", "scikit-learn",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "sql", "postgresql", "mongodb", "redis",
        "react", "node.js", "django", "fastapi", "flask",
        "git", "linux", "ci/cd", "terraform",
    ]
    text_lower = text.lower()
    found_tech = [t for t in tech_keywords if t in text_lower]

    return JobData(
        job_title=title,
        min_years_experience=years,
        required_technologies=found_tech,
        raw_text=text,
    )
