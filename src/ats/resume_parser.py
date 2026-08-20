"""Extract structured data from resume text using LLM."""
import logging

from src.ats.llm_client import complete_json
from src.ats.models import Education, ResumeData, WorkExperience

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract structured information from resumes. "
    "Return ONLY a JSON object matching the schema provided. "
    "Do NOT invent information. If a field is not present, use null or empty list."
)

_PROMPT_TEMPLATE = """Extract structured data from this resume. Return ONLY a JSON object with these EXACT keys:
{{
    "candidate_name": "string",
    "current_title": "string or empty",
    "previous_titles": ["string", ...],
    "total_years_experience": 0.0,
    "relevant_years_experience": 0.0,
    "education": [
        {{"degree": "string", "field": "string", "institution": "string", "year": "string"}}
    ],
    "degrees": ["string", ...],
    "skills": ["string", ...],
    "programming_languages": ["string", ...],
    "frameworks": ["string", ...],
    "tools": ["string", ...],
    "cloud_technologies": ["string", ...],
    "databases": ["string", ...],
    "certifications": ["string", ...],
    "locations": ["string", ...],
    "work_authorization": "string or empty"
}}

Rules:
- Do NOT add any keys not listed above
- Keep lists short (max 10 items). Only include skills that are explicitly stated.
- total_years_experience: calculate from work history dates
- relevant_years_experience: years in roles related to tech/ML/engineering (or the field of the resume)
- Do NOT guess or infer information not explicitly stated

Resume text:
---
{resume_text}
---
"""


def parse_resume(resume_text: str) -> ResumeData:
    """Extract structured data from resume text using LLM.

    Returns a ResumeData model with all extracted fields.
    Falls back to basic extraction if LLM fails.
    """
    if not resume_text.strip():
        return ResumeData()

    # Truncate if very long (keep total prompt under the 4096-token model window).
    # The first ~1500 chars of a resume contain summary + skills + experience.
    text = resume_text[:1500]

    try:
        data = complete_json(
            _PROMPT_TEMPLATE.format(resume_text=text),
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=4096,
        )
        return _to_model(data, resume_text)
    except Exception as e:
        logger.warning("LLM resume parsing failed, using fallback: %s", e)
        return _fallback_parse(resume_text)


def _to_model(data: dict, raw_text: str) -> ResumeData:
    """Convert raw LLM dict to ResumeData model."""
    try:
        education = [Education(**e) for e in data.get("education", [])]
        return ResumeData(
            candidate_name=data.get("candidate_name", ""),
            current_title=data.get("current_title", ""),
            previous_titles=data.get("previous_titles", []),
            total_years_experience=float(data.get("total_years_experience", 0)),
            relevant_years_experience=float(data.get("relevant_years_experience", 0)),
            education=education,
            degrees=data.get("degrees", []),
            skills=data.get("skills", []),
            programming_languages=data.get("programming_languages", []),
            frameworks=data.get("frameworks", []),
            tools=data.get("tools", []),
            cloud_technologies=data.get("cloud_technologies", []),
            databases=data.get("databases", []),
            certifications=data.get("certifications", []),
            locations=data.get("locations", []),
            work_authorization=data.get("work_authorization", ""),
            raw_text=raw_text,
        )
    except Exception as e:
        logger.warning("Failed to parse resume data: %s", e)
        return _fallback_parse(raw_text)


def _fallback_parse(text: str) -> ResumeData:
    """Basic regex/heuristic fallback when LLM fails."""
    import re

    # Extract years of experience mention
    years = 0.0
    year_match = re.search(r"(\d+)\+?\s*years?\s*(?:of\s+)?experience", text.lower())
    if year_match:
        years = float(year_match.group(1))

    # Basic skill detection
    skill_keywords = [
        "python", "java", "javascript", "typescript", "c++", "go", "rust",
        "pytorch", "tensorflow", "keras", "opencv", "scikit-learn",
        "machine learning", "deep learning", "computer vision", "nlp", "llm",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "sql", "postgresql", "mongodb", "redis",
        "react", "node.js", "django", "fastapi", "flask",
        "git", "linux", "ci/cd", "terraform",
    ]
    text_lower = text.lower()
    found_skills = [sk for sk in skill_keywords if sk in text_lower]

    # Basic education detection
    education = []
    if re.search(r"ph\.?d|doctorate", text_lower):
        education.append(Education(degree="PhD", field=""))
    elif re.search(r"master|m\.?s\.?|m\.?tech|mba", text_lower):
        education.append(Education(degree="Master's", field=""))
    elif re.search(r"bachelor|b\.?s\.?|b\.?tech|b\.?e\.", text_lower):
        education.append(Education(degree="Bachelor's", field=""))

    return ResumeData(
        total_years_experience=years,
        relevant_years_experience=years,
        skills=found_skills,
        programming_languages=[s for s in found_skills if s in {"python", "java", "javascript", "typescript", "c++", "go", "rust"}],
        education=education,
        raw_text=text,
    )
