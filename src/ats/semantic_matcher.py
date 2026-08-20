"""Semantic skill matching — classifies evidence strength using LLM."""
import logging

from src.ats.llm_client import complete_json
from src.ats.models import JobData, MatchLevel, ResumeData, SkillMatch

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You classify how well a resume demonstrates a specific skill. "
    "Return ONLY a JSON object with the classification. "
    "Be strict: vague mentions are WEAK, specific evidence is STRONG."
)

_PROMPT_TEMPLATE = """Classify how well this resume demonstrates the skill: "{skill}"

Classification levels:
- EXACT (1.0): Resume explicitly mentions this exact skill/tool/technology
- STRONG_SEMANTIC (0.8): Resume describes work that clearly requires this skill, using related terms
- PARTIAL (0.5): Resume shows some related experience but not direct demonstration
- WEAK (0.2): Resume has vague or indirect connection to this skill
- NONE (0.0): No evidence of this skill in the resume

Return ONLY a JSON object:
{{
    "match_level": "exact" | "strong_semantic" | "partial" | "weak" | "none",
    "score_value": 1.0 | 0.8 | 0.5 | 0.2 | 0.0,
    "evidence": "exact quote or paraphrase from resume showing this skill",
    "explanation": "brief explanation of why this classification"
}}

Resume:
---
{resume_text}
---
"""


def classify_skill_match(resume: ResumeData, skill_name: str) -> SkillMatch:
    """Classify how well the resume demonstrates a specific skill.

    Uses LLM for semantic understanding, not just keyword matching.
    """
    text = resume.raw_text[:1500]

    try:
        data = complete_json(
            _PROMPT_TEMPLATE.format(skill=skill_name, resume_text=text),
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=256,
        )
        level_str = data.get("match_level", "none")
        try:
            level = MatchLevel(level_str)
        except ValueError:
            level = MatchLevel.NONE

        return SkillMatch(
            skill_name=skill_name,
            match_level=level,
            score_value=float(data.get("score_value", 0.0)),
            evidence=data.get("evidence", ""),
            explanation=data.get("explanation", ""),
        )
    except Exception as e:
        logger.warning("LLM classification failed for '%s', using fallback: %s", skill_name, e)
        return _fallback_classify(resume, skill_name)


def _fallback_classify(resume: ResumeData, skill_name: str) -> SkillMatch:
    """Fallback: keyword matching when LLM fails."""
    text = resume.raw_text.lower()
    skill_lower = skill_name.lower()

    # Combine all skills the resume has
    all_skills = (
        resume.skills + resume.programming_languages + resume.frameworks
        + resume.tools + resume.cloud_technologies + resume.databases
    )
    all_skills_lower = [s.lower() for s in all_skills]

    # Exact match
    if skill_lower in all_skills_lower or skill_lower in text:
        return SkillMatch(
            skill_name=skill_name,
            match_level=MatchLevel.EXACT,
            score_value=1.0,
            evidence=f"Found '{skill_name}' in resume",
            explanation="Exact keyword match",
        )

    # Check if any skill contains this term or vice versa
    for s in all_skills_lower:
        if skill_lower in s or s in skill_lower:
            return SkillMatch(
                skill_name=skill_name,
                match_level=MatchLevel.STRONG_SEMANTIC,
                score_value=0.8,
                evidence=f"Related skill found: {s}",
                explanation="Partial keyword overlap",
            )

    return SkillMatch(
        skill_name=skill_name,
        match_level=MatchLevel.NONE,
        score_value=0.0,
        evidence="",
        explanation="No matching evidence found",
    )


def match_all_skills(
    resume: ResumeData,
    job: JobData,
) -> list[SkillMatch]:
    """Classify evidence for all JD skills against the resume.

    Processes both required and preferred skills.
    """
    matches = []

    # Required skills
    for skill in job.required_skills:
        match = classify_skill_match(resume, skill.name)
        matches.append(match)

    # Required technologies (treat as required skills)
    for tech in job.required_technologies:
        # Skip if already matched as a skill
        if not any(m.skill_name.lower() == tech.lower() for m in matches):
            match = classify_skill_match(resume, tech)
            matches.append(match)

    # Preferred skills
    for skill in job.preferred_skills:
        match = classify_skill_match(resume, skill.name)
        matches.append(match)

    # Preferred technologies
    for tech in job.preferred_technologies:
        if not any(m.skill_name.lower() == tech.lower() for m in matches):
            match = classify_skill_match(resume, tech)
            matches.append(match)

    return matches
