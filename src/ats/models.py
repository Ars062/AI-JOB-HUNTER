"""Pydantic models for structured resume and JD data."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ── Enums ───────────────────────────────────────────────────────────────────


class MatchLevel(str, Enum):
    EXACT = "exact"
    STRONG_SEMANTIC = "strong_semantic"
    PARTIAL = "partial"
    WEAK = "weak"
    NONE = "none"


class ScoreCategory(str, Enum):
    REQUIRED_SKILLS = "required_skills"
    EXPERIENCE = "experience"
    ROLE_ALIGNMENT = "role_alignment"
    PREFERRED_SKILLS = "preferred_skills"
    EDUCATION = "education"
    KEYWORD_RELEVANCE = "keyword_relevance"
    CERTIFICATIONS = "certifications"
    ATS_READABILITY = "ats_readability"


# ── Resume Structured Data ──────────────────────────────────────────────────


class Education(BaseModel):
    degree: str = ""
    field: str = ""
    institution: str = ""
    year: str = ""

    @model_validator(mode="before")
    @classmethod
    def _fix_none(cls, data):
        if isinstance(data, dict):
            for key in ["degree", "field", "institution", "year"]:
                if data.get(key) is None:
                    data[key] = ""
        return data


class WorkExperience(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    years: float = 0.0
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """Structured data extracted from a resume by LLM."""
    candidate_name: str = ""
    current_title: str = ""
    previous_titles: list[str] = Field(default_factory=list)
    total_years_experience: float = 0.0
    relevant_years_experience: float = 0.0

    education: list[Education] = Field(default_factory=list)
    degrees: list[str] = Field(default_factory=list)

    skills: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    cloud_technologies: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)

    certifications: list[str] = Field(default_factory=list)

    projects: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)

    achievements: list[str] = Field(default_factory=list)
    measurable_results: list[str] = Field(default_factory=list)

    locations: list[str] = Field(default_factory=list)
    work_authorization: str = ""

    raw_text: str = ""

    @model_validator(mode="before")
    @classmethod
    def _fix_none_strings(cls, data):
        if isinstance(data, dict):
            for key, val in data.items():
                if val is None and key in {
                    "candidate_name", "current_title", "work_authorization", "raw_text"
                }:
                    data[key] = ""
        return data


# ── Job Description Structured Data ─────────────────────────────────────────


class JDSkill(BaseModel):
    name: str
    importance: str = "required"  # "required" or "preferred"
    evidence_required: bool = True


class JobData(BaseModel):
    """Structured data extracted from a job description by LLM."""
    job_title: str = ""
    company: str = ""
    location: str = ""

    min_years_experience: float = 0.0
    required_education: str = ""
    required_field: str = ""

    required_skills: list[JDSkill] = Field(default_factory=list)
    preferred_skills: list[JDSkill] = Field(default_factory=list)

    required_technologies: list[str] = Field(default_factory=list)
    preferred_technologies: list[str] = Field(default_factory=list)

    certifications: list[str] = Field(default_factory=list)

    responsibilities: list[str] = Field(default_factory=list)
    domain: str = ""
    soft_skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    work_authorization_required: bool = False
    work_authorization_detail: str = ""

    raw_text: str = ""


# ── Semantic Match Result ───────────────────────────────────────────────────


class SkillMatch(BaseModel):
    """Result of matching one JD skill against the resume."""
    skill_name: str
    match_level: MatchLevel
    score_value: float = 0.0  # 1.0 / 0.8 / 0.5 / 0.2 / 0.0
    evidence: str = ""  # quote from resume
    explanation: str = ""


# ── Scoring Result ──────────────────────────────────────────────────────────


class CategoryScore(BaseModel):
    category: ScoreCategory
    score: float  # points earned (out of max)
    max_points: float
    percentage: float  # 0-100
    details: str = ""


class HardBlocker(BaseModel):
    requirement: str
    status: str  # "missing", "not_specified", "below_minimum"
    explanation: str


class ScoringResult(BaseModel):
    """Complete scoring result — deterministic, reproducible."""
    overall_score: int  # 0-100

    category_scores: dict[str, CategoryScore] = Field(default_factory=dict)

    strong_matches: list[SkillMatch] = Field(default_factory=list)
    partial_matches: list[SkillMatch] = Field(default_factory=list)
    missing_requirements: list[SkillMatch] = Field(default_factory=list)

    hard_blockers: list[HardBlocker] = Field(default_factory=list)

    recommendations: list[str] = Field(default_factory=list)

    resume_data: ResumeData | None = None
    job_data: JobData | None = None

    disclaimer: str = (
        "This is an ATS-style Resume–Job Match Score and does not represent "
        "the proprietary score of any specific ATS."
    )
