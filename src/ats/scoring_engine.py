"""Deterministic scoring engine — computes 0-100 from structured data.

Architecture:
    Resume → LLM → Structured Evidence
    JD → LLM → Structured Requirements
    ↓
    Semantic Matcher → evidence classification (LLM)
    ↓
    THIS MODULE → Deterministic scoring (Python math)
    ↓
    LLM → Explanation & recommendations

No LLM calls in scoring. Pure math. Deterministic and reproducible.
"""
import logging

from src.ats.ats_readability import check_readability
from src.ats.job_parser import parse_job
from src.ats.models import (
    CategoryScore,
    HardBlocker,
    JobData,
    MatchLevel,
    ResumeData,
    ScoreCategory,
    ScoringResult,
    SkillMatch,
)
from src.ats.recommendation_engine import generate_recommendations
from src.ats.resume_parser import parse_resume
from src.ats.semantic_matcher import match_all_skills

logger = logging.getLogger(__name__)

# ── Category weights (must sum to 100) ─────────────────────────────────────

CATEGORY_MAX = {
    ScoreCategory.REQUIRED_SKILLS: 30,
    ScoreCategory.EXPERIENCE: 20,
    ScoreCategory.ROLE_ALIGNMENT: 10,
    ScoreCategory.PREFERRED_SKILLS: 10,
    ScoreCategory.EDUCATION: 10,
    ScoreCategory.KEYWORD_RELEVANCE: 10,
    ScoreCategory.CERTIFICATIONS: 5,
    ScoreCategory.ATS_READABILITY: 5,
}

# ── Match level scores ─────────────────────────────────────────────────────

MATCH_SCORES = {
    MatchLevel.EXACT: 1.0,
    MatchLevel.STRONG_SEMANTIC: 0.8,
    MatchLevel.PARTIAL: 0.5,
    MatchLevel.WEAK: 0.2,
    MatchLevel.NONE: 0.0,
}


# ── Main entry point ───────────────────────────────────────────────────────


def score_resume_vs_jd(
    resume_text: str,
    jd_text: str,
) -> ScoringResult:
    """Score a resume against a job description.

    Flow:
        1. Parse resume → ResumeData (LLM)
        2. Parse JD → JobData (LLM)
        3. Match skills → SkillMatch list (LLM)
        4. Score categories → deterministic math
        5. Check readability → formatting analysis
        6. Identify hard blockers
        7. Generate recommendations

    Returns a complete ScoringResult.
    """
    # Step 1 & 2: Parse both documents
    logger.info("Parsing resume...")
    resume = parse_resume(resume_text)

    logger.info("Parsing job description...")
    job = parse_job(jd_text)

    # Step 3: Semantic skill matching
    logger.info("Classifying skill matches...")
    all_matches = match_all_skills(resume, job)

    # Step 4: Score each category
    category_scores = {}

    # A. Required Skills (30 pts)
    category_scores[ScoreCategory.REQUIRED_SKILLS] = _score_required_skills(
        all_matches, job
    )

    # B. Experience (20 pts)
    category_scores[ScoreCategory.EXPERIENCE] = _score_experience(resume, job)

    # C. Role Alignment (10 pts)
    category_scores[ScoreCategory.ROLE_ALIGNMENT] = _score_role_alignment(
        resume, job
    )

    # D. Preferred Skills (10 pts)
    category_scores[ScoreCategory.PREFERRED_SKILLS] = _score_preferred_skills(
        all_matches, job
    )

    # E. Education (10 pts)
    category_scores[ScoreCategory.EDUCATION] = _score_education(resume, job)

    # F. Keyword/Semantic Relevance (10 pts)
    category_scores[ScoreCategory.KEYWORD_RELEVANCE] = _score_keyword_relevance(
        all_matches, job
    )

    # G. Certifications (5 pts)
    category_scores[ScoreCategory.CERTIFICATIONS] = _score_certifications(
        resume, job
    )

    # H. ATS Readability (5 pts)
    readability = check_readability(resume_text)
    category_scores[ScoreCategory.ATS_READABILITY] = CategoryScore(
        category=ScoreCategory.ATS_READABILITY,
        score=readability["score"],
        max_points=5,
        percentage=round((readability["score"] / 5) * 100),
        details="; ".join(readability["issues"]) if readability["issues"] else "Good formatting",
    )

    # Step 5: Calculate overall score
    total = sum(cs.score for cs in category_scores.values())
    overall = min(100, max(0, int(round(total))))

    # Step 6: Identify hard blockers
    hard_blockers = _find_hard_blockers(resume, job)

    # Step 7: Classify matches
    strong = [m for m in all_matches if m.match_level in (MatchLevel.EXACT, MatchLevel.STRONG_SEMANTIC)]
    partial = [m for m in all_matches if m.match_level == MatchLevel.PARTIAL]
    missing = [m for m in all_matches if m.match_level == MatchLevel.NONE]

    # Step 8: Build result
    result = ScoringResult(
        overall_score=overall,
        category_scores={cs.category.value: cs for cs in category_scores.values()},
        strong_matches=strong,
        partial_matches=partial,
        missing_requirements=missing,
        hard_blockers=hard_blockers,
        resume_data=resume,
        job_data=job,
    )

    # Step 9: Generate recommendations
    result.recommendations = generate_recommendations(result, resume, job, all_matches)

    return result


# ── Category scoring functions ──────────────────────────────────────────────


def _score_required_skills(
    matches: list[SkillMatch],
    job: JobData,
) -> CategoryScore:
    """A. Required Skills — 30 points.

    Weighted average of required skill match levels.
    """
    max_pts = CATEGORY_MAX[ScoreCategory.REQUIRED_SKILLS]

    # Collect required skills (from required_skills + required_technologies)
    required_names = {s.name.lower() for s in job.required_skills}
    required_names.update(t.lower() for t in job.required_technologies)

    required_matches = [m for m in matches if m.skill_name.lower() in required_names]

    if not required_matches:
        return CategoryScore(
            category=ScoreCategory.REQUIRED_SKILLS,
            score=max_pts,  # No requirements = full points (can't penalize)
            max_points=max_pts,
            percentage=100,
            details="No required skills specified",
        )

    avg = sum(MATCH_SCORES[m.match_level] for m in required_matches) / len(required_matches)
    score = max_pts * avg

    return CategoryScore(
        category=ScoreCategory.REQUIRED_SKILLS,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(avg * 100),
        details=f"{len(required_matches)} required skills evaluated",
    )


def _score_experience(resume: ResumeData, job: JobData) -> CategoryScore:
    """B. Relevant Experience — 20 points.

    Compares required years against candidate's relevant experience.
    """
    max_pts = CATEGORY_MAX[ScoreCategory.EXPERIENCE]

    if job.min_years_experience <= 0:
        return CategoryScore(
            category=ScoreCategory.EXPERIENCE,
            score=max_pts,
            max_points=max_pts,
            percentage=100,
            details="No experience requirement specified",
        )

    candidate_years = resume.relevant_years_experience or resume.total_years_experience

    if candidate_years >= job.min_years_experience:
        ratio = 1.0
    else:
        ratio = candidate_years / job.min_years_experience

    score = max_pts * min(ratio, 1.0)

    return CategoryScore(
        category=ScoreCategory.EXPERIENCE,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(ratio * 100),
        details=f"Candidate: {candidate_years:.0f}y relevant, Required: {job.min_years_experience:.0f}y",
    )


def _score_role_alignment(resume: ResumeData, job: JobData) -> CategoryScore:
    """C. Job Title / Role Alignment — 10 points.

    Compares candidate's titles against the target role.
    """
    max_pts = CATEGORY_MAX[ScoreCategory.ROLE_ALIGNMENT]

    if not job.job_title:
        return CategoryScore(
            category=ScoreCategory.ROLE_ALIGNMENT,
            score=max_pts,
            max_points=max_pts,
            percentage=100,
            details="No target role specified",
        )

    # Build candidate title set
    candidate_titles = [resume.current_title.lower()]
    candidate_titles.extend(t.lower() for t in resume.previous_titles)

    job_title_lower = job.job_title.lower()

    # Check for overlap
    best_ratio = 0.0

    for title in candidate_titles:
        if not title:
            continue
        # Exact match
        if title == job_title_lower:
            best_ratio = 1.0
            break
        # Substring match
        if job_title_lower in title or title in job_title_lower:
            best_ratio = max(best_ratio, 0.9)
            continue
        # Word overlap
        job_words = set(job_title_lower.split())
        title_words = set(title.split())
        if job_words and title_words:
            overlap = len(job_words & title_words) / len(job_words)
            best_ratio = max(best_ratio, overlap * 0.8)

    # If no title match but resume has relevant skills, give partial credit
    if best_ratio == 0 and resume.skills:
        best_ratio = 0.3  # Base credit for having a tech resume

    score = max_pts * best_ratio

    return CategoryScore(
        category=ScoreCategory.ROLE_ALIGNMENT,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(best_ratio * 100),
        details=f"Target: '{job.job_title}', Candidate: '{resume.current_title}'",
    )


def _score_preferred_skills(
    matches: list[SkillMatch],
    job: JobData,
) -> CategoryScore:
    """D. Preferred Skills — 10 points.

    Same as required but for nice-to-have skills.
    Missing preferred hurts less than missing required.
    """
    max_pts = CATEGORY_MAX[ScoreCategory.PREFERRED_SKILLS]

    preferred_names = {s.name.lower() for s in job.preferred_skills}
    preferred_names.update(t.lower() for t in job.preferred_technologies)

    preferred_matches = [m for m in matches if m.skill_name.lower() in preferred_names]

    if not preferred_matches:
        return CategoryScore(
            category=ScoreCategory.PREFERRED_SKILLS,
            score=max_pts,
            max_points=max_pts,
            percentage=100,
            details="No preferred skills specified",
        )

    avg = sum(MATCH_SCORES[m.match_level] for m in preferred_matches) / len(preferred_matches)
    score = max_pts * avg

    return CategoryScore(
        category=ScoreCategory.PREFERRED_SKILLS,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(avg * 100),
        details=f"{len(preferred_matches)} preferred skills evaluated",
    )


def _score_education(resume: ResumeData, job: JobData) -> CategoryScore:
    """E. Education — 10 points.

    Evaluates degree and field relevance.
    """
    max_pts = CATEGORY_MAX[ScoreCategory.EDUCATION]

    if not job.required_education:
        return CategoryScore(
            category=ScoreCategory.EDUCATION,
            score=max_pts,
            max_points=max_pts,
            percentage=100,
            details="No education requirement specified",
        )

    # Degree hierarchy
    degree_order = {
        "phd": 4, "doctorate": 4, "ph.d": 4, "ph.d.": 4,
        "master": 3, "m.s": 3, "m.s.": 3, "m.tech": 3, "mba": 3, "m.sc": 3, "m.sc.": 3,
        "msc": 3, "meng": 3, "m.eng": 3,
        "bachelor": 2, "b.s": 2, "b.s.": 2, "b.tech": 2, "b.e": 2, "b.sc": 2, "b.sc.": 2,
        "bsc": 2, "beng": 2, "b.eng": 2,
        "associate": 1, "diploma": 1,
    }

    # Required degree level
    req_level = 0
    for key, level in degree_order.items():
        if key in job.required_education.lower():
            req_level = max(req_level, level)

    # Candidate degree level
    cand_level = 0
    for edu in resume.education:
        for key, level in degree_order.items():
            if key in edu.degree.lower():
                cand_level = max(cand_level, level)

    if req_level == 0:
        ratio = 1.0
    elif cand_level >= req_level:
        ratio = 1.0
    elif cand_level > 0:
        ratio = cand_level / req_level
    else:
        ratio = 0.0

    # Field check
    if job.required_field and resume.education:
        # Normalize common abbreviations
        field_aliases = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "cv": "computer vision",
            "nlp": "natural language processing",
            "ds": "data science",
            "se": "software engineering",
            "cs": "computer science",
        }
        req_field_lower = job.required_field.lower()
        req_field_expanded = field_aliases.get(req_field_lower, req_field_lower)

        field_match = False
        for e in resume.education:
            edu_text = (e.field.lower() + " " + e.degree.lower())
            if req_field_lower in edu_text or req_field_expanded in edu_text:
                field_match = True
                break

        if not field_match and ratio > 0:
            ratio *= 0.8  # Slight penalty for wrong field

    score = max_pts * ratio

    return CategoryScore(
        category=ScoreCategory.EDUCATION,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(ratio * 100),
        details=f"Required: {job.required_education}, Candidate: {[e.degree for e in resume.education]}",
    )


def _score_keyword_relevance(
    matches: list[SkillMatch],
    job: JobData,
) -> CategoryScore:
    """F. Keyword / Semantic Relevance — 10 points.

    How well the resume contains important JD terminology overall.
    """
    max_pts = CATEGORY_MAX[ScoreCategory.KEYWORD_RELEVANCE]

    if not matches:
        return CategoryScore(
            category=ScoreCategory.KEYWORD_RELEVANCE,
            score=0,
            max_points=max_pts,
            percentage=0,
            details="No skill matches to evaluate",
        )

    # Overall match quality across ALL skills
    avg = sum(MATCH_SCORES[m.match_level] for m in matches) / len(matches)

    # Bonus for having measurable results (keyword-rich resumes)
    if len(matches) > 0:
        strong_ratio = sum(1 for m in matches if m.match_level in (MatchLevel.EXACT, MatchLevel.STRONG_SEMANTIC)) / len(matches)
        avg = avg * 0.8 + strong_ratio * 0.2  # Blend average with strong-match ratio

    score = max_pts * avg

    return CategoryScore(
        category=ScoreCategory.KEYWORD_RELEVANCE,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(avg * 100),
        details=f"{len(matches)} total skills evaluated",
    )


def _score_certifications(resume: ResumeData, job: JobData) -> CategoryScore:
    """G. Certifications / Additional Requirements — 5 points."""
    max_pts = CATEGORY_MAX[ScoreCategory.CERTIFICATIONS]

    if not job.certifications:
        return CategoryScore(
            category=ScoreCategory.CERTIFICATIONS,
            score=max_pts,
            max_points=max_pts,
            percentage=100,
            details="No certifications required",
        )

    matched = [c for c in job.certifications if c in resume.certifications]
    ratio = len(matched) / len(job.certifications) if job.certifications else 1.0
    score = max_pts * ratio

    return CategoryScore(
        category=ScoreCategory.CERTIFICATIONS,
        score=round(score, 1),
        max_points=max_pts,
        percentage=round(ratio * 100),
        details=f"Required: {job.certifications}, Matched: {matched}",
    )


# ── Hard blockers ───────────────────────────────────────────────────────────


def _find_hard_blockers(
    resume: ResumeData,
    job: JobData,
) -> list[HardBlocker]:
    """Identify hard blockers that significantly hurt the candidate's chances."""
    blockers = []

    # Work authorization
    if job.work_authorization_required:
        if not resume.work_authorization:
            blockers.append(HardBlocker(
                requirement="Work authorization",
                status="not_specified",
                explanation=(
                    f"Job requires {job.work_authorization_detail or 'work authorization'}; "
                    "resume does not indicate eligibility."
                ),
            ))
        elif job.work_authorization_detail.lower() not in resume.work_authorization.lower():
            blockers.append(HardBlocker(
                requirement="Work authorization",
                status="mismatch",
                explanation=(
                    f"Job requires {job.work_authorization_detail}, "
                    f"resume states: {resume.work_authorization}"
                ),
            ))

    # Severe experience gap (>50% below required)
    if job.min_years_experience > 0:
        cand_years = resume.relevant_years_experience or resume.total_years_experience
        if cand_years < job.min_years_experience * 0.5:
            blockers.append(HardBlocker(
                requirement="Experience",
                status="below_minimum",
                explanation=(
                    f"Job requires {job.min_years_experience:.0f}+ years; "
                    f"candidate has ~{cand_years:.0f} years of relevant experience."
                ),
            ))

    # Missing required certification (explicitly required)
    for cert in job.certifications:
        if cert not in resume.certifications:
            blockers.append(HardBlocker(
                requirement=f"Certification: {cert}",
                status="missing",
                explanation=f"Required certification '{cert}' not found on resume.",
            ))

    return blockers
