"""Generate actionable recommendations based on scoring results."""
from src.ats.models import JobData, MatchLevel, ResumeData, ScoringResult, SkillMatch


def generate_recommendations(
    result: ScoringResult,
    resume: ResumeData,
    job: JobData,
    all_matches: list[SkillMatch],
) -> list[str]:
    """Generate prioritized, actionable recommendations.

    Returns a list of strings, ordered by impact (highest first).
    """
    recs = []

    # ── Hard blockers first ──
    for blocker in result.hard_blockers:
        recs.append(f"CRITICAL: {blocker.explanation}")

    # ── Missing required skills ──
    missing_required = [
        m for m in all_matches
        if m.match_level == MatchLevel.NONE
        and m.skill_name in [s.name for s in job.required_skills]
    ]
    if missing_required:
        skills = ", ".join(m.skill_name for m in missing_required[:5])
        recs.append(
            f"Add these required skills to your resume if you have them: {skills}. "
            "Use the exact terms from the job description."
        )

    # ── Partial matches — strengthen evidence ──
    partial = [
        m for m in all_matches
        if m.match_level in (MatchLevel.PARTIAL, MatchLevel.WEAK)
    ]
    if partial:
        skills = ", ".join(m.skill_name for m in partial[:3])
        recs.append(
            f"Strengthen evidence for: {skills}. "
            "Add specific examples, projects, or metrics that demonstrate these skills."
        )

    # ── Experience gap ──
    if job.min_years_experience > 0:
        if resume.relevant_years_experience < job.min_years_experience:
            gap = job.min_years_experience - resume.relevant_years_experience
            recs.append(
                f"You have {resume.relevant_years_experience:.0f} years of relevant experience "
                f"but the role requires {job.min_years_experience:.0f}+. "
                "Highlight related projects, internships, or transferable experience."
            )

    # ── Missing preferred skills ──
    missing_preferred = [
        m for m in all_matches
        if m.match_level == MatchLevel.NONE
        and m.skill_name in [s.name for s in job.preferred_skills]
    ]
    if missing_preferred:
        skills = ", ".join(m.skill_name for m in missing_preferred[:3])
        recs.append(
            f"Nice-to-have skills you could mention: {skills}. "
            "Even partial experience counts for preferred skills."
        )

    # ── No measurable results ──
    if not resume.measurable_results:
        recs.append(
            "Add quantified achievements to your resume. "
            "Use the XYZ formula: 'Accomplished [X] as measured by [Y] by doing [Z]'. "
            "Example: 'Reduced API latency by 40% by implementing Redis caching.'"
        )

    # ── Education ──
    if job.required_education and not resume.degrees:
        recs.append(
            f"The job requires {job.required_education}. "
            "If you have this degree, make sure it's clearly listed in your resume."
        )

    # ── Certifications ──
    missing_certs = [c for c in job.certifications if c not in resume.certifications]
    if missing_certs:
        recs.append(
            f"Consider obtaining: {', '.join(missing_certs[:2])}. "
            "Certifications can be a differentiator."
        )

    # ── Domain/industry alignment ──
    if job.domain and job.domain.lower() not in [i.lower() for i in resume.industries]:
        recs.append(
            f"This role is in {job.domain}. "
            "If you have relevant domain experience, highlight it prominently."
        )

    # ── General formatting ──
    if result.category_scores.get("ats_readability"):
        read_score = result.category_scores["ats_readability"].percentage
        if read_score < 70:
            recs.append(
                "Your resume formatting may cause parsing issues. "
                "Use standard section headings, avoid tables, and ensure clean text."
            )

    # ── Top matched skills — highlight them ──
    strong = [m for m in all_matches if m.match_level in (MatchLevel.EXACT, MatchLevel.STRONG_SEMANTIC)]
    if strong:
        skills = ", ".join(m.skill_name for m in strong[:5])
        recs.append(
            f"Your strongest matches: {skills}. "
            "Make sure these are prominent in your resume summary and experience sections."
        )

    return recs[:10]  # Cap at 10 recommendations
