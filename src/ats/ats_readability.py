"""ATS readability check — evaluates machine-parseable formatting."""
import re


def check_readability(resume_text: str) -> dict:
    """Check if the resume is likely to be parsed reliably by ATS systems.

    Returns:
        {
            "score": 0-5 (points for this category),
            "checks": [{"name": str, "passed": bool, "detail": str}, ...],
            "issues": [str, ...]
        }
    """
    checks = []
    issues = []
    score = 5.0  # Start at max, deduct for issues

    text = resume_text.strip()
    text_lower = text.lower()

    # ── 1. Standard section headings ──
    common_sections = [
        "experience", "education", "skills", "summary", "profile",
        "work", "employment", "projects", "certifications", "awards",
    ]
    found_sections = [s for s in common_sections if s in text_lower]
    has_headings = len(found_sections) >= 2
    checks.append({
        "name": "Standard section headings",
        "passed": has_headings,
        "detail": f"Found {len(found_sections)} standard sections: {', '.join(found_sections[:5])}",
    })
    if not has_headings:
        score -= 1.0
        issues.append("Missing standard section headings (Experience, Education, Skills)")

    # ── 2. Contact information ──
    has_email = bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text))
    has_phone = bool(re.search(r"[\+]?[\d\s\-\(\)]{7,}", text))
    has_contact = has_email or has_phone
    checks.append({
        "name": "Contact information",
        "passed": has_contact,
        "detail": f"Email: {'yes' if has_email else 'no'}, Phone: {'yes' if has_phone else 'no'}",
    })
    if not has_contact:
        score -= 0.5
        issues.append("No contact information found (email or phone)")

    # ── 3. Date format consistency ──
    date_patterns = [
        r"\b\d{4}\s*[-–]\s*(?:present|current|\d{4})\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",
        r"\b\d{1,2}/\d{4}\b",
    ]
    date_count = sum(len(re.findall(p, text_lower)) for p in date_patterns)
    has_dates = date_count >= 1
    checks.append({
        "name": "Date information",
        "passed": has_dates,
        "detail": f"Found {date_count} date references",
    })
    if not has_dates:
        score -= 0.5
        issues.append("No dates found — ATS cannot verify experience timeline")

    # ── 4. Excessive special characters ──
    special_chars = len(re.findall(r"[^\w\s\-\.\,\;\:\!\?\/\@\#\$\%\&\*\+\=\(\)\[\]\{\}\'\"]", text))
    special_ratio = special_chars / max(len(text), 1)
    ok_special = special_ratio < 0.02
    checks.append({
        "name": "Clean text formatting",
        "passed": ok_special,
        "detail": f"Special character ratio: {special_ratio:.3f}",
    })
    if not ok_special:
        score -= 0.5
        issues.append("Excessive special characters — may confuse ATS parsers")

    # ── 5. Tables/columns indicators ──
    # Heuristic: multiple pipe characters on same lines suggest table format
    lines = text.split("\n")
    pipe_lines = sum(1 for l in lines if l.count("|") >= 2)
    has_tables = pipe_lines > len(lines) * 0.1
    checks.append({
        "name": "Table-free formatting",
        "passed": not has_tables,
        "detail": f"{pipe_lines} lines with table-like formatting",
    })
    if has_tables:
        score -= 0.5
        issues.append("Table/column formatting detected — may not parse correctly")

    # ── 6. Readable text length ──
    word_count = len(text.split())
    ok_length = 100 <= word_count <= 1500
    checks.append({
        "name": "Appropriate length",
        "passed": ok_length,
        "detail": f"{word_count} words",
    })
    if not ok_length:
        if word_count < 100:
            score -= 1.0
            issues.append("Resume is too short — limited content for ATS to extract")
        else:
            score -= 0.5
            issues.append("Resume is very long — consider condensing")

    # ── 7. Consistent bullet points ──
    bullet_chars = set()
    for line in lines:
        stripped = line.strip()
        if stripped and stripped[0] in "-•●○▪▸►→*":
            bullet_chars.add(stripped[0])
    consistent_bullets = len(bullet_chars) <= 2
    checks.append({
        "name": "Consistent bullet formatting",
        "passed": consistent_bullets,
        "detail": f"Found {len(bullet_chars)} different bullet styles",
    })
    if not consistent_bullets:
        score -= 0.25
        issues.append("Inconsistent bullet point formatting")

    return {
        "score": max(0, round(score, 1)),
        "checks": checks,
        "issues": issues,
    }
