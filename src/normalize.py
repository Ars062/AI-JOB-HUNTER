"""Normalize raw items from babbucha/job-search into our DB schema."""


def normalize_item(item: dict) -> dict | None:
    """Map a raw Apify job item to our schema. Returns None if unusable."""
    title = item.get("title") or item.get("positionName") or item.get("jobTitle")
    url = item.get("url") or item.get("link") or item.get("jobLink") or item.get("postingUrl")
    if not title or not url:
        return None

    company = (
        item.get("company")
        or item.get("companyName")
        or item.get("employer")
        or item.get("organization")
        or "Unknown"
    )

    description = (
        item.get("description")
        or item.get("descriptionText")
        or item.get("jobDescription")
        or item.get("snippet")
        or ""
    )

    location = item.get("location") or item.get("locationText") or item.get("city") or ""
    if isinstance(location, (list, dict)):
        location = str(location)

    skills = item.get("skills") or item.get("keySkills") or ""
    if isinstance(skills, (list, tuple)):
        skills = ", ".join(str(s) for s in skills)

    salary = item.get("salary") or item.get("salaryInfo") or item.get("salaryMin") or ""
    if isinstance(salary, (list, dict)):
        salary = ", ".join(str(s) for s in salary) if isinstance(salary, list) else str(salary)

    job_type = item.get("employmentType") or item.get("jobType") or ""

    posted_date = (
        item.get("datePosted")
        or item.get("postedDate")
        or item.get("postedAt")
        or item.get("publicationDate")
        or item.get("scrapedAt")
        or ""
    )

    source = item.get("source") or item.get("provider") or _infer_source(url)
    if not source or (source == "linkedin" and company == "Unknown"):
        source = "linkedin" if "linkedin" in url else "apify"

    return {
        "title": str(title)[:500],
        "company": str(company)[:300],
        "location": str(location)[:300],
        "description": str(description)[:20000],
        "url": str(url)[:1000],
        "source": str(source)[:100],
        "salary": str(salary)[:200],
        "job_type": str(job_type)[:200],
        "skills": str(skills)[:1000],
        "posted_date": str(posted_date)[:100],
    }


def _infer_source(url: str) -> str:
    if "linkedin.com" in url:
        return "linkedin"
    if "indeed.com" in url:
        return "indeed"
    if "greenhouse.io" in url:
        return "greenhouse"
    if "jobs.lever.co" in url:
        return "lever"
    return ""


def normalize_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for item in items:
        norm = normalize_item(item)
        if norm is None:
            continue
        key = norm["url"]
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out
