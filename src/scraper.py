import logging
from datetime import timedelta
from urllib.parse import urlencode

from config import APIFY_API_KEY

logger = logging.getLogger(__name__)


class ApifyError(Exception):
    pass


def build_linkedin_search_url(query: str = "", location: str = "", max_age_days: int = 30) -> str:
    """Build a public LinkedIn jobs search URL. All params optional.

    f_TPR (time posted range): r86400=24h, r604800=7d, r2592000=30d.
    max_age_days <= 0 omits the date filter (Anytime).
    """
    params = {}
    if query:
        params["keywords"] = query
    if location:
        params["location"] = location
    if max_age_days and max_age_days > 0:
        days_to_r = {1: "r86400", 7: "r604800", 30: "r2592000"}
        r = min((d for d in days_to_r if d >= max_age_days), default="r2592000")
        params["f_TPR"] = days_to_r[r]
    return "https://www.linkedin.com/jobs/search/?" + urlencode(params)


def run_job_search(
    query: str = "software engineer",
    location: str = "",
    country: str = "US",
    max_items: int = 30,
    max_age_days: int = 30,
    actor: str | None = None,
    extra_input: dict | None = None,
    wait_for_finish: int = 600,
) -> list[dict]:
    """Run a job-scraping actor and return raw dataset items.

    Default actor: curious_coder/linkedin-jobs-scraper (LinkedIn, pay-per-event)
    https://apify.com/curious_coder/linkedin-jobs-scraper

    Its input schema:
        urls:        [str]  LinkedIn jobs search URLs
        scrapeCompany: bool
        count:       int    max jobs to scrape
        splitByLocation: bool  split search by cities to bypass the 1000-job limit
    """
    if not APIFY_API_KEY:
        raise ApifyError("APIFY_API_KEY is not set. Add it to your .env file.")

    from apify_client import ApifyClient

    client = ApifyClient(APIFY_API_KEY)
    actor_id = actor or "curious_coder/linkedin-jobs-scraper"

    search_url = build_linkedin_search_url(query, location or country, max_age_days)
    actor_input: dict = {
        "urls": [search_url],
        "scrapeCompany": False,
        "count": max(10, int(max_items)),
        "splitByLocation": False,
    }
    actor_input.update(extra_input or {})

    run = client.actor(actor_id).call(
        run_input=actor_input, wait_duration=timedelta(seconds=wait_for_finish)
    )

    status = run.status if not isinstance(run, dict) else run.get("status")
    if run is None or status != "SUCCEEDED":
        raise ApifyError(f"Apify run failed or timed out: {run}")

    dataset_id = (
        run.default_dataset_id
        if not isinstance(run, dict)
        else run.get("defaultDatasetId")
    )
    dataset = client.dataset(dataset_id)
    return list(dataset.iterate_items())
