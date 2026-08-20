"""CLI: scrape jobs from Apify into the local database.

Usage:
    python run_scrape.py --query "computer vision" --location Denmark --country DK --max-items 50
    python run_scrape.py --seed        # load demo data, no API key needed
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Scrape jobs via Apify into local SQLite.")
    parser.add_argument("--query", default="computer vision OR football")
    parser.add_argument("--location", default="")
    parser.add_argument("--max-items", type=int, default=20, help="Max jobs to scrape")
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        choices=[1, 7, 30],
        help="Only fetch jobs posted within this many days (LinkedIn filter)",
    )
    parser.add_argument("--seed", action="store_true", help="Load demo data instead of scraping.")
    args = parser.parse_args()

    if args.seed:
        from src.seed import seed_demo
        from src.db import stats

        result = seed_demo()
        print(f"Seeded {result['new']} new demo jobs. {stats()['total']} total in DB.")
        return

    from src import pipeline
    from src.db import stats

    result = pipeline.scrape_and_store(
        query=args.query,
        location=args.location,
        max_items=args.max_items,
        max_age_days=args.max_age_days,
    )
    print(
        f"Fetched {result['fetched']} items, added {result['new']} new jobs "
        f"({result['duplicates']} duplicates)."
    )
    print(stats())


if __name__ == "__main__":
    main()
