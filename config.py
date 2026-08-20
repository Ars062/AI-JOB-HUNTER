import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APIFY_API_KEY = os.getenv("APIFY_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
APIFY_ACTOR = os.getenv("APIFY_ACTOR", "curious_coder/linkedin-jobs-scraper")
GROQ_MODEL = os.getenv("GROQ_MODEL", "allam-2-7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DB_PATH = (BASE_DIR / os.getenv("DB_PATH", "data/jobs.db")).resolve()

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TOP_K_DEFAULT = 8
