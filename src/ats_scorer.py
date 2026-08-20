"""ATS keyword-matching scorer — fully local, no API needed.

Uses TF-IDF + a curated skills database to extract important keywords from
a job description, then checks which ones appear in the resume. Returns an
ATS score (0-100) with matched/missing keyword breakdowns.
"""

import re
import string

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ── curated skills database (what ATS systems actually look for) ────────────

SKILLS_DB = {
    # programming languages
    "python", "c++", "java", "javascript", "typescript", "go", "golang",
    "rust", "matlab", "r", "scala", "kotlin", "swift", "ruby", "php", "perl",
    "sql", "html", "css",
    # ml / ai
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "opencv",
    "hugging face", "huggingface", "xgboost", "lightgbm", "catboost",
    "machine learning", "deep learning", "computer vision", "nlp",
    "natural language processing", "llm", "large language model",
    "reinforcement learning", "neural network", "transformer", "diffusion",
    "object detection", "image segmentation", "pose estimation",
    "speech recognition", "generative ai", "gen ai",
    # data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "spark", "hadoop", "hive", "kafka", "airflow",
    "data science", "data engineering", "data analysis", "etl",
    "tableau", "power bi", "looker",
    # databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "dynamodb", "sqlite", "neo4j",
    # cloud / infra
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "helm",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "ci/cd", "cicd", "linux", "bash", "shell",
    "git", "github", "gitlab", "bitbucket",
    # web / backend
    "react", "vue", "angular", "next.js", "nextjs", "node.js", "nodejs",
    "express", "fastapi", "flask", "django", "spring", "graphql", "rest",
    "rest api", "grpc", "microservice", "microservices",
    # mobile
    "android", "ios", "flutter", "react native", "swift",
    # devops / mlops
    "mlops", "mlflow", "weights & biases", "wandb", "dvc",
    "model deployment", "model monitoring", "feature store",
    # domain specific
    "ros", "ros2", "slam", "lidar", "camera calibration", "sensor fusion",
    "embedded", "embedded systems", "rtos", "firmware",
    "edge ai", "edge computing", "streaming", "video processing",
    "football", "sports analytics", "player tracking", "xg", "expected goals",
    # backend / web (extra)
    "node", "jest", "cypress", "playwright", "selenium", "postman", "jquery",
    "bootstrap", "tailwind", "sass", "redux", "next.js", "spring boot",
    "laravel", "ruby on rails", "asp.net", ".net", "dotnet", "gin", "echo",
    "rabbitmq", "celery", "graphql", "gin", "dart", "expo", "xamarin",
    "pytest", "mocha", "kafka", "elasticsearch", "logstash", "kibana",
    # data engineering / analytics
    "snowflake", "bigquery", "redshift", "databricks", "dbt", "polars",
    "duckdb", "clickhouse", "presto", "trino", "glue", "emr", "etl piping",
    "maria db", "mariadb", "sql server", "oracle", "db2", "timescale",
    # mlops / deployment
    "langchain", "llamaindex", "ollama", "transformers", "embedding",
    "embeddings", "fine-tuning", "finetuning", "rag", "vector database",
    "semantic search", "whisper", "asr", "text to speech", "text-to-speech",
    "speech to text", "speech-to-text", "onnx", "openvino", "tflite", "torchserve",
    "vllm", "generative ai", "diffusion", "stable diffusion", "prompt engineering",
    "mlflow", "dvc", "kubeflow", "ray", "bentoml", "sagemaker", "vertex ai",
    # cloud / devops (extra)
    "ecs", "fargate", "lambda", "serverless", "cloudformation", "circleci",
    "gitlab ci", "travis", "teamcity", "podman", "vagrant", "prometheus",
    "grafana", "datadog", "sentry", "splunk", "cloudwatch", "azure devops",
    "argo", "flux", "istio", "openshift", "nginx", "apache", "active directory",
    # security / network
    "cybersecurity", "penetration testing", "owasp", "siem", "firewall",
    "encryption", "cryptography", "ccna", "networking", "tcp/ip", "cisco",
    "information security", "vulnerability", "threat detection", "incident response",
    # programming languages (extra)
    "c#", "c#.net", "vb.net", "objective-c", "delphi", "pascal", "fortran",
    "cobol", "assembly", "haskell", "julia", "racket", "elixir", "erlang",
    "solidity", "powershell", "batch", "pl/sql", "t-sql", "vba",
    # testing / qa
    "unit testing", "integration testing", "test automation", "ci pipeline",
    "quality assurance", "qa testing", "regression testing", "load testing",
    "smoke testing", "test case", "test plan", "sonarqube",
    # design / creative
    "figma", "adobe xd", "sketch", "illustrator", "photoshop", "indesign",
    "after effects", "premiere pro", "canva", "ux design", "ui design",
    "wireframe", "prototyping", "user research", "a/b testing", "design system",
    "motion graphics", "3d modeling", "blender", "maya", "unity", "unreal engine",
    "game design", "game development", "gan", "stable diffusion",
    # hardware / embedded / cad
    "arduino", "raspberry pi", "fpga", "verilog", "vhdl", "pcb design",
    "altium", "autocad", "solidworks", "catia", "simulink", "plc", "scada",
    "power electronics", "circuit design", "cnc", "hardware design", "ethernet",
    # robotics / automotive
    "gazebo", "moveit", "automotive", "electric vehicles", "ev battery",
    "powertrain", "adas", "autonomous driving", "object detection", "yolo",
    # business / process
    "business analysis", "business intelligence", "product management",
    "product owner", "roadmap", "stakeholder management", "lean", "six sigma",
    "process improvement", "operations management", "supply chain", "procurement",
    "logistics", "warehouse", "inventory management", "inventory control",
    "sap", "erp", "crm", "salesforce", "hubspot", "zendesk",
    # marketing / sales
    "seo", "sem", "google analytics", "ga4", "content marketing", "email marketing",
    "social media marketing", "social media", "lead generation", "copywriting",
    "cold calling", "negotiation", "b2b sales", "b2c", "direct sales", "key account",
    "account management", "digital marketing", "marketing automation",
    "facebook ads", "google ads", "adwords", "inbound marketing", "branding",
    "market research", "competitor analysis", "customer acquisition",
    # finance / accounting
    "accounting", "bookkeeping", "financial reporting", "budgeting", "forecasting",
    "payroll", "tax", "gst", "tds", "quickbooks", "xero", "tally", "sap fico",
    "financial analysis", "auditing", "compliance", "financial modeling",
    "risk management", "investment", "equity research", "fp&a", "reconciliation",
    # hr / recruiting
    "recruiting", "talent acquisition", "hiring", "onboarding", "offboarding",
    "employee relations", "performance management", "hr operations", "payroll processing",
    "interviewing", "employee engagement", "training and development",
    "workday", "bamboo", "linkedin recruiter", "job postings", "candidate sourcing",
    # education / teaching
    "teaching", "education", "curriculum", "lesson planning", "e-learning", "lms",
    "instructional design", "tutoring", "assessment", "classroom management",
    "lesson plans", "k-12", "higher education", "online teaching", "vocational training",
    # healthcare / medical
    "nursing", "patient care", "clinical", "medical", "phlebotomy", "pharmacology",
    "cardiology", "emergency", "radiology", "laboratory", "icu", "hospital",
    "medical coding", "billing", "healthcare", "aged care", "pharmacy", "dental",
    "physiotherapy", "occupational therapy", "midwifery", "first aid", "cpr",
    "patient safety", "care planning", "wound care", "medication administration",
    # hospitality / food / retail
    "hospitality", "restaurant", "hotel", "front desk", "concierge", "reservations",
    "housekeeping", "catering", "banquet", "event planning", "bartending", "barista",
    "culinary", "cooking", "baking", "food safety", "food handler", "haccp",
    "retail", "sales associate", "cashier", "merchandising", "visual merchandising",
    "store management", "shift management", "customer experience", "customer service",
    "technical support", "helpdesk", "ticketing", "call center", "client relationship",
    "customer support", "customer facing", "client facing",
    # legal / compliance
    "legal", "contract drafting", "contract review", "intellectual property",
    "litigation", "paralegal", "regulatory", "data privacy", "gdpr", "iso 9001",
    "iso 27001", "internal audit", "governance", "policy", "risk assessment",
    # real estate / property / construction
    "real estate", "property management", "leasing", "brokerage", "appraisal",
    "tenant relations", "construction", "site management", "project engineer",
    "civil engineering", "mechanical engineering", "quality control", "hse",
    "hse safety", "occupational health", "environmental", "surveying", "estimation",
    # office / admin / operations
    "administration", "administrative", "office management", "executive assistant",
    "reception", "data entry", "documentation", "reporting", "scheduling",
    "coordination", "project coordination", "time management", "ms office",
    "excel", "powerpoint", "word", "outlook", "google sheets", "google workspace",
    # creative writing / communication
    "content writing", "technical writing", "creative writing", "copy editing",
    "proofreading", "translation", "transcribing", "public speaking", "presentation",
    "communication", "story telling", "storytelling", "journalism", "editing",
    # sports / fitness
    "coaching", "fitness", "personal training", "gym", "nutrition", "athletic",
    "recreation", "soccer", "cricket", "basketball", "swimming", "yoga",
    "strength training", "conditioning", "referee", "sports coaching",
    # soft / process
    "agile", "scrum", "kanban", "jira", "confluence",
    "team lead", "technical lead", "project management", "mentoring",
    "code review", "unit test", "integration test", "tdd", "bdd",
}

# Normalisation map: variant -> canonical form
_ALIASES = {
    "sklearn": "scikit-learn",
    "k8s": "kubernetes",
    "cicd": "ci/cd",
    "nextjs": "next.js",
    "nodejs": "node.js",
    "huggingface": "hugging face",
    "wandb": "weights & biases",
    "ros2": "ros2",
    "golang": "go",
}

_STOP = frozenset(
    "a an the and or but in on at for of to with is are was were be been "
    "being have has had do does did will would shall should may might can "
    "could this that these those it its we our you your he she they them "
    "their his her i me my am not no nor so if then than too very just "
    "also about above after again all any as back because before between "
    "both by come each even from get got how into more most much must "
    "need new now only other over own same some still such take there "
    "under up us what when where which while who whom why looking need "
    "please company companies jobs job like best top show find list get".split()
)


# ── pre-processing ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase, remove extra whitespace, normalise punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _normalise(term: str) -> str:
    """Map variant to canonical skill name."""
    t = term.lower().strip()
    return _ALIASES.get(t, t)


# ── keyword extraction ──────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 25) -> list[tuple[str, float]]:
    """Extract the top-n most important keywords from a JD.

    Strategy:
    1. Find all curated skills mentioned in the JD (exact match, high weight)
    2. Use TF-IDF to find additional domain terms not in the skills DB
    3. Merge and return sorted by weight

    Returns [(keyword, weight), ...] sorted by weight descending.
    """
    cleaned = _clean(text)

    # ── Step 1: find curated skills in the JD ──
    found_skills: list[tuple[str, float]] = []
    for skill in SKILLS_DB:
        # word boundary match for single words, substring for phrases
        pattern = r"\b" + re.escape(skill) + r"\b" if " " not in skill else re.escape(skill)
        if re.search(pattern, cleaned):
            # Weight based on skill importance (shorter = more common = slightly lower)
            weight = 0.9 if len(skill) <= 3 else 0.85
            found_skills.append((skill, weight))

    # ── Step 2: TF-IDF for additional terms (multi-word phrases or real skills) ──
    corpus = [cleaned]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(_STOP),
        max_features=300,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(corpus)
    jd_scores = tfidf[0].toarray().flatten()
    feature_names = vec.get_feature_names_out()

    # Skill set (normalised) so single non-skill words never get scored
    _skills_norm = {_normalise(s) for s in SKILLS_DB}
    tfidf_terms: list[tuple[str, float]] = []
    curated_set = {s for s, _ in found_skills}
    # Generic terms that TF-IDF extracts but aren't useful as ATS keywords
    _generic = {
        "engineer", "experience", "software", "systems", "developer",
        "senior", "junior", "team", "role", "position", "requirements",
        "preferred", "required", "including", "ability", "working",
        "knowledge", "years", "strong", "good", "great", "looking",
        "work", "works", "job", "training", "trainer", "international",
        "comfortable", "background", "rotational",
    }
    for name, score in zip(feature_names.tolist(), jd_scores.tolist()):
        name_clean = _normalise(name)
        words = name_clean.split()
        is_real_skill = name_clean in _skills_norm
        if (
            (len(words) > 1 or is_real_skill)
            and len(name_clean) > 2
            and score > 0.15
            and name_clean not in _STOP
            and name_clean not in _generic
            and not any(c.isdigit() for c in name_clean)
            and name_clean not in curated_set
        ):
            tfidf_terms.append((name_clean, min(score, 0.5)))

    tfidf_terms.sort(key=lambda x: x[1], reverse=True)

    # ── Step 3: merge ──
    merged: list[tuple[str, float]] = list(found_skills)
    seen = {s for s, _ in merged}

    for term, weight in tfidf_terms:
        if term not in seen:
            merged.append((term, weight))
            seen.add(term)

    # Sort by weight, take top_n
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_n]


# ── matching ────────────────────────────────────────────────────────────────

def _text_contains(text_clean: str, keyword: str) -> bool:
    """Check if a keyword appears in cleaned text (word boundary aware)."""
    kw = keyword.lower().strip()
    if " " not in kw:
        return bool(re.search(r"\b" + re.escape(kw) + r"\b", text_clean))
    return kw in text_clean


def score_ats(
    resume_text: str,
    jd_text: str,
    top_n: int = 25,
    resume_skills: list[str] | None = None,
) -> dict:
    """Score a resume against a job description using ATS keyword matching.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.
        top_n: Number of top keywords to extract from the JD.
        resume_skills: Optional pre-extracted skill list to also check against.

    Returns:
        {
            "score": 72,                          # 0-100
            "matched_keywords": ["python", ...],  # keywords found in resume
            "missing_keywords": ["terraform", ...],# keywords NOT found
            "keyword_details": [                   # full breakdown
                {"keyword": "python", "weight": 0.9, "found": True},
                ...
            ]
        }
    """
    if not resume_text.strip() or not jd_text.strip():
        return {
            "score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "keyword_details": [],
        }

    # Extract important keywords from JD
    jd_keywords = extract_keywords(jd_text, top_n=top_n)

    # Clean resume text for matching
    resume_clean = _clean(resume_text)
    resume_skills_norm = [_normalise(s) for s in (resume_skills or [])]
    # If no explicit skill list, derive skills straight from the resume text
    if not resume_skills_norm:
        for skill in SKILLS_DB:
            pattern = r"\b" + re.escape(skill) + r"\b" if " " not in skill else re.escape(skill)
            if re.search(pattern, resume_clean):
                resume_skills_norm.append(_normalise(skill))

    # ── Relevance gate: does the job actually relate to the resume's skills? ──
    # If the resume has clearly identified skills, require at least one of them
    # to appear in the JD. Otherwise unrelated jobs (sales, admin, etc.) get
    # absurdly high scores from generic keywords.
    if resume_skills_norm and len(resume_skills_norm) >= 3:
        jd_clean = _clean(jd_text)
        skill_hits = [s for s in resume_skills_norm if s in jd_clean]
        if not skill_hits:
            return {
                "score": 0,
                "matched_keywords": [],
                "missing_keywords": [kw for kw, _ in jd_keywords[:15]],
                "keyword_details": [
                    {"keyword": kw, "weight": round(w, 3), "found": False}
                    for kw, w in jd_keywords[:15]
                ],
            }

    # Score each keyword
    matched = []
    missing = []
    details = []

    for kw, weight in jd_keywords:
        found = _text_contains(resume_clean, kw)
        # Also check against pre-extracted skills (normalised)
        if not found:
            found = any(kw in s or s in kw for s in resume_skills_norm)

        entry = {"keyword": kw, "weight": round(weight, 3), "found": found}
        details.append(entry)

        if found:
            matched.append(kw)
        else:
            missing.append(kw)

    # Weighted score calculation
    if not jd_keywords:
        return {
            "score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "keyword_details": [],
        }

    total_weight = sum(w for _, w in jd_keywords)
    matched_weight = sum(w for kw, w in jd_keywords if kw in matched)
    raw = (matched_weight / total_weight) * 100 if total_weight > 0 else 0

    # Guard against spurious 100s: a job that only yields 1-2 keywords is
    # likely a generic/irrelevant posting — cap score by evaluation depth.
    n_kw = len(jd_keywords)
    if n_kw <= 1:
        raw = min(raw, 50)
    elif n_kw == 2:
        raw = min(raw, 65)
    elif n_kw <= 4:
        raw = min(raw, 80)

    score = int(round(raw))

    return {
        "score": min(score, 100),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_details": details,
    }


def score_job(resume_text: str, job: dict, resume_skills: list[str] | None = None) -> dict:
    """Score a resume against a job dict (from DB). Returns ATS score dict + job meta."""
    jd_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('skills', '')}"
    result = score_ats(resume_text, jd_text, resume_skills=resume_skills)
    result["job_id"] = job.get("id")
    result["job_title"] = job.get("title")
    result["company"] = job.get("company")
    return result


def score_all_jobs(
    resume_text: str,
    jobs: list[dict],
    resume_skills: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Score a resume against all jobs. Returns top_k results sorted by score."""
    results = [score_job(resume_text, job, resume_skills) for job in jobs]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
