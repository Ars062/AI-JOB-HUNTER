"""Groq LLM client with graceful fallback when no key is set."""

import json
import logging
import re
import time

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 4


class LLMError(Exception):
    pass


def _client():
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)


def complete(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            client = _client()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            content = re.sub(r" thinking[\s\S]*? response", "", content)
            content = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", content)
            return content.strip()
        except Exception as e:  # noqa: BLE001
            last_error = e
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
            else:
                break
    raise LLMError(str(last_error))


def extract_search_params(question: str) -> dict:
    """Ask Groq to turn a natural-language question into structured search filters.

    Returns a dict; on any failure returns an empty dict (caller falls back to
    plain keyword/vector search).
    """
    system = (
        "You extract job-search filters from a user question. "
        'Respond with ONLY a JSON object using keys: "keywords" (list of strings), '
        '"location" (string), "company" (string), "source" (string, e.g. linkedin/indeed, or ""). '
        'Example: {"keywords": ["computer vision", "football"], "location": "Denmark", "company": "", "source": ""}'
    )
    try:
        raw = complete(question, system=system, temperature=0.0, max_tokens=256)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(raw[start : end + 1])
    except Exception as e:  # noqa: BLE001
        logger.warning("Param extraction failed: %s", e)
        return {}


def answer_question(
    question: str,
    job_context: str,
    history: list[dict] | None = None,
    resume_context: str | None = None,
) -> str:
    system = (
        "You are a personal job-search assistant. You are given a list of REAL jobs from the user's "
        "search, each with an official ATS match score (0-100) that was computed by a scoring engine. "
        "This score is authoritative and appears exactly in the job list as 'ATS <N>/100'. "
        "STRICT RULES:\n"
        "- NEVER invent or guess a job, company, location, or ATS score. Use ONLY what is in the job list.\n"
        "- NEVER claim 'no matching jobs' unless the job list is empty (shown as '(no matching jobs found)').\n"
        "- If jobs exist, recommend the 3-5 with the HIGHEST ATS scores and quote their exact scores.\n"
        "- Base every sentence strictly on the job list. Be concise (3-6 sentences).\n"
        'End your answer with the line "Results shown below."'
    )
    if resume_context:
        system += (
            f"\nThe user uploaded their resume. Their profile highlights: {resume_context}."
        )
    history_text = ""
    if history:
        lines = [f"{h['role']}: {h['content']}" for h in history[-4:]]
        history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        f"{history_text}Job list (ATS scores are exact, use them):\n{job_context or '(no matching jobs found)'}\n\n"
        f"User question: {question}\n"
        "Answer:"
    )
    try:
        return complete(prompt, system=system, temperature=0.0, max_tokens=1024)
    except LLMError:
        return _fallback_answer(question, job_context)


def _fallback_answer(question: str, job_context: str) -> str:
    if not job_context:
        return "No matching jobs found. Try a different query or location."
    n = job_context.count("- ") if "- " in job_context else 0
    return f"Here are the top {max(n, 1)} job matches I found for you. See details below."
