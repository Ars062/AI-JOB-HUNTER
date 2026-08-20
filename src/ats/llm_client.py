"""LLM client abstraction — wraps Groq for extraction tasks."""
import json
import logging
import time

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class LLMError(Exception):
    pass


def _get_client():
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set.")
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def complete(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Call the LLM with retry on rate limit."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            client = _get_client()
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
            # Strip thinking blocks from models that output them
            import re
            content = re.sub(r"<think>[\s\S]*?</think>", "", content)
            content = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", content)
            content = re.sub(r"<思考>[\s\S]*?</思考>", "", content)
            return content.strip()
        except Exception as e:
            last_error = e
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning("Rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
            else:
                break
    raise LLMError(str(last_error))


def complete_json(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 2048) -> dict:
    """Call the LLM and parse the response as JSON."""
    raw = complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    # Strip thinking/reasoning blocks from models that output them
    import re
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw)
    raw = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", raw)
    raw = re.sub(r"<思考>[\s\S]*?</思考>", "", raw)
    raw = re.sub(r"<inner_monologue>[\s\S]*?</inner_monologue>", "", raw)
    raw = raw.strip()
    # Try to extract JSON from the response
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        # Try array
        start = raw.find("[")
        end = raw.rfind("]")
    if start == -1 or end == -1:
        raise LLMError(f"No JSON found in LLM response: {raw[:300]}")
    def _to_dict(data):
        """Coerce a parsed JSON value into a dict (unwrap single-element arrays)."""
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item
            raise LLMError(f"Expected JSON object, got list: {raw[:300]}")
        raise LLMError(f"Expected JSON object, got {type(data).__name__}: {raw[:300]}")

    try:
        return _to_dict(json.loads(raw[start : end + 1]))
    except json.JSONDecodeError:
        # Try to fix common issues: trailing commas, missing quotes
        text = raw[start : end + 1]
        # Remove trailing commas before } or ]
        import re
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return _to_dict(json.loads(text))
        except json.JSONDecodeError:
            raise LLMError(f"Invalid JSON in LLM response: {raw[:300]}")
