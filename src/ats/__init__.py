"""ATS-style Resume–Job Match Scoring Engine.

Architecture:
    Resume → LLM → Structured Evidence
    JD → LLM → Structured Requirements
    ↓
    Semantic Matcher → evidence classification
    ↓
    Deterministic Scoring Engine → 0–100
    ↓
    LLM Explanation → recommendations

The LLM extracts and classifies. Python computes the score.
"""
from src.ats.scoring_engine import score_resume_vs_jd

__all__ = ["score_resume_vs_jd"]
