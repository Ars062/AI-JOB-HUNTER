import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import APIFY_API_KEY, GROQ_API_KEY
from src.agent import ask
from src.ats.scoring_engine import score_resume_vs_jd
from src.db import clear_chat_history, init_db, stats
from src.embeddings import embed_text
from src.resume_matcher import extract_skills, parse_resume

st.set_page_config(page_title="Job Search AI", page_icon="🎯", layout="wide")
init_db()


@st.cache_resource
def _ensure_embeddings():
    return embed_text(["embedding warmup"])


def _ats_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    return "🔴"


def _ats_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Moderate"
    if score >= 50:
        return "Weak"
    return "Poor"


def _save_uploaded_cv(upload) -> str | None:
    suffix = ".pdf" if upload.name.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getvalue())
        return tmp.name


def _build_jd_text(job: dict) -> str:
    return f"{job.get('title', '')} {job.get('description', '')} {job.get('skills', '')}"


def _render_full_analysis(result):
    """Render the full ATS breakdown for a job."""
    score = result.overall_score
    color = _ats_color(score)
    label = _ats_label(score)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label="Overall Match", value=f"{score}/100", delta=f"{color} {label}")

    if result.hard_blockers:
        st.error("⚠️ **Hard Blockers**")
        for b in result.hard_blockers:
            st.markdown(f"- **{b.requirement}**: {b.explanation}")

    st.subheader("📊 Category Breakdown")
    cat_rows = []
    for key, cs in result.category_scores.items():
        cat_rows.append({
            "Category": key.replace("_", " ").title(),
            "Score": f"{cs.score:.0f}/{cs.max_points:.0f}",
            "%": f"{cs.percentage}%",
            "Details": cs.details,
        })
    st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)

    if result.strong_matches:
        st.subheader(f"✅ Strong Matches ({len(result.strong_matches)})")
        for m in result.strong_matches:
            ev = f" — *\"{m.evidence}\"*" if m.evidence else ""
            st.markdown(f"- **{m.skill_name}** ({m.match_level.value}){ev}")

    if result.partial_matches:
        st.subheader(f"⚠️ Partial Matches ({len(result.partial_matches)})")
        for m in result.partial_matches:
            ev = f" — *\"{m.evidence}\"*" if m.evidence else ""
            st.markdown(f"- **{m.skill_name}** ({m.match_level.value}){ev}")

    if result.missing_requirements:
        st.subheader(f"❌ Missing Requirements ({len(result.missing_requirements)})")
        for m in result.missing_requirements:
            st.markdown(f"- **{m.skill_name}** — {m.explanation}")

    if result.recommendations:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(result.recommendations, 1):
            st.markdown(f"{i}. {rec}")

    st.caption(f"ℹ️ {result.disclaimer}")


def sidebar():
    with st.sidebar:
        st.title("🎯 Job Search AI")

        cv = st.file_uploader("📄 Upload CV", type=["pdf", "txt"], key="cv_uploader")
        if cv is not None:
            key = f"{cv.name}:{cv.size}"
            if st.session_state.get("cv_key") != key:
                tmp_path = _save_uploaded_cv(cv)
                text = parse_resume(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)
                if len(text) < 50:
                    st.warning("Could not read this CV.")
                else:
                    st.session_state["cv_key"] = key
                    st.session_state["cv_text"] = text
                    st.session_state["cv_skills"] = extract_skills(text)
            if st.session_state.get("cv_text"):
                skills = st.session_state.get("cv_skills") or []
                st.success(f"✅ CV active — {len(skills)} skills")
        else:
            st.session_state.pop("cv_key", None)
            st.session_state.pop("cv_text", None)
            st.info("Upload CV for ATS scoring.")

        st.markdown("---")
        st.caption("Filters (all optional)")
        q = st.text_input("Query", placeholder="e.g. computer vision")
        location = st.text_input("Location", placeholder="e.g. India, Copenhagen")
        job_type = st.multiselect("Job type", ["Remote", "Hybrid", "Onsite"])
        freshness = st.radio(
            "Posted within",
            ["Anytime", "24 hours", "7 days", "30 days"],
            index=0,
            horizontal=True,
        )
        max_items = st.slider("Max results", 10, 100, 20, 5)

        if st.session_state.get("cv_text") and st.session_state.get("cv_skills"):
            skills = st.session_state["cv_skills"]
            st.markdown(
                f"<div style='font-size:12px;color:#888;'>Your CV skills detected:\n"
                f"<b>{', '.join(skills[:8])}</b>" + ("…" if len(skills) > 8 else "") + "</div>",
                unsafe_allow_html=True,
            )
            st.caption("💬 Type in chat to find matching jobs")

        s = stats()
        st.caption(f"📊 {s['total']} jobs indexed")

    return q, location, job_type, freshness, max_items


# ── main ────────────────────────────────────────────────────────────────────


def render_job_cards(jobs: list[dict]):
    """Render job result cards with links, ATS score, and Full Analysis button."""
    cv_text = st.session_state.get("cv_text")
    for idx, job in enumerate(jobs):
        ats = job.get("ats_score")
        score_txt = f" — ATS **{ats}/100** {_ats_color(ats)}" if isinstance(ats, int) else ""

        st.markdown(f"### [{job.get('title') or 'Untitled'}]({job.get('url') or '#'}){score_txt}")
        meta = " | ".join(
            x for x in [
                job.get("company") or "",
                job.get("location") or "",
                job.get("job_type") or "",
                job.get("posted_date") or "",
            ] if x
        )
        if meta:
            st.markdown(f"*{meta}*")
        if job.get("salary"):
            st.markdown(f"💰 {job['salary']}")
        desc = (job.get("description") or "")[:400]
        if desc:
            st.markdown(desc)
        if job.get("skills"):
            st.markdown(f"**Skills:** {job['skills'][:200]}")

        matched = job.get("matched_keywords", [])
        missing = job.get("missing_keywords", [])
        if matched:
            st.markdown(f"**✅ Matched:** {', '.join(matched[:10])}")
        if missing:
            st.markdown(f"**❌ Missing:** {', '.join(missing[:10])}")

        if cv_text and job.get("description"):
            open_idx = st.session_state.get("full_analysis_job_idx")
            is_open = open_idx == idx
            if is_open:
                if st.button("✖ Close Analysis", key=f"fa_close_{idx}"):
                    st.session_state.full_analysis_job_idx = None
                    st.session_state.full_analysis_result = None
                    st.rerun()
                result = st.session_state.get("full_analysis_result")
                if result is None:
                    jd_text = _build_jd_text(job)
                    if jd_text.strip():
                        with st.spinner("Running full ATS analysis..."):
                            try:
                                result = score_resume_vs_jd(cv_text, jd_text)
                            except Exception as e:
                                st.error(f"Analysis failed: {e}")
                                result = None
                        st.session_state.full_analysis_result = result
                if result is not None:
                    st.markdown("---")
                    _render_full_analysis(result)
            else:
                if st.button("📋 Full Analysis", key=f"fa_{idx}"):
                    st.session_state.full_analysis_job_idx = idx
                    st.session_state.full_analysis_result = None
                    st.rerun()

        st.markdown("---")


def main():
    q, location, job_type, freshness, max_items = sidebar()
    cv_text = st.session_state.get("cv_text")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_jobs" not in st.session_state:
        st.session_state.last_jobs = []
    if "full_analysis_job_idx" not in st.session_state:
        st.session_state.full_analysis_job_idx = None
    if "full_analysis_result" not in st.session_state:
        st.session_state.full_analysis_result = None

    # Show chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Show job results from previous runs
    render_job_cards(st.session_state.last_jobs)

    # Chat input
    if prompt := st.chat_input("Ask anything, or paste a JD to score..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        prompt_lower = prompt.lower()
        is_jd_request = (
            cv_text
            and len(prompt) > 200
            and any(w in prompt_lower for w in ["score", "ats", "match", "compare", "fit", "rate"])
        )

        if is_jd_request:
            with st.chat_message("assistant"):
                with st.spinner("Scoring..."):
                    try:
                        result = score_resume_vs_jd(cv_text, prompt)
                    except Exception as e:
                        st.error(f"Scoring failed: {e}")
                        return
                score = result.overall_score
                st.metric("ATS Score", f"{score}/100", delta=f"{_ats_color(score)} {_ats_label(score)}")
                if result.strong_matches:
                    st.markdown(f"**✅ Strong:** {', '.join(m.skill_name for m in result.strong_matches[:8])}")
                if result.missing_requirements:
                    st.markdown(f"**❌ Missing:** {', '.join(m.skill_name for m in result.missing_requirements[:5])}")
                if result.recommendations:
                    with st.expander("💡 Recommendations"):
                        for i, rec in enumerate(result.recommendations, 1):
                            st.markdown(f"{i}. {rec}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"ATS Score: {score}/100 ({_ats_label(score)})",
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching matching jobs..."):
                    try:
                        cv_skills = st.session_state.get("cv_skills")
                        result = ask(
                            prompt,
                            resume_text=cv_text,
                            resume_skills=", ".join(cv_skills[:12]) if cv_skills else None,
                            ui_filters={
                                "query": q,
                                "location": location,
                                "job_type": job_type,
                                "freshness": freshness,
                                "max_items": max_items,
                            },
                        )
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        return
                st.write(result["answer"])

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            st.session_state.last_jobs = result["jobs"]
            render_job_cards(result["jobs"])

    if st.button("🗑 Clear"):
        st.session_state.messages = []
        st.session_state.last_jobs = []
        st.session_state.full_analysis_job_idx = None
        st.session_state.full_analysis_result = None
        clear_chat_history()
        st.rerun()


if __name__ == "__main__":
    main()
