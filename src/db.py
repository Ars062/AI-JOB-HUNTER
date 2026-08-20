import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

import numpy as np

from config import DB_PATH


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                company TEXT,
                location TEXT,
                description TEXT,
                url TEXT UNIQUE,
                source TEXT,
                salary TEXT,
                job_type TEXT,
                skills TEXT,
                posted_date TEXT,
                embedding BLOB,
                fetched_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
            CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
            CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )


def _serialize(emb: Optional[np.ndarray]) -> Optional[bytes]:
    if emb is None:
        return None
    return emb.astype(np.float32).tobytes()


def _deserialize(blob: Optional[bytes], dim: int) -> Optional[np.ndarray]:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).reshape(1, dim)


def insert_job(job: dict, embedding: Optional[np.ndarray] = None) -> bool:
    """Insert one normalized job. Returns True if newly added, False if duplicate."""
    with get_conn() as conn:
        try:
            conn.execute(
                """
                INSERT INTO jobs
                (title, company, location, description, url, source, salary,
                 job_type, skills, posted_date, embedding, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.get("title"),
                    job.get("company"),
                    job.get("location"),
                    job.get("description"),
                    job.get("url"),
                    job.get("source"),
                    job.get("salary"),
                    job.get("job_type"),
                    job.get("skills"),
                    job.get("posted_date"),
                    _serialize(embedding),
                    datetime.utcnow().isoformat(),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def upsert_embedding(job_id: int, embedding: np.ndarray) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET embedding = ? WHERE id = ?", (_serialize(embedding), job_id))


def get_all_jobs(with_embedding: bool = False) -> list[dict]:
    sql = "SELECT id, title, company, location, description, url, source, salary, job_type, skills, posted_date FROM jobs"
    if with_embedding:
        sql = "SELECT *, embedding FROM jobs"
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_all_embeddings() -> tuple[np.ndarray, list[dict]]:
    """Return (embedding_matrix, metadata) for all jobs that have embeddings."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, company, location, description, url, source, salary, job_type, skills, posted_date, embedding FROM jobs WHERE embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return np.zeros((0, 384), dtype=np.float32), []
    dim = len(rows[0]["embedding"]) // 4
    matrix = np.vstack([_deserialize(r["embedding"], dim) for r in rows]).astype(np.float32)
    meta = [dict(r) for r in rows]
    for m in meta:
        m.pop("embedding", None)
    return matrix, meta


def keyword_search(terms: list[str], filters: Optional[dict] = None, limit: int = 20) -> list[dict]:
    """Simple LIKE-based search over title/company/description."""
    with get_conn() as conn:
        where = []
        params: list[Any] = []
        if terms:
            for t in terms:
                where.append("(title LIKE ? OR company LIKE ? OR description LIKE ? OR skills LIKE ?)")
                like = f"%{t}%"
                params += [like, like, like, like]
        if filters:
            if filters.get("source"):
                where.append("source = ?")
                params.append(filters["source"])
            if filters.get("company"):
                where.append("company LIKE ?")
                params.append(f"%{filters['company']}%")
            if filters.get("location"):
                where.append("location LIKE ?")
                params.append(f"%{filters['location']}%")
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY posted_date DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_jobs() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


def stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        sources = conn.execute("SELECT source, COUNT(*) as n FROM jobs GROUP BY source").fetchall()
    return {"total": total, "by_source": {r["source"] or "unknown": r["n"] for r in sources}}


def add_chat_message(role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))


def get_chat_history(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_chat_history() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_history")


def clear_db() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM chat_history")


def summary() -> str:
    s = stats()
    src = ", ".join(f"{k}: {v}" for k, v in s["by_source"].items()) or "none"
    return f"{s['total']} jobs indexed ({src})"
