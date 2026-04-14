# Stage 6: SQLite-backed per-upload session store.
#
# Metadata (filename, timings, page/chunk counts) lives in a small SQLite
# file on disk; the EmbeddingIndex is reconstructed on demand by binding
# to its already-persisted Qdrant collection. The full upload → analyze
# → chat flow survives uvicorn restarts as long as both the SQLite file
# and the Qdrant server are still around.
#
# The public surface (Session dataclass, get/put) matches the prior
# in-memory OrderedDict so main.py doesn't change.

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .embedder import EmbeddingIndex, _make_qdrant_client

logger = logging.getLogger(__name__)

# Hard cap on retained sessions. When exceeded, the oldest rows (by
# created_at) are evicted from SQLite AND their Qdrant collections are
# dropped so disk usage doesn't grow unboundedly.
MAX_SESSIONS = 20


@dataclass
class Session:
    upload_id: str
    filename: str
    page_count: int
    chunk_count: int
    index: EmbeddingIndex
    parse_sec: float
    chunk_sec: float
    embed_sec: float
    created_at: float = field(default_factory=time.time)


_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    Path(settings.session_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.session_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                upload_id   TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                page_count  INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                parse_sec   REAL NOT NULL,
                chunk_sec   REAL NOT NULL,
                embed_sec   REAL NOT NULL,
                created_at  REAL NOT NULL
            )
            """
        )


_init_db()


def put(session: Session) -> None:
    """Persist session metadata; evict oldest entries past MAX_SESSIONS."""
    evicted: list[str] = []
    with _lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (upload_id, filename, page_count, chunk_count,
                     parse_sec, chunk_sec, embed_sec, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(upload_id) DO UPDATE SET
                    filename    = excluded.filename,
                    page_count  = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    parse_sec   = excluded.parse_sec,
                    chunk_sec   = excluded.chunk_sec,
                    embed_sec   = excluded.embed_sec,
                    created_at  = excluded.created_at
                """,
                (
                    session.upload_id,
                    session.filename,
                    session.page_count,
                    session.chunk_count,
                    session.parse_sec,
                    session.chunk_sec,
                    session.embed_sec,
                    session.created_at,
                ),
            )

            old = conn.execute(
                """
                SELECT upload_id FROM sessions
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
                """,
                (MAX_SESSIONS,),
            ).fetchall()
            if old:
                evicted = [row["upload_id"] for row in old]
                placeholders = ",".join("?" * len(evicted))
                conn.execute(
                    f"DELETE FROM sessions WHERE upload_id IN ({placeholders})",
                    evicted,
                )

    # Drop Qdrant collections after the SQLite transaction committed, so
    # a crash mid-eviction can't strand metadata pointing at deleted
    # collections.
    for upload_id in evicted:
        _drop_qdrant_collection(upload_id)
        logger.info("session[%s]: evicted (LRU cap=%d)", upload_id, MAX_SESSIONS)


def get(upload_id: str) -> Session | None:
    """Fetch metadata and attach to the matching Qdrant collection."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE upload_id = ?", (upload_id,)
        ).fetchone()
    if row is None:
        return None

    index = EmbeddingIndex.attach(upload_id)
    if index is None:
        # SQLite says the upload exists but Qdrant doesn't have its
        # collection — drift from manual intervention, disk loss, etc.
        # Treat as gone; clean up the orphan metadata so we don't keep
        # serving 500s.
        logger.warning(
            "session[%s]: metadata present but Qdrant collection missing; "
            "deleting orphan row",
            upload_id,
        )
        with _lock, _connect() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE upload_id = ?", (upload_id,)
            )
        return None

    return Session(
        upload_id=row["upload_id"],
        filename=row["filename"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        index=index,
        parse_sec=row["parse_sec"],
        chunk_sec=row["chunk_sec"],
        embed_sec=row["embed_sec"],
        created_at=row["created_at"],
    )


def delete(upload_id: str) -> None:
    """Remove a session from SQLite and drop its Qdrant collection.

    Idempotent: missing rows / missing collections are silently ignored.
    """
    with _lock, _connect() as conn:
        conn.execute(
            "DELETE FROM sessions WHERE upload_id = ?", (upload_id,)
        )
    _drop_qdrant_collection(upload_id)
    logger.info("session[%s]: deleted", upload_id)


def _drop_qdrant_collection(upload_id: str) -> None:
    try:
        client = _make_qdrant_client()
        name = f"upload_{upload_id}"
        if client.collection_exists(name):
            client.delete_collection(name)
    except Exception:
        logger.exception(
            "failed to drop qdrant collection for %s", upload_id
        )
