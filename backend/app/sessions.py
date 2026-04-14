# Stage 6: Per-upload session store.
# Keeps an EmbeddingIndex alive between the upload call (parse + chunk +
# embed) and the analyze call (retrieve + analyze). LRU-capped so long-
# running processes don't grow unboundedly.

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .embedder import EmbeddingIndex

# Hard cap on concurrent uploads retained in-memory. Oldest entries are
# evicted when the cap is exceeded. A take-home-scale default; bump via
# settings if needed.
MAX_SESSIONS = 20


@dataclass
class Session:
    upload_id: str
    filename: str
    page_count: int
    chunk_count: int
    index: EmbeddingIndex
    # Timings captured at upload time so Stage 6's AnalysisResponse can
    # report end-to-end numbers without re-running parse/chunk/embed.
    parse_sec: float
    chunk_sec: float
    embed_sec: float
    created_at: float = field(default_factory=time.time)


_lock = threading.Lock()
_sessions: "OrderedDict[str, Session]" = OrderedDict()


def put(session: Session) -> None:
    """Register a session; evict the oldest entries if over capacity."""
    with _lock:
        _sessions[session.upload_id] = session
        _sessions.move_to_end(session.upload_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def get(upload_id: str) -> Session | None:
    """Look up a session by upload_id; refresh its LRU position on hit."""
    with _lock:
        session = _sessions.get(upload_id)
        if session is not None:
            _sessions.move_to_end(upload_id)
        return session
