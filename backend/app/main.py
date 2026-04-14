# Stage 6: FastAPI entry point.
# Two-step request shape keeps the per-upload index in memory between the
# upload call (parse + chunk + embed) and the analyze call (retrieve +
# analyze), so a future chat endpoint can reuse the same session without
# re-running the upstream pipeline.

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path

import tiktoken
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import sessions
from .analyzer import analyze_all
from .chunker import chunk_pages
from .config import settings
from .embedder import EmbeddingIndex
from .parser import parse_pdf
from .retriever import COMPLIANCE_QUESTIONS, retrieve_all
from .schemas import (
    AnalysisMetadata,
    AnalysisResponse,
    QuestionRetrievalStats,
    StageTimings,
    UploadResponse,
)

logger = logging.getLogger(__name__)

# Matches the chunker/retriever tokenizer so reported context_tokens lines
# up with the budget math in retriever._consolidate.
_tokenizer = tiktoken.get_encoding("cl100k_base")


app = FastAPI(title="Contract Compliance Analyzer")

# Streamlit runs on a separate port (8501 by default); allow any origin in
# dev so local iteration doesn't need an nginx-style proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Parse, chunk, and embed an uploaded PDF; register a session."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    upload_id = uuid.uuid4().hex
    # NamedTemporaryFile with delete=False so we can close it before PyMuPDF
    # reopens by path, then explicitly unlink after parse (decision #5).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        t0 = time.perf_counter()
        pages = await parse_pdf(tmp_path)
        parse_sec = time.perf_counter() - t0
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError as e:
            logger.warning("Failed to remove temp PDF %s: %s", tmp_path, e)

    t0 = time.perf_counter()
    chunks = chunk_pages(pages)
    chunk_sec = time.perf_counter() - t0

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No content extracted from PDF (empty or unreadable).",
        )

    t0 = time.perf_counter()
    index = EmbeddingIndex(upload_id)
    index.build(chunks)
    embed_sec = time.perf_counter() - t0

    sessions.put(sessions.Session(
        upload_id=upload_id,
        filename=file.filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        index=index,
        parse_sec=parse_sec,
        chunk_sec=chunk_sec,
        embed_sec=embed_sec,
    ))

    logger.info(
        "upload[%s]: %s — %d pages, %d chunks (parse=%.2fs chunk=%.2fs embed=%.2fs)",
        upload_id, file.filename, len(pages), len(chunks),
        parse_sec, chunk_sec, embed_sec,
    )

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        parse_sec=parse_sec,
        chunk_sec=chunk_sec,
        embed_sec=embed_sec,
    )


@app.post("/api/analyze/{upload_id}", response_model=AnalysisResponse)
async def analyze(upload_id: str) -> AnalysisResponse:
    """Retrieve + grade the 5 compliance questions for a prior upload."""
    session = sessions.get(upload_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown upload_id '{upload_id}'. Upload the PDF first.",
        )

    t0 = time.perf_counter()
    retrieved = retrieve_all(session.index)
    retrieve_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = await analyze_all(retrieved)
    analyze_sec = time.perf_counter() - t0

    stats = [
        QuestionRetrievalStats(
            question_id=q.id,
            chunks_used=len(retrieved.get(q.id, [])),
            context_tokens=sum(
                len(_tokenizer.encode(c.text, disallowed_special=()))
                for c in retrieved.get(q.id, [])
            ),
            top_score=max(
                (c.score for c in retrieved.get(q.id, [])), default=None,
            ),
        )
        for q in COMPLIANCE_QUESTIONS
    ]

    total_sec = (
        session.parse_sec + session.chunk_sec + session.embed_sec
        + retrieve_sec + analyze_sec
    )

    metadata = AnalysisMetadata(
        upload_id=upload_id,
        filename=session.filename,
        page_count=session.page_count,
        chunk_count=session.chunk_count,
        timings=StageTimings(
            parse_sec=session.parse_sec,
            chunk_sec=session.chunk_sec,
            embed_sec=session.embed_sec,
            retrieve_sec=retrieve_sec,
            analyze_sec=analyze_sec,
            total_sec=total_sec,
        ),
        retrieval=stats,
        models={
            "analysis": settings.analysis_model,
            "embedding": settings.embedding_model,
            "vision": settings.vision_model,
        },
    )

    errors = sum(1 for r in results if r.error)
    logger.info(
        "analyze[%s]: done in %.2fs (retrieve=%.2fs analyze=%.2fs) — %d error(s)",
        upload_id, retrieve_sec + analyze_sec, retrieve_sec, analyze_sec, errors,
    )

    return AnalysisResponse(results=results, metadata=metadata)
