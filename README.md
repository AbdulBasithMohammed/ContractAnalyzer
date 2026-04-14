# Contract Compliance Analyzer

End-to-end system that ingests a PDF contract, retrieves evidence with RAG, and answers five fixed security-compliance questions with a state (Fully / Partially / Non-Compliant), confidence score, supporting quotes, and rationale. Also ships a document chat.

Pipeline: **PDF → Parse → Chunk → Embed → Retrieve → Analyze (LLM) → Validate → Response.**

## Highlights

- **PyMuPDF-only parse** — text, tables (`page.find_tables()`), images in one library. Scanned pages route to Claude Haiku vision for OCR; significant images get vision descriptions. See [backend/app/parser.py](backend/app/parser.py).
- **Hybrid chunking** — tables atomic, prose section-split with a 3-level oversized fallback (paragraph → sentence → token-slice) and token overlap. See [backend/app/chunker.py](backend/app/chunker.py).
- **Voyage `voyage-law-2` embeddings** — 1024-dim, 16k context, legal-domain-tuned; stored in per-upload Qdrant collections. See [backend/app/embedder.py](backend/app/embedder.py).
- **Multi-query retrieval** — 5 questions × 5 hand-written sub-queries, all 25 embedded in a single Voyage call, union-by-max-score, rank-then-doc-order selection under an 8k-token budget. See [backend/app/retriever.py](backend/app/retriever.py).
- **Forced `tool_use` analysis** — Claude Sonnet returns the 4-field verdict via tool schema, parallelised per question with `asyncio.gather(return_exceptions=True)`. See [backend/app/analyzer.py](backend/app/analyzer.py).
- **Two-step API + persistent sessions** — `/api/upload` parses+embeds, `/api/analyze/{id}` scores, `/api/chat/{id}` streams NDJSON. SQLite metadata + Qdrant collection survives uvicorn restarts. See [backend/app/main.py](backend/app/main.py) and [backend/app/sessions.py](backend/app/sessions.py).
- **Streamlit frontend** — scorecard, per-question tabs with inline quote highlighting, cache rehydration, streaming chat. See [frontend/app.py](frontend/app.py).

Full write-up of how and *why* each stage looks the way it does: [details.md](details.md). Product requirements and trade-off notes: [PRD.md](PRD.md).

## Setup

```bash
cp backend/.env.example backend/.env   # fill in ANTHROPIC_API_KEY + VOYAGE_API_KEY
pip install -r backend/requirements.txt
docker compose up -d                    # starts Qdrant on :6333 (persistent vectors)
```

Required keys:

- `ANTHROPIC_API_KEY` — Claude (vision + analysis + chat)
- `VOYAGE_API_KEY` — Voyage AI (embeddings)

Remaining defaults in [backend/app/config.py](backend/app/config.py) are tuned for the Voyage free tier (3 RPM / 10K TPM). On a paid tier, drop `EMBEDDING_BATCH_SLEEP_SEC` to `0` and raise `EMBEDDING_BATCH_SIZE` to `128`.

## Run

```bash
# backend
uvicorn backend.app.main:app --reload --port 8000

# frontend (separate terminal)
streamlit run frontend/app.py
```

Frontend at `http://localhost:8501`, API at `http://localhost:8000` (`/docs` for Swagger).

## Smoke tests

```bash
cd backend && python test_pipeline.py   # parse → chunk → embed → retrieve per question
cd backend && python test_api.py        # upload → analyze via FastAPI TestClient
```

## Project layout

```
backend/
  app/
    parser.py       # stage 1: PDF → PageContent (+ vision OCR)
    chunker.py      # stage 2: PageContent → Chunk
    embedder.py     # stage 3: Chunk → Qdrant (voyage-law-2)
    retriever.py    # stage 4: question → RetrievedChunk[]
    analyzer.py     # stage 5: Claude Sonnet forced tool_use
    prompts.py      # system prompts + tool schema
    main.py         # stage 6: FastAPI (upload / analyze / chat)
    sessions.py     # SQLite + Qdrant-backed session store
    schemas.py      # pydantic contracts shared across stages
    config.py       # pydantic-settings from backend/.env
  test_pipeline.py  # in-process pipeline smoke test
  test_api.py       # in-process API smoke test
frontend/
  app.py            # Streamlit UI
docker-compose.yml  # Qdrant v1.12.4
details.md          # deep technical walkthrough
PRD.md              # product requirements + design rationale
CLAUDE.md           # guidance for Claude Code
```

## API surface

- `POST /api/upload` — multipart PDF → `{ upload_id, page_count, chunk_count, stage_timings }`.
- `POST /api/analyze/{upload_id}` → 5 `ComplianceResult`s + metadata.
- `POST /api/chat/{upload_id}` → NDJSON stream (`token` / `sources` / `done` / `error` events).
- `GET /api/health` → liveness probe.

Schemas: [backend/app/schemas.py](backend/app/schemas.py).

## Key design choices

- **RAG for all document sizes** — single code path; small docs naturally retrieve most of their chunks anyway.
- **Vision LLM at parse time, not multimodal embeddings** — keeps a single text embedding space and avoids CLIP-style retrieval quality loss.
- **Multi-query retrieval** — compliance questions span multiple contract sections (e.g. Section 6.6 + Exhibit G); one embedding per question misses evidence.
- **Batched query embedding** — 25 sub-queries → 1 Voyage call dropped retrieval from ~4.5s to ~0.4s.
- **Rank-then-doc-order** — pick chunks by score, hand them to the LLM in reading order.
- **Forced `tool_use` + tolerant coercion** — tool schema constrains shape; analyzer only patches mild drift (clamp confidence, placeholder empty quotes).
- **Two-step API** — same index backs analyze *and* chat without re-running upstream stages.
- **SQLite + Qdrant session store** — metadata in SQLite, vectors in Qdrant; `EmbeddingIndex.attach()` rebuilds the handle across restarts.
- **Position-preserving failure isolation** — one question's analyzer failure doesn't take down the other four.

Full rationale: [details.md](details.md) §14 and [PRD.md](PRD.md) §6–§8.
