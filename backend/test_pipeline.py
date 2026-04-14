"""End-to-end smoke test for Stages 1-4 (Parse, Chunk, Embed, Retrieve).

Run from the backend/ dir:  python test_pipeline.py

Prints a human-readable trace of what each stage produces, including
Qdrant collection internals, embedding dimensions, vector samples,
live similarity-search results, and multi-query retrieval output per
compliance question against the sample contract.
"""

import asyncio
import logging
import os
import time

from app.analyzer import analyze_all
from app.chunker import chunk_pages
from app.config import settings
from app.embedder import EmbeddingIndex, _compose_embed_text
from app.parser import parse_pdf
from app.retriever import COMPLIANCE_QUESTIONS, retrieve_all

# Show embedder batch progress live so the throttled free-tier run isn't silent.
logging.basicConfig(
    level=logging.INFO, format="    [%(levelname)s %(name)s] %(message)s"
)

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "Sample Contract.pdf")
UPLOAD_ID = "smoke-test"

TEST_QUERIES = [
    "password management and credential vaulting requirements",
    "breach notification timeline for security incidents",
    "multi-factor authentication for administrative access",
    "encryption at rest and in transit",
    "subprocessor approval and audit rights",
]


def hr(title: str, char: str = "=") -> None:
    print("\n" + char * 72)
    print(f"  {title}")
    print(char * 72)


def preview(text: str, n: int = 140) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:n] + ("..." if len(collapsed) > n else "")


async def main():
    pdf_path = os.path.abspath(SAMPLE_PDF)
    print(f"PDF: {pdf_path}")

    # ------------------------------------------------------------------
    # Stage 1 — Parser
    # ------------------------------------------------------------------
    hr("STAGE 1 — PARSER")
    t0 = time.perf_counter()
    pages = await parse_pdf(pdf_path)
    t_parse = time.perf_counter() - t0

    total_tables = sum(len(p.tables) for p in pages)
    pages_with_images = sum(1 for p in pages if p.image_descriptions)
    total_chars = sum(len(p.text) for p in pages)

    print(f"  Pages parsed:          {len(pages)}")
    print(f"  Prose chars total:     {total_chars:,}")
    print(f"  Tables extracted:      {total_tables}")
    print(f"  Pages w/ vision desc:  {pages_with_images}")
    print(f"  Elapsed:               {t_parse:.2f}s")

    print("\n  --- Per-page summary ---")
    for p in pages:
        marks = []
        if p.text:
            marks.append(f"text={len(p.text):>5}ch")
        if p.tables:
            marks.append(f"tables={len(p.tables)}")
        if p.image_descriptions:
            marks.append("img-desc")
        print(f"    p{p.page_number:>2}  {'  '.join(marks) or '(empty)'}")

    # ------------------------------------------------------------------
    # Stage 2 — Chunker
    # ------------------------------------------------------------------
    hr("STAGE 2 — CHUNKER")
    t0 = time.perf_counter()
    chunks = chunk_pages(pages)
    t_chunk = time.perf_counter() - t0

    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    prose_chunks = [c for c in chunks if c not in table_chunks]
    header_coverage = sum(1 for c in chunks if c.section_header)

    print(f"  Chunks produced:       {len(chunks)}")
    print(f"    table chunks:        {len(table_chunks)}")
    print(f"    prose chunks:        {len(prose_chunks)}")
    print(f"  With section_header:   {header_coverage}/{len(chunks)}")
    print(f"  Elapsed:               {t_chunk:.2f}s")

    print("\n  --- First 6 chunks ---")
    for c in chunks[:6]:
        kind = "TABLE" if c.text.lstrip().startswith("|") else "PROSE"
        header = c.section_header or "(preamble)"
        print(f"    [{c.chunk_index:>3}] {kind}  pages={c.page_numbers}  header={header!r}")
        print(f"          {preview(c.text, 120)}")

    # ------------------------------------------------------------------
    # Stage 3 — Embedder
    # ------------------------------------------------------------------
    hr("STAGE 3 — EMBEDDER (voyage-law-2 → in-memory Qdrant)")

    index = EmbeddingIndex(UPLOAD_ID)
    print(f"  Collection name:       {index.collection}")
    print(f"  Qdrant location:       :memory:")
    print(f"  Model:                 {settings.embedding_model}")
    print(f"  Batch size / sleep:    {settings.embedding_batch_size} / "
          f"{settings.embedding_batch_sleep_sec}s")

    # Show what text actually goes to the embedding API for one chunk.
    sample = next((c for c in chunks if c.section_header), chunks[0])
    composed = _compose_embed_text(sample)
    print(f"\n  --- Sample embed input (chunk {sample.chunk_index}) ---")
    print(f"    section_header: {sample.section_header!r}")
    print(f"    composed text (first 200 chars):")
    print(f"    {preview(composed, 200)}")

    # Estimate wall time (dominated by the free-tier throttle).
    n_batches = (len(chunks) + settings.embedding_batch_size - 1) // settings.embedding_batch_size
    est = (n_batches - 1) * settings.embedding_batch_sleep_sec
    print(f"\n  --- Building index over {len(chunks)} chunks "
          f"({n_batches} batches, est. ≥{est:.0f}s) ---")
    t0 = time.perf_counter()
    index.build(chunks)
    t_build = time.perf_counter() - t0
    print(f"    Elapsed: {t_build:.2f}s")

    # Inspect Qdrant state.
    info = index.client.get_collection(index.collection)
    print(f"\n  --- Qdrant collection state ---")
    print(f"    name:            {index.collection}")
    print(f"    status:          {info.status}")
    print(f"    points_count:    {info.points_count}")
    print(f"    indexed_vectors: {info.indexed_vectors_count}")
    print(f"    segments:        {info.segments_count}")
    print(f"    vector config:   size={info.config.params.vectors.size}  "
          f"distance={info.config.params.vectors.distance}")

    # Pull 3 stored points back out to prove payloads round-trip.
    scrolled, _ = index.client.scroll(
        collection_name=index.collection,
        limit=3,
        with_payload=True,
        with_vectors=True,
    )
    print(f"\n  --- 3 stored points (scroll) ---")
    for pt in scrolled:
        pl = pt.payload
        vec = pt.vector
        print(f"    id={pt.id}  dim={len(vec)}  first3={[round(v, 4) for v in vec[:3]]}")
        print(f"      pages={pl['page_numbers']}  header={pl['section_header']!r}")
        print(f"      text: {preview(pl['text'], 100)}")

    # ------------------------------------------------------------------
    # Retrieval sanity check — this previews Stage 4 behavior.
    # ------------------------------------------------------------------
    hr("STAGE 3 — RETRIEVAL SANITY CHECK (top-3 per query)")
    for q in TEST_QUERIES:
        print(f"\n  Q: {q!r}")
        hits = index.search(q, top_k=3)
        for h in hits:
            header = h.section_header or "(preamble)"
            print(f"    score={h.score:.4f}  idx={h.chunk_index:>3}  "
                  f"pages={h.page_numbers}  header={header!r}")
            print(f"      {preview(h.text, 110)}")

    # ------------------------------------------------------------------
    # Stage 4 — Retriever (multi-query per compliance question)
    # ------------------------------------------------------------------
    hr("STAGE 4 — RETRIEVER (multi-query per compliance question)")
    print(f"  Questions:             {len(COMPLIANCE_QUESTIONS)}")
    print(f"  Sub-queries/question:  "
          f"{[len(q.sub_queries) for q in COMPLIANCE_QUESTIONS]}")
    print(f"  top_k per sub-query:   {settings.retrieval_top_k}")
    print(f"  Context token budget:  {settings.retrieval_context_tokens}")

    t0 = time.perf_counter()
    retrieved = retrieve_all(index)
    t_retrieve = time.perf_counter() - t0
    print(f"  Elapsed (all 5):       {t_retrieve:.2f}s")

    for q in COMPLIANCE_QUESTIONS:
        hits = retrieved[q.id]
        print(f"\n  [{q.id}] {q.title} — {len(hits)} chunks in doc order")
        for h in hits:
            header = h.section_header or "(preamble)"
            print(f"    score={h.score:.4f}  idx={h.chunk_index:>3}  "
                  f"pages={h.page_numbers}  header={header!r}")
            print(f"      {preview(h.text, 110)}")

    # ------------------------------------------------------------------
    # Stage 5 — Analyzer (5 parallel Claude Sonnet calls)
    # ------------------------------------------------------------------
    hr("STAGE 5 — ANALYZER (Claude Sonnet, 5 parallel calls)")
    print(f"  Model:                 {settings.analysis_model}")
    print(f"  Max tokens/response:   {settings.analysis_max_tokens}")

    t0 = time.perf_counter()
    results = await analyze_all(retrieved)
    t_analyze = time.perf_counter() - t0
    print(f"  Elapsed (all 5):       {t_analyze:.2f}s")

    for q, result in zip(COMPLIANCE_QUESTIONS, results):
        print(f"\n  [{q.id}] {q.title}")
        print(f"    state:       {result.compliance_state.value}")
        print(f"    confidence:  {result.confidence:.1f}")
        print(f"    quotes:      {preview(result.relevant_quotes, 200)}")
        print(f"    rationale:   {preview(result.rationale, 200)}")

    # ------------------------------------------------------------------
    hr("DONE", char="-")
    print(f"  parse:     {t_parse:>5.2f}s")
    print(f"  chunk:     {t_chunk:>5.2f}s")
    print(f"  embed:     {t_build:>5.2f}s  ({len(chunks)} chunks)")
    print(f"  retrieve:  {t_retrieve:>5.2f}s  (5 questions)")
    print(f"  analyze:   {t_analyze:>5.2f}s  (5 questions)")


if __name__ == "__main__":
    asyncio.run(main())
