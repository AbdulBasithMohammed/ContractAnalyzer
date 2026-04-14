# Stage 3: Embedding
# Voyage voyage-law-2 embeddings stored in Qdrant (server or in-memory).

import logging
import time

import voyageai
import voyageai.error
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings
from .schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# Process-wide Voyage client — created on first use.
_voyage_client: voyageai.Client | None = None


def _get_voyage_client() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    return _voyage_client


def _make_qdrant_client() -> QdrantClient:
    """Build a client for either an embedded (`:memory:`) or server Qdrant.

    qdrant-client's param names diverge: in-process wants `location=`, a
    server wants `url=`. Passing a URL via `location=` works on some
    versions but isn't the documented path; branch explicitly.
    """
    loc = settings.qdrant_location
    if loc.startswith(":"):
        return QdrantClient(location=loc)
    return QdrantClient(url=loc)


class EmbeddingIndex:
    """Per-upload vector index backed by Qdrant.

    Persistent when QDRANT_LOCATION is a server URL, ephemeral when it's
    ":memory:". `attach()` re-binds to a pre-existing collection so the
    SQLite session store can rebuild the index handle after a restart
    without re-embedding.
    """

    def __init__(self, upload_id: str, *, create: bool = True):
        self.upload_id = upload_id
        self.collection = f"upload_{upload_id}"
        self.client = _make_qdrant_client()
        if create:
            # Drop-and-create so re-uploading the same upload_id starts
            # from a clean collection. UUID upload_ids mean collisions
            # don't happen in practice, but the idempotency is cheap.
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    @classmethod
    def attach(cls, upload_id: str) -> "EmbeddingIndex | None":
        """Bind to an existing Qdrant collection without recreating it.

        Returns None if the collection is missing — the caller treats
        that as "session gone" and 404s the request.
        """
        inst = cls.__new__(cls)
        inst.upload_id = upload_id
        inst.collection = f"upload_{upload_id}"
        inst.client = _make_qdrant_client()
        if not inst.client.collection_exists(inst.collection):
            return None
        return inst

    def build(self, chunks: list[Chunk]) -> None:
        """Embed and upsert all chunks. No-op on empty input."""
        if not chunks:
            logger.warning("build() called with 0 chunks for %s", self.upload_id)
            return

        texts = [_compose_embed_text(c) for c in chunks]
        vectors = _embed_documents(texts)

        points = [
            PointStruct(
                id=chunk.chunk_index,
                vector=vector,
                payload={
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "page_numbers": chunk.page_numbers,
                    "section_header": chunk.section_header,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Indexed %d chunks into %s", len(points), self.collection)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed a query and return the top-k nearest chunks."""
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(
        self, queries: list[str], top_k: int | None = None
    ) -> list[list[RetrievedChunk]]:
        """Embed all queries in one API call, then run per-query vector search.

        Collapses N Voyage HTTP round-trips into 1 — the dominant cost for
        small in-memory indexes where Qdrant itself is ~free per query.
        Returns results aligned with the input order.
        """
        top_k = top_k or settings.retrieval_top_k
        if not queries:
            return []
        vectors = _embed_queries(queries)
        results: list[list[RetrievedChunk]] = []
        for vector in vectors:
            hits = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
            ).points
            results.append([
                RetrievedChunk(
                    text=hit.payload["text"],
                    page_numbers=hit.payload["page_numbers"],
                    section_header=hit.payload["section_header"],
                    chunk_index=hit.payload["chunk_index"],
                    score=float(hit.score),
                )
                for hit in hits
            ])
        return results


def build_index(upload_id: str, chunks: list[Chunk]) -> EmbeddingIndex:
    """Convenience constructor: create + populate in one call."""
    index = EmbeddingIndex(upload_id)
    index.build(chunks)
    return index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compose_embed_text(chunk: Chunk) -> str:
    """Prepend section header to chunk body for richer semantic context."""
    if chunk.section_header:
        return f"{chunk.section_header}\n\n{chunk.text}"
    return chunk.text


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus texts. Batched + throttled to respect Voyage rate limits.

    Default settings are tuned for the free tier (3 RPM / 10K TPM):
    batch_size=8 keeps each request under 10K tokens; a 21s gap between
    requests respects the 3 RPM ceiling. On paid tiers, drop
    `embedding_batch_sleep_sec` to 0 and raise `embedding_batch_size`.
    """
    batch_size = settings.embedding_batch_size
    sleep_sec = settings.embedding_batch_sleep_sec
    vectors: list[list[float]] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_idx = i // batch_size + 1
        batch = texts[i : i + batch_size]
        logger.info("Embedding batch %d/%d (%d texts)", batch_idx, n_batches, len(batch))
        vectors.extend(_embed_with_retry(batch, "document"))

        # Throttle between batches (not after the last).
        if i + batch_size < len(texts) and sleep_sec > 0:
            time.sleep(sleep_sec)

    return vectors


def _embed_queries(texts: list[str]) -> list[list[float]]:
    """Batch-embed query strings. input_type='query' is asymmetric to 'document'."""
    return _embed_with_retry(texts, "query")


def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    """Call Voyage embed with exponential backoff on rate-limit errors."""
    client = _get_voyage_client()
    max_retries = settings.embedding_max_retries
    for attempt in range(max_retries):
        try:
            result = client.embed(
                texts=texts,
                model=settings.embedding_model,
                input_type=input_type,
            )
            return result.embeddings
        except voyageai.error.RateLimitError as e:
            if attempt == max_retries - 1:
                logger.error("Rate-limit retries exhausted: %s", e)
                raise
            # Backoff is independent of batch throttle — we always want a real
            # wait on rate-limit even when batch sleep is 0 (paid tier).
            wait = max(5.0, settings.embedding_batch_sleep_sec) * (2 ** attempt)
            logger.warning(
                "Voyage rate-limited (attempt %d/%d); sleeping %.1fs",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # defensive
