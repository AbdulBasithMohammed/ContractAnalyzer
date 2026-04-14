from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --- Parser output ---

class TableBlock(BaseModel):
    page_number: int
    markdown: str
    # Nearest heading appearing above the table (computed at parse time
    # using block positions). None if no heading precedes this table.
    section_header: str | None = None


class PageContent(BaseModel):
    page_number: int
    text: str  # prose only — tables are excluded and emitted separately
    tables: list[TableBlock] = []
    image_descriptions: str = ""


# --- Chunker output ---

class Chunk(BaseModel):
    text: str
    page_numbers: list[int]
    section_header: str | None = None
    chunk_index: int


# --- Retriever output ---

class RetrievedChunk(Chunk):
    score: float


# --- Compliance analysis ---

class ComplianceState(str, Enum):
    FULLY_COMPLIANT = "Fully Compliant"
    PARTIALLY_COMPLIANT = "Partially Compliant"
    NON_COMPLIANT = "Non-Compliant"


class ComplianceResult(BaseModel):
    """One verdict per compliance question.

    Invariant: `error is None` iff the analyzer succeeded, in which case all
    other fields are populated. When analysis fails for this question, `error`
    carries a short message and the verdict fields are left null so the
    position in the overall results list is preserved.
    """
    compliance_question: str
    compliance_state: ComplianceState | None = None
    confidence: float | None = Field(default=None, ge=0, le=100)
    relevant_quotes: str | None = None
    rationale: str | None = None
    error: str | None = None


# --- Response metadata ---

class StageTimings(BaseModel):
    parse_sec: float
    chunk_sec: float
    embed_sec: float
    retrieve_sec: float
    analyze_sec: float
    total_sec: float


class QuestionRetrievalStats(BaseModel):
    question_id: str
    chunks_used: int
    context_tokens: int
    top_score: float | None = None


class AnalysisMetadata(BaseModel):
    upload_id: str
    filename: str
    page_count: int
    chunk_count: int
    timings: StageTimings
    retrieval: list[QuestionRetrievalStats]
    models: dict[str, str]


class AnalysisResponse(BaseModel):
    results: list[ComplianceResult]
    metadata: AnalysisMetadata


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    page_count: int
    chunk_count: int
    parse_sec: float
    chunk_sec: float
    embed_sec: float


# --- Chat ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Client sends the raw user turn plus the prior conversation.

    History is client-owned — the server re-runs retrieval for each turn
    against the latest `message`, so prior assistant responses in `history`
    should be the user-visible text only (no injected context excerpts).
    """
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class ChatSource(BaseModel):
    chunk_index: int
    section_header: str | None = None
    page_numbers: list[int]
    score: float
