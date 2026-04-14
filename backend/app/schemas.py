from enum import Enum

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
    compliance_question: str
    compliance_state: ComplianceState
    confidence: float = Field(ge=0, le=100)
    relevant_quotes: str
    rationale: str


class AnalysisResponse(BaseModel):
    results: list[ComplianceResult]
    metadata: dict = Field(default_factory=dict)


# --- Chat ---

class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    sources: list[dict] = Field(default_factory=list)
