# Stage 2: Chunking
# Section-aware prose splitting + tables as atomic chunks.

import re

import tiktoken

from .config import settings
from .schemas import Chunk, PageContent

# Heading patterns for contract documents.
# IGNORECASE is intentionally NOT set — ARTICLE is uppercase-only, while
# Exhibit/Schedule/Appendix/Section appear in mixed case, handled by
# explicit alternation below.
_SECTION_RE = re.compile(
    r"^(?:"
    r"\d+\.[\d.]*\s+"                    # "1. ", "1.1 ", "1.1.1 "
    r"|ARTICLE\s+[IVXLC\d]+"            # "ARTICLE I", "ARTICLE 1"
    r"|[Ee]xhibit\s+[A-Z\d]"            # "Exhibit A"
    r"|[Ss]chedule\s+[\dA-Z]"           # "Schedule 1"
    r"|[Aa]ppendix\s+[A-Z\d]"           # "Appendix A"
    r"|SECTION\s+\d+"                   # "SECTION 1"
    r")",
    re.MULTILINE,
)

# cl100k_base matches GPT-4 / Claude's tokenization closely enough for sizing.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    """Split parsed pages into retrieval-ready chunks.

    Pipeline:
    1. Each table on each page → one atomic chunk (never split).
    2. Prose text from all pages is joined, section-split by heading regex.
    3. Sections within token limit → atomic chunk.
    4. Oversized sections → paragraph-level fallback split with overlap.
    """
    if not pages:
        return []

    # --- Build combined prose doc with page boundary tracking ----------
    doc_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []  # (char_start, char_end, page_num)

    offset = 0
    for page in pages:
        text = page.text
        if page.image_descriptions:
            text = f"{text}\n\n{page.image_descriptions}" if text else page.image_descriptions
        if not text:
            continue
        doc_parts.append(text)
        page_ranges.append((offset, offset + len(text), page.page_number))
        offset += len(text) + 2  # +2 for "\n\n" join separator

    doc_text = "\n\n".join(doc_parts)
    sections = _split_into_sections(doc_text) if doc_text.strip() else []

    chunks: list[Chunk] = []
    chunk_idx = 0

    # --- Table chunks (one per table, header attributed by parser) -----
    for page in pages:
        for table in page.tables:
            chunks.append(Chunk(
                text=table.markdown,
                page_numbers=[table.page_number],
                section_header=table.section_header,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

    # --- Prose chunks from sections ------------------------------------
    for header, body, section_start in sections:
        section_text = f"{header}\n{body}".strip() if header else body.strip()
        if not section_text:
            continue

        if _count_tokens(section_text) <= settings.max_chunk_tokens:
            chunks.append(Chunk(
                text=section_text,
                page_numbers=_get_page_numbers(
                    section_start,
                    section_start + len(section_text),
                    page_ranges,
                ),
                section_header=header,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        else:
            for sub_text, sub_offset in _split_oversized(section_text):
                abs_start = section_start + sub_offset
                chunks.append(Chunk(
                    text=sub_text,
                    page_numbers=_get_page_numbers(
                        abs_start,
                        abs_start + len(sub_text),
                        page_ranges,
                    ),
                    section_header=header,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------


def _split_into_sections(text: str) -> list[tuple[str | None, str, int]]:
    """Split document at heading boundaries.

    Returns list of (header | None, body, start_char_offset_in_text).
    """
    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return [(None, text, 0)]

    sections: list[tuple[str | None, str, int]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble, 0))

    for i, match in enumerate(matches):
        header_end = text.find("\n", match.start())
        if header_end == -1:
            header_end = len(text)

        header = text[match.start() : header_end].strip()
        body_start = header_end + 1
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        sections.append((header, body, match.start()))

    return sections


# ---------------------------------------------------------------------------
# Oversized-section fallback (prose only — tables have been extracted)
# ---------------------------------------------------------------------------


def _split_oversized(text: str) -> list[tuple[str, int]]:
    """Split oversized prose into token-limited chunks with overlap.

    Splits on paragraph boundaries (blank lines). If a single paragraph
    still exceeds the limit, splits on sentences. Returns (chunk_text,
    start_offset_in_text).
    """
    max_tokens = settings.max_chunk_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    # Split on blank lines, keeping offsets accurate.
    paragraphs = _segment_by_blank_lines(text)
    # Each paragraph over the limit → sentence-split in place.
    units: list[tuple[str, int]] = []
    for p_text, p_offset in paragraphs:
        if _count_tokens(p_text) <= max_tokens:
            units.append((p_text, p_offset))
        else:
            for s_text, s_offset_in_p in _segment_by_sentences(p_text):
                units.append((s_text, p_offset + s_offset_in_p))

    # Greedy pack units into chunks; walk back for overlap.
    chunks: list[tuple[str, int]] = []
    unit_tokens = [_count_tokens(u[0]) for u in units]
    start = 0
    while start < len(units):
        end = start
        total = 0
        while end < len(units) and total + unit_tokens[end] <= max_tokens:
            total += unit_tokens[end]
            end += 1

        if end == start:  # single unit already exceeds limit — take it anyway
            end = start + 1

        chunk_text = "\n\n".join(u[0] for u in units[start:end])
        chunks.append((chunk_text, units[start][1]))

        if end >= len(units):
            break

        # Overlap: walk back from `end`, including units until we hit overlap budget.
        overlap_total = 0
        overlap_start = end
        while (overlap_start > start + 1
               and overlap_total + unit_tokens[overlap_start - 1] <= overlap_tokens):
            overlap_start -= 1
            overlap_total += unit_tokens[overlap_start]

        start = overlap_start

    return chunks


def _segment_by_blank_lines(text: str) -> list[tuple[str, int]]:
    """Split on runs of blank lines. Returns (segment, offset_in_text)."""
    segments: list[tuple[str, int]] = []
    for match in re.finditer(r"(.+?)(?:\n\s*\n|\Z)", text, flags=re.DOTALL):
        segment = match.group(1).strip()
        if segment:
            segments.append((segment, match.start(1)))
    return segments if segments else [(text, 0)]


def _segment_by_sentences(text: str) -> list[tuple[str, int]]:
    """Lightweight sentence split on ". ", "! ", "? " boundaries."""
    segments: list[tuple[str, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]\s+", text):
        segment = text[start : match.end()].strip()
        if segment:
            segments.append((segment, start))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        segments.append((tail, start))
    return segments if segments else [(text, 0)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def _get_page_numbers(
    start: int,
    end: int,
    page_ranges: list[tuple[int, int, int]],
) -> list[int]:
    """Return sorted, deduplicated page numbers overlapping [start, end)."""
    if not page_ranges:
        raise ValueError("page_ranges must not be empty")

    seen: set[int] = set()
    for pg_start, pg_end, page_num in page_ranges:
        if pg_start < end and pg_end > start:
            seen.add(page_num)

    return sorted(seen)


