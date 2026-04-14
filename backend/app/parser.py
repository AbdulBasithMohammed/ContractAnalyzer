# Stage 1: PDF Parsing
# PyMuPDF for everything (text, tables, image detection, rendering) + Claude Haiku (vision)

import base64
import io
import logging
import re

import anthropic
import fitz  # PyMuPDF
from PIL import Image

from .config import settings
from .schemas import PageContent, TableBlock

logger = logging.getLogger(__name__)

# Heading patterns for contract documents. HEADING_RE matches a single line
# (used by the parser for y-attributed heading tracking); SECTION_RE is the
# MULTILINE version (re-exported for the chunker to split full documents).
# IGNORECASE is intentionally NOT set — ARTICLE is uppercase-only, while
# Exhibit/Schedule/Appendix/Section appear in mixed case and are handled by
# explicit alternation below.
_HEADING_PATTERN = (
    r"^(?:"
    r"\d+\.[\d.]*\s+"                    # "1. ", "1.1 ", "1.1.1 "
    r"|ARTICLE\s+[IVXLC\d]+"            # "ARTICLE I", "ARTICLE 1"
    r"|[Ee]xhibit\s+[A-Z\d]"            # "Exhibit A"
    r"|[Ss]chedule\s+[\dA-Z]"           # "Schedule 1"
    r"|[Aa]ppendix\s+[A-Z\d]"           # "Appendix A"
    r"|SECTION\s+\d+"                   # "SECTION 1"
    r")"
)
HEADING_RE = re.compile(_HEADING_PATTERN)
SECTION_RE = re.compile(_HEADING_PATTERN, re.MULTILINE)

# Safety cap for extracted headings. Legitimate contract titles are
# short; anything longer is a symptom of merged body text we didn't trim.
_MAX_HEADING_LEN = 150


def clean_heading(line: str) -> str | None:
    """Extract just the heading title from a line that may also contain
    the first sentence of the section's body.

    PyMuPDF's block extraction frequently merges a heading and its opening
    sentence onto the same visual line (no `\\n` between them). Taking the
    raw first line as the heading — as we did originally — pollutes the
    section label with prose ("2.2 Service Description (semi-structured).
    The parties intend the following summary to describe the ").

    Strategy: match the heading prefix, then cut at the first ". " that
    isn't part of the numbering (so "ARTICLE I. Definitions. Body" still
    keeps "Definitions"). Headings without any internal period-space
    are returned whole.
    """
    m = HEADING_RE.match(line)
    if not m:
        return None

    title_start = m.end()
    rest = line[title_start:]

    # Skip a leading ". " which belongs to the numbering itself
    # (e.g., "ARTICLE I" + ". Definitions. Body...").
    search_from = 2 if rest.startswith(". ") else 0
    sep = rest.find(". ", search_from)

    if sep == -1:
        heading = line.rstrip()
    else:
        heading = line[: title_start + sep + 1]  # include the trailing period

    if len(heading) > _MAX_HEADING_LEN:
        heading = heading[:_MAX_HEADING_LEN].rstrip() + "…"
    return heading

# Formats Claude vision API accepts
_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "jpg", "gif", "webp"}

# Table/prose overlap: a text block is treated as part of a table when more
# than this fraction of its area overlaps any table bbox.
_TABLE_OVERLAP_THRESHOLD = 0.5

# Scanned-page gating: flag text as "garbage" when the ratio of letters /
# digits / common punctuation drops below this. PDFs with broken CID font
# maps produce long strings of PUA chars or (cid:N) tokens that sail past a
# pure length check, so we gate on content quality as well.
_MIN_READABLE_RATIO = 0.7
_READABLE_EXTRA = set(" .,;:!?()[]-'\"/%$&@#*+=<>\n\t")


async def parse_pdf(file_path: str) -> list[PageContent]:
    """Parse a PDF into structured page content.

    Three modes per page:
    - text_only: sufficient text, no images → text extraction only
    - scanned: sparse text + images → full-page vision OCR
    - mixed: sufficient text + significant images → text + image descriptions

    Tables are extracted separately from prose text. Text inside a table's
    bounding box is excluded from `text` so tables aren't double-indexed.
    """
    pages: list[PageContent] = []
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    doc = fitz.open(file_path)

    # Last heading seen across pages — tables at the top of a page inherit
    # this if no earlier heading appears above them on the current page.
    last_heading_across_pages: str | None = None

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Extract tables first — we need their bboxes to filter prose
            tables = _extract_tables(page, page_num)
            table_rects = [fitz.Rect(t["bbox"]) for t in tables]

            # 2. Extract prose text + per-page headings (y-position aware)
            text, headings = _extract_prose_and_headings(page, table_rects)

            # 3. Attribute each table to the nearest heading above it
            table_blocks: list[TableBlock] = []
            for t in tables:
                y_top = t["bbox"][1]
                header = last_heading_across_pages
                for h_y, h_text in headings:
                    if h_y < y_top:
                        header = h_text
                    else:
                        break
                table_blocks.append(TableBlock(
                    page_number=page_num,
                    markdown=t["markdown"],
                    section_header=header,
                ))

            if headings:
                last_heading_across_pages = headings[-1][1]

            # 3. Image detection & page classification
            images = page.get_images(full=True)
            has_images = len(images) > 0
            is_text_sparse = (
                len(text) < settings.min_text_length
                or _looks_like_garbage(text)
            )

            image_descriptions = ""

            if is_text_sparse and has_images:
                logger.info("Page %d: scanned → vision OCR", page_num)
                image_descriptions = await _vision_ocr_page(client, page, page_num)
                if len(text) < 10:
                    text = ""

            elif not is_text_sparse and has_images:
                significant = _get_significant_images(doc, images)
                if significant:
                    logger.info(
                        "Page %d: mixed → %d significant image(s)", page_num, len(significant)
                    )
                    image_descriptions = await _vision_describe_images(
                        client, significant, page_num
                    )
                else:
                    logger.info("Page %d: text-only (images too small)", page_num)
            else:
                logger.info("Page %d: text-only", page_num)

            pages.append(PageContent(
                page_number=page_num,
                text=text,
                tables=table_blocks,
                image_descriptions=image_descriptions,
            ))
    finally:
        doc.close()

    return pages


# ---------------------------------------------------------------------------
# Text extraction (prose only, tables excluded by bbox)
# ---------------------------------------------------------------------------

def _extract_prose_and_headings(
    page: fitz.Page, table_rects: list[fitz.Rect]
) -> tuple[str, list[tuple[float, str]]]:
    """Extract page prose text and detect headings with their y-positions.

    Returns (prose_text, [(y_top, heading_text), ...] sorted by y).
    Blocks whose center lies inside a table bbox are skipped.
    """
    blocks = page.get_text("blocks")
    # Sort by (y_top, x_left) so our heading-tracking walks top-to-bottom.
    blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))

    parts: list[str] = []
    headings: list[tuple[float, str]] = []

    for block in blocks_sorted:
        x0, y0, x1, y1, block_text, *_ = block
        block_rect = fitz.Rect(x0, y0, x1, y1)
        block_area = block_rect.get_area()
        if block_area > 0 and any(
            (block_rect & tb).get_area() / block_area > _TABLE_OVERLAP_THRESHOLD
            for tb in table_rects
        ):
            continue

        parts.append(block_text)

        first_line = block_text.strip().split("\n", 1)[0]
        heading = clean_heading(first_line)
        if heading:
            headings.append((y0, heading))

    return "\n".join(parts).strip(), headings


def _looks_like_garbage(text: str) -> bool:
    """Heuristic: PDF text quality check for scanned-page gating.

    Returns True when the stripped text is too short to judge, or when the
    ratio of readable chars (letters, digits, common punctuation) falls
    below the threshold. Catches CID-mapping failures and PUA-heavy
    encodings that slip past a pure length check.
    """
    stripped = text.strip()
    if len(stripped) < 50:
        return False
    if "(cid:" in stripped:
        return True
    readable = sum(
        1 for c in stripped if c.isalnum() or c in _READABLE_EXTRA
    )
    return readable / len(stripped) < _MIN_READABLE_RATIO


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def _extract_tables(page: fitz.Page, page_num: int) -> list[dict]:
    """Extract tables as markdown with their bounding boxes.

    Returns a list of dicts: {"markdown": str, "bbox": (x0, y0, x1, y1)}.
    """
    found = page.find_tables()
    if not found.tables:
        return []

    results = []
    for table in found:
        data = table.extract()
        if not data or not data[0]:
            continue

        header = data[0]
        md = "| " + " | ".join(str(cell or "") for cell in header) + " |\n"
        md += "| " + " | ".join("---" for _ in header) + " |\n"
        for row in data[1:]:
            md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"

        results.append({"markdown": md.strip(), "bbox": tuple(table.bbox)})

    return results


# ---------------------------------------------------------------------------
# Image filtering
# ---------------------------------------------------------------------------

def _get_significant_images(doc: fitz.Document, images: list) -> list[dict]:
    """Extract images that exceed the minimum dimension threshold."""
    significant = []
    for img_info in images:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            if not base_image:
                continue
            if (base_image["width"] >= settings.significant_image_min_dim
                    and base_image["height"] >= settings.significant_image_min_dim):
                significant.append(base_image)
        except Exception:
            continue
    return significant


def _to_supported_format(img_bytes: bytes, ext: str) -> tuple[bytes, str]:
    """Ensure image is in a format Claude vision API accepts (jpeg/png/gif/webp).

    Converts unsupported formats (jpx, bmp, tiff, etc.) to PNG.
    """
    if ext.lower() in _SUPPORTED_IMAGE_FORMATS:
        media_type = "image/jpeg" if ext.lower() in ("jpg", "jpeg") else f"image/{ext.lower()}"
        return img_bytes, media_type

    img = Image.open(io.BytesIO(img_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


# ---------------------------------------------------------------------------
# Vision LLM calls
# ---------------------------------------------------------------------------

async def _vision_ocr_page(
    client: anthropic.AsyncAnthropic, page: fitz.Page, page_num: int
) -> str:
    """Render a full page as PNG and OCR via Claude Haiku vision."""
    pix = page.get_pixmap(dpi=150)
    img_b64 = base64.standard_b64encode(pix.tobytes("png")).decode()

    try:
        response = await client.messages.create(
            model=settings.vision_model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Extract all text from this scanned document page (page {page_num}). "
                            "Preserve the original structure including headings, paragraphs, "
                            "lists, and tables. Return only the extracted text, no commentary."
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text
    except Exception as e:
        logger.error("Vision OCR failed for page %d: %s", page_num, e)
        return ""


async def _vision_describe_images(
    client: anthropic.AsyncAnthropic, images: list[dict], page_num: int
) -> str:
    """Describe significant images via Claude Haiku vision."""
    descriptions = []

    for i, img_data in enumerate(images):
        img_bytes, media_type = _to_supported_format(img_data["image"], img_data.get("ext", "png"))
        img_b64 = base64.standard_b64encode(img_bytes).decode()

        try:
            response = await client.messages.create(
                model=settings.vision_model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Describe this image from page {page_num} of a contract document. "
                                "Focus on any text, diagrams, charts, or compliance-relevant "
                                "information. Be concise but thorough."
                            ),
                        },
                    ],
                }],
            )
            descriptions.append(f"[Image {i + 1}]: {response.content[0].text}")
        except Exception as e:
            logger.error("Vision describe failed for image %d on page %d: %s", i + 1, page_num, e)
            continue

    return "\n\n".join(descriptions)
