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

# Formats Claude vision API accepts
_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "jpg", "gif", "webp"}

# Matches heading lines (same patterns as chunker). Duplicated here so the
# parser can attribute tables to their nearest preceding heading using
# positional info that is only available during parsing.
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+\.[\d.]*\s+"
    r"|ARTICLE\s+[IVXLC\d]+"
    r"|[Ee]xhibit\s+[A-Z\d]"
    r"|[Ss]chedule\s+[\dA-Z]"
    r"|[Aa]ppendix\s+[A-Z\d]"
    r"|SECTION\s+\d+"
    r")"
)


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
            is_text_sparse = len(text) < settings.min_text_length

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
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        if any(tb.x0 <= cx <= tb.x1 and tb.y0 <= cy <= tb.y1 for tb in table_rects):
            continue

        parts.append(block_text)

        first_line = block_text.strip().split("\n", 1)[0]
        if _HEADING_RE.match(first_line):
            headings.append((y0, first_line))

    return "\n".join(parts).strip(), headings


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
