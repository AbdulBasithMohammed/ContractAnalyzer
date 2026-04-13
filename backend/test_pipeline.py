"""Quick smoke test for Stage 1 (Parser) and Stage 2 (Chunker)."""

import asyncio
import os

# Ensure dummy env vars so config loads without a real .env
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app.parser import parse_pdf
from app.chunker import chunk_pages

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "Sample Contract.pdf")


async def main():
    pdf_path = os.path.abspath(SAMPLE_PDF)
    print(f"Using: {pdf_path}\n")

    try:

        # --- Stage 1: Parse ---
        print("=" * 60)
        print("STAGE 1: PARSER")
        print("=" * 60)
        pages = await parse_pdf(pdf_path)
        print(f"\nParsed {len(pages)} pages:\n")
        for page in pages:
            text_preview = page.text[:120].replace("\n", " ") if page.text else "(empty)"
            print(f"  Page {page.page_number}:")
            print(f"    Text:   {text_preview}...")
            print(f"    Tables: {len(page.tables)}")
            print(f"    Images: {'Yes' if page.image_descriptions else 'No'}")
            print()

        # --- Stage 2: Chunk ---
        print("=" * 60)
        print("STAGE 2: CHUNKER")
        print("=" * 60)
        chunks = chunk_pages(pages)
        print(f"\nProduced {len(chunks)} chunks:\n")
        for chunk in chunks:
            token_est = len(chunk.text.split())
            text_preview = chunk.text[:100].replace("\n", " ")
            print(f"  Chunk {chunk.chunk_index}:")
            print(f"    Header: {chunk.section_header or '(preamble)'}")
            print(f"    Pages:  {chunk.page_numbers}")
            print(f"    Tokens: ~{token_est}")
            print(f"    Text:   {text_preview}...")
            print()

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
