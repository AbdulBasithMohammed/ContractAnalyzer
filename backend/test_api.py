"""End-to-end smoke test for Stage 6 (FastAPI layer).

Exercises the two-step HTTP flow against the sample contract using
FastAPI's in-process TestClient — no port, no uvicorn boot.

Run from the backend/ dir:  python test_api.py
"""

import logging
import os

from fastapi.testclient import TestClient

from app.main import app

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "Sample Contract.pdf")

logging.basicConfig(
    level=logging.INFO, format="    [%(levelname)s %(name)s] %(message)s"
)


def hr(title: str, char: str = "=") -> None:
    print("\n" + char * 72)
    print(f"  {title}")
    print(char * 72)


def preview(text: str | None, n: int = 180) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed[:n] + ("…" if len(collapsed) > n else "")


def main() -> None:
    pdf_path = os.path.abspath(SAMPLE_PDF)
    print(f"PDF: {pdf_path}")

    client = TestClient(app)

    # --- /api/health ---
    hr("HEALTH")
    r = client.get("/api/health")
    r.raise_for_status()
    print(f"  {r.json()}")

    # --- /api/upload ---
    hr("POST /api/upload")
    with open(pdf_path, "rb") as fh:
        r = client.post(
            "/api/upload",
            files={"file": ("Sample Contract.pdf", fh, "application/pdf")},
        )
    if r.status_code != 200:
        print(f"  FAILED: {r.status_code} {r.text}")
        raise SystemExit(1)
    upload = r.json()
    upload_id = upload["upload_id"]
    print(f"  upload_id:     {upload_id}")
    print(f"  filename:      {upload['filename']}")
    print(f"  pages:         {upload['page_count']}")
    print(f"  chunks:        {upload['chunk_count']}")
    print(f"  parse:         {upload['parse_sec']:.2f}s")
    print(f"  chunk:         {upload['chunk_sec']:.2f}s")
    print(f"  embed:         {upload['embed_sec']:.2f}s")

    # --- /api/analyze/{upload_id} ---
    hr(f"POST /api/analyze/{upload_id}")
    r = client.post(f"/api/analyze/{upload_id}")
    if r.status_code != 200:
        print(f"  FAILED: {r.status_code} {r.text}")
        raise SystemExit(1)
    body = r.json()

    meta = body["metadata"]
    t = meta["timings"]
    print(f"  parse:         {t['parse_sec']:.2f}s")
    print(f"  chunk:         {t['chunk_sec']:.2f}s")
    print(f"  embed:         {t['embed_sec']:.2f}s")
    print(f"  retrieve:      {t['retrieve_sec']:.2f}s")
    print(f"  analyze:       {t['analyze_sec']:.2f}s")
    print(f"  total:         {t['total_sec']:.2f}s")
    print(f"  models:        {meta['models']}")

    print("\n  --- Per-question retrieval stats ---")
    for s in meta["retrieval"]:
        top = f"{s['top_score']:.3f}" if s["top_score"] is not None else "n/a"
        print(
            f"    [{s['question_id']:>36}]  chunks={s['chunks_used']:>2}  "
            f"ctx_tokens={s['context_tokens']:>4}  top_score={top}"
        )

    print("\n  --- Results ---")
    for res in body["results"]:
        q = res["compliance_question"].split(".")[0]
        if res.get("error"):
            print(f"\n  [{q}] ERROR: {res['error']}")
            continue
        print(f"\n  [{q}]")
        print(f"    state:       {res['compliance_state']}")
        print(f"    confidence:  {res['confidence']:.1f}")
        print(f"    quotes:      {preview(res['relevant_quotes'])}")
        print(f"    rationale:   {preview(res['rationale'])}")

    # --- unknown upload_id ---
    hr("POST /api/analyze/<bogus>  (expect 404)")
    r = client.post("/api/analyze/does-not-exist")
    print(f"  status={r.status_code}  body={r.json()}")

    hr("DONE", char="-")


if __name__ == "__main__":
    main()
