"""Streamlit UI for the Contract Compliance Analyzer.

Two-step flow mirrors the backend: POST /api/upload → POST /api/analyze/{id}.
Shows per-stage timings as each call resolves (via st.status). Renders 5
verdict cards with rationale, quotes, and retrieval diagnostics. Bonus chat
panel streams answers (NDJSON) from POST /api/chat/{id} and reuses the
session's in-memory EmbeddingIndex.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import requests
import streamlit as st


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
UPLOAD_TIMEOUT = 180  # seconds; parse+embed on a 100-page PDF ~tens of seconds
ANALYZE_TIMEOUT = 180
CHAT_TIMEOUT = 120

# Survive browser refresh: persist the last analysis to a small JSON file
# in the OS temp dir and rehydrate on startup. The backend's in-memory
# session is still ephemeral, so chat will 404 after a uvicorn restart —
# we handle that case with a banner instead of silent failure.
_CACHE_PATH = Path(tempfile.gettempdir()) / "cca_frontend_cache.json"

# Quote format emitted by the prompt:
#   [§6.6 Password Management Standard | p.7]: "verbatim text"
_QUOTE_RE = re.compile(r'^\s*\[([^\]]+)\]\s*:\s*[\"“](.+?)[\"”]\s*$', re.DOTALL)
_NO_EVIDENCE_PREFIX = "No relevant provisions"

# Display titles for the 5 fixed compliance questions, keyed on the
# question_id that the backend returns in AnalysisMetadata.retrieval.
# Kept here (not derived via str.title()) so "network_authn_authz" and
# "security_training_background_checks" render readably.
QUESTION_TITLES = {
    "password_management": "Password Management",
    "it_asset_management": "IT Asset Management",
    "security_training_background_checks": "Security Training & Background Checks",
    "data_in_transit_encryption": "Data in Transit Encryption",
    "network_authn_authz": "Network Authentication & Authorization",
}

STATE_STYLE = {
    "Fully Compliant":     {"emoji": "🟢", "color": "#16a34a"},
    "Partially Compliant": {"emoji": "🟡", "color": "#d97706"},
    "Non-Compliant":       {"emoji": "🔴", "color": "#dc2626"},
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    defaults = {
        "upload_id": None,
        "upload_meta": None,      # UploadResponse dict
        "analysis": None,         # AnalysisResponse dict
        "chat_history": [],       # list[{role, content, sources?}]
        "current_file_id": None,  # tracks uploader to detect new files
        "session_stale": False,   # true if rehydrated but backend session is gone
        "jump_to_tab": None,      # set by overview row buttons; consumed by main()
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # Rehydrate from disk if this is a fresh page load.
    if st.session_state.analysis is None:
        cached = _load_cache()
        if cached:
            st.session_state.upload_id = cached.get("upload_id")
            st.session_state.upload_meta = cached.get("upload_meta")
            st.session_state.analysis = cached.get("analysis")
            # With persistent Qdrant + SQLite the backend usually still
            # holds this session after a restart. Probe it so we only
            # show the "re-upload to chat" banner when it's truly gone.
            st.session_state.session_stale = not _backend_has_session(
                st.session_state.upload_id
            )


def _backend_has_session(upload_id: str | None) -> bool:
    """Return True iff the backend still knows this upload_id."""
    if not upload_id:
        return False
    try:
        resp = requests.get(
            f"{BACKEND_URL}/api/session/{upload_id}",
            timeout=3,
        )
    except requests.RequestException:
        return False
    return resp.status_code == 200


def _clear_analysis(upload_id: str | None) -> None:
    """Drop the backend session, wipe local cache, and reset UI state."""
    if upload_id:
        try:
            requests.delete(
                f"{BACKEND_URL}/api/session/{upload_id}",
                timeout=5,
            )
        except requests.RequestException:
            # Backend unreachable is fine — we still clear local state
            # so the UI returns to empty; the stale row will self-clean
            # on next access via the orphan-metadata branch in sessions.get.
            pass
    reset_document_state()
    st.session_state.current_file_id = None


def reset_document_state() -> None:
    st.session_state.upload_id = None
    st.session_state.upload_meta = None
    st.session_state.analysis = None
    st.session_state.chat_history = []
    st.session_state.session_stale = False
    _clear_cache()


def _save_cache() -> None:
    if st.session_state.get("analysis") is None:
        return
    try:
        _CACHE_PATH.write_text(json.dumps({
            "upload_id": st.session_state.upload_id,
            "upload_meta": st.session_state.upload_meta,
            "analysis": st.session_state.analysis,
        }))
    except OSError:
        pass


def _load_cache() -> dict | None:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _clear_cache() -> None:
    try:
        _CACHE_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Pipeline: upload + analyze
# ---------------------------------------------------------------------------


def run_pipeline(uploaded_file) -> None:
    """Drive /api/upload then /api/analyze, surfacing per-stage timings live."""
    with st.status("Processing PDF...", expanded=True) as status:
        st.write("⏳ Uploading, parsing, chunking, embedding...")
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                timeout=UPLOAD_TIMEOUT,
            )
        except requests.ConnectionError:
            status.update(label="Backend unreachable", state="error")
            _render_backend_down()
            return
        except requests.RequestException as e:
            status.update(label="Upload failed", state="error")
            st.error(f"Upload request failed: {e}")
            return
        wall = time.perf_counter() - t0

        if resp.status_code != 200:
            status.update(label="Upload failed", state="error")
            st.error(_extract_error(resp))
            return

        upload = resp.json()
        st.session_state.upload_id = upload["upload_id"]
        st.session_state.upload_meta = upload
        st.session_state.session_stale = False

        st.write(f"✅ Parsed **{upload['page_count']} pages** — {upload['parse_sec']:.2f}s")
        st.write(f"✅ Chunked into **{upload['chunk_count']} chunks** — {upload['chunk_sec']:.2f}s")
        st.write(f"✅ Embedded with voyage-law-2 — {upload['embed_sec']:.2f}s")
        status.update(label=f"Indexed in {wall:.2f}s", state="complete")

    _run_analyze(upload["upload_id"])


def run_analyze_only(upload_id: str) -> None:
    """Re-analyze an already-uploaded document (no parse/chunk/embed)."""
    _run_analyze(upload_id)


def _run_analyze(upload_id: str) -> None:
    with st.status("Analyzing compliance...", expanded=True) as status:
        st.write("🔍 Retrieving excerpts (5 questions × 5 sub-queries, batched)...")
        st.write("🤖 Grading with Claude Sonnet (5 parallel calls)...")
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/analyze/{upload_id}",
                timeout=ANALYZE_TIMEOUT,
            )
        except requests.ConnectionError:
            status.update(label="Backend unreachable", state="error")
            _render_backend_down()
            return
        except requests.RequestException as e:
            status.update(label="Analysis failed", state="error")
            st.error(f"Analyze request failed: {e}")
            return
        wall = time.perf_counter() - t0

        if resp.status_code == 404:
            status.update(label="Upload session expired", state="error")
            st.warning(
                "The backend no longer has this upload in memory "
                "(uvicorn likely restarted). Re-upload the PDF to continue."
            )
            reset_document_state()
            return
        if resp.status_code != 200:
            status.update(label="Analysis failed", state="error")
            st.error(_extract_error(resp))
            return

        analysis = resp.json()
        st.session_state.analysis = analysis
        st.session_state.session_stale = False
        timings = analysis["metadata"]["timings"]
        st.write(f"✅ Retrieved in {timings['retrieve_sec']:.2f}s")
        st.write(f"✅ Analyzed in {timings['analyze_sec']:.2f}s")
        errors = sum(1 for r in analysis["results"] if r.get("error"))
        label = f"Analysis complete in {wall:.2f}s"
        if errors:
            label += f" ({errors} error{'s' if errors != 1 else ''})"
        status.update(label=label, state="complete")

    _save_cache()
    st.rerun()


def _render_backend_down() -> None:
    st.error(
        f"⚠️ Backend not reachable at `{BACKEND_URL}`.\n\n"
        "Start it from the repo root:\n\n"
        "```\nuvicorn backend.app.main:app --reload --port 8000\n```"
    )


def _extract_error(resp: requests.Response) -> str:
    try:
        return f"{resp.status_code}: {resp.json().get('detail', resp.text)}"
    except ValueError:
        return f"{resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def render_overview(analysis: dict) -> None:
    """Overview tab: scorecard + at-a-glance verdict matrix."""
    meta = analysis["metadata"]
    results = analysis["results"]
    retrieval = meta["retrieval"]

    # --- Aggregate scorecard ----------------------------------------------
    counts = {"Fully Compliant": 0, "Partially Compliant": 0, "Non-Compliant": 0}
    errors = 0
    score_sum = 0.0
    scored = 0
    weight = {"Fully Compliant": 1.0, "Partially Compliant": 0.5, "Non-Compliant": 0.0}
    for r in results:
        if r.get("error"):
            errors += 1
            continue
        s = r.get("compliance_state")
        if s in counts:
            counts[s] += 1
        if s in weight:
            score_sum += weight[s]
            scored += 1
    overall_pct = (score_sum / scored * 100) if scored else 0.0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🟢 Fully", counts["Fully Compliant"])
    s2.metric("🟡 Partially", counts["Partially Compliant"])
    s3.metric("🔴 Non-Compliant", counts["Non-Compliant"])
    s4.metric(
        "Overall",
        f"{overall_pct:.0f}%",
        help="Weighted: Fully=1.0, Partial=0.5, Non=0. Errors excluded.",
    )
    if errors:
        st.warning(f"{errors} question{'s' if errors != 1 else ''} failed to grade.")

    st.markdown("##### Pipeline")
    t = meta["timings"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Pages / Chunks", f"{meta['page_count']} / {meta['chunk_count']}")
    m2.metric("End-to-end", f"{t['total_sec']:.2f}s")
    m3.metric("Analyze (5 parallel)", f"{t['analyze_sec']:.2f}s")

    # --- Verdict matrix (rows jump to per-question tabs) ------------------
    st.markdown("##### Compliance at a glance")
    st.caption("Click a row to jump to its detail tab.")
    for idx, (result, stats) in enumerate(zip(results, retrieval)):
        _render_summary_row(result, stats, tab_idx=idx + 1)


def _render_summary_row(result: dict, stats: dict, tab_idx: int) -> None:
    qid = stats["question_id"]
    title = QUESTION_TITLES.get(qid, qid.replace("_", " ").title())

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([4, 3, 3, 1], vertical_alignment="center")
        if result.get("error"):
            c1.markdown(f"**⚫ &nbsp; {title}**")
            c2.error("Failed")
            c3.caption(result["error"][:60])
        else:
            state = result["compliance_state"]
            style = STATE_STYLE.get(state, {"emoji": "⚪", "color": "#6b7280"})
            confidence = result["confidence"]
            c1.markdown(f"**{style['emoji']} &nbsp; {title}**")
            c2.markdown(
                f"<span style='color:{style['color']};font-weight:600'>{state}</span>",
                unsafe_allow_html=True,
            )
            c3.progress(
                min(max(confidence / 100, 0.0), 1.0),
                text=f"Confidence {confidence:.0f}%",
            )
        # Tab-jump trigger. Actual switch is done via a post-render JS click
        # in main() since st.tabs has no programmatic selection API.
        if c4.button("→", key=f"jump_{qid}", help="Open detail tab"):
            st.session_state.jump_to_tab = tab_idx
            st.rerun()


def _render_quotes(raw: str) -> None:
    """Render the analyzer's quote blob as blockquotes + citation chips."""
    if not raw or not raw.strip():
        st.caption("_No quotes returned._")
        return

    text = raw.strip()
    if text.startswith(_NO_EVIDENCE_PREFIX):
        st.info(text)
        return

    # Quotes are emitted one per line by the prompt. Keep the original
    # line as a fallback blockquote if the citation format drifts.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        m = _QUOTE_RE.match(line)
        if m:
            citation, quote = m.group(1).strip(), m.group(2).strip()
            st.markdown(f"> {quote}")
            st.caption(f"— {citation}")
        else:
            st.markdown(f"> {line}")


def render_verdict_detail(result: dict, stats: dict) -> None:
    """Full-width per-question view shown inside its own tab."""
    qid = stats["question_id"]
    title = QUESTION_TITLES.get(qid, qid.replace("_", " ").title())

    if result.get("error"):
        st.markdown(f"## ⚫ {title}")
        st.error(f"Analysis failed: {result['error']}")
        return

    state = result["compliance_state"]
    style = STATE_STYLE.get(state, {"emoji": "⚪", "color": "#6b7280"})
    confidence = result["confidence"]

    st.markdown(f"## {style['emoji']} {title}")
    st.markdown(
        f"<span style='color:{style['color']};font-weight:700;font-size:1.1rem'>{state}</span>"
        f" &nbsp;·&nbsp; confidence **{confidence:.0f}%**",
        unsafe_allow_html=True,
    )

    st.divider()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Rationale")
        st.write(result["rationale"])

        st.markdown("##### Retrieval")
        rcols = st.columns(3)
        rcols[0].metric("Chunks", stats["chunks_used"])
        rcols[1].metric("Context tokens", stats["context_tokens"])
        top = stats.get("top_score")
        rcols[2].metric("Top score", f"{top:.3f}" if top is not None else "—")

    with right:
        st.markdown("#### Relevant quotes")
        _render_quotes(result["relevant_quotes"])


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def render_chat(upload_id: str) -> None:
    # Streamlit's st.container(height=N) only accepts fixed pixels. Override
    # the 720px sentinel below with a viewport-relative height so the chat
    # panel grows with the window. Keep the CSS selector and the height=
    # literal in sync.
    st.markdown(
        """
        <style>
        div[style*="height: 720px;"] {
            height: calc(100vh - 220px) !important;
            max-height: calc(100vh - 220px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Grounded Q&A against the same in-memory index the verdicts used. "
        "Each turn re-retrieves based on your latest message."
    )

    # Order matters: declare the scrollable messages panel first, then the
    # input, so Streamlit lays them out as panel-above-input. All message
    # writes (history + the streaming turn) happen inside `messages_area`
    # so a new turn can't shove the input around the page.
    messages_area = st.container(height=720, border=False)
    prompt = st.chat_input("Ask about clauses, obligations, exhibits...")

    with messages_area:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    _render_sources(msg["sources"])

        if not prompt:
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Plain-text history only (no injected context). Exclude the turn we
        # just appended — the backend takes it as `message` and fuses
        # retrieved excerpts in.
        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            sources: list[dict] = []
            full_text = ""
            errored = False
            try:
                with requests.post(
                    f"{BACKEND_URL}/api/chat/{upload_id}",
                    json={"message": prompt, "history": history_payload},
                    stream=True,
                    timeout=CHAT_TIMEOUT,
                ) as resp:
                    if resp.status_code != 200:
                        st.error(_extract_error(resp))
                        errored = True
                    else:
                        for event in _iter_ndjson(resp.iter_lines(decode_unicode=True)):
                            etype = event.get("type")
                            if etype == "sources":
                                sources = event.get("sources", [])
                            elif etype == "delta":
                                full_text += event.get("text", "")
                                placeholder.markdown(full_text + "▌")
                            elif etype == "error":
                                st.error(event.get("error", "Unknown chat error"))
                                errored = True
                                break
                            elif etype == "done":
                                break
            except requests.RequestException as e:
                st.error(f"Chat request failed: {e}")
                errored = True

            if errored and not full_text:
                # Drop the user turn so retry isn't double-counted, then
                # rerun so the dropped turn disappears from the panel.
                st.session_state.chat_history.pop()
                st.rerun()

            placeholder.markdown(full_text)
            if sources:
                _render_sources(sources)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_text,
            "sources": sources,
        })


def _iter_ndjson(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Ignore stray keep-alive / partial lines.
            continue


def _render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for s in sources:
            header = s.get("section_header") or "(preamble)"
            pages = ", ".join(f"p.{p}" for p in s.get("page_numbers", []))
            st.markdown(
                f"- **{header}** · {pages} · score `{s['score']:.3f}`"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[Any, bool]:
    """Owns the uploader, run button, and post-analysis pipeline summary."""
    with st.sidebar:
        st.markdown("### 📑 Contract Compliance")
        st.caption("voyage-law-2 RAG · Claude Sonnet 4.6")

        uploaded_file = st.file_uploader(
            "Contract PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )

        if (
            uploaded_file is not None
            and st.session_state.current_file_id != uploaded_file.file_id
        ):
            st.session_state.current_file_id = uploaded_file.file_id
            reset_document_state()

        has_analysis = st.session_state.analysis is not None
        same_file = (
            uploaded_file is not None
            and st.session_state.upload_id is not None
            and st.session_state.current_file_id == uploaded_file.file_id
            and not st.session_state.session_stale
        )

        # Three button modes:
        #   - Fresh or new file:  "Analyze contract"  (full pipeline)
        #   - Re-run on same file: "Re-analyze"       (skip upload+embed)
        # Stale-rehydrated sessions fall into the full pipeline path
        # because the backend no longer holds the index.
        btn_label = "Re-analyze" if same_file and has_analysis else "Analyze contract"
        run_btn = st.button(
            btn_label,
            type="primary",
            disabled=(uploaded_file is None),
            use_container_width=True,
            help=(
                "Re-runs retrieval + grading on the existing in-memory index."
                if same_file and has_analysis
                else None
            ),
        )

        if has_analysis:
            analysis = st.session_state.analysis
            meta = analysis["metadata"]
            t = meta["timings"]
            st.divider()
            st.markdown("**Document**")
            st.caption(f"📄 {meta['filename']}")
            st.caption(f"{meta['page_count']} pages · {meta['chunk_count']} chunks")

            st.markdown("**Pipeline timings**")
            st.caption(
                f"Parse `{t['parse_sec']:.1f}s` · "
                f"Chunk `{t['chunk_sec']:.1f}s` · "
                f"Embed `{t['embed_sec']:.1f}s`"
            )
            st.caption(
                f"Retrieve `{t['retrieve_sec']:.1f}s` · "
                f"Analyze `{t['analyze_sec']:.1f}s`"
            )
            st.caption(f"**Total `{t['total_sec']:.1f}s`**")

            st.download_button(
                "⬇️ Download results (JSON)",
                data=json.dumps(analysis, indent=2).encode(),
                file_name=f"{Path(meta['filename']).stem}_compliance.json",
                mime="application/json",
                use_container_width=True,
            )

            if st.button(
                "🗑️ Clear analysis",
                use_container_width=True,
                help="Drop this session and return to the empty state.",
            ):
                _clear_analysis(st.session_state.upload_id)
                st.rerun()

            with st.expander("Models"):
                st.code(
                    f"analysis:  {meta['models']['analysis']}\n"
                    f"embedding: {meta['models']['embedding']}\n"
                    f"vision:    {meta['models']['vision']}",
                    language=None,
                )

    return uploaded_file, run_btn, same_file


def render_empty_state() -> None:
    st.title("Contract Compliance Analyzer")
    st.write(
        "Upload a vendor security contract PDF from the sidebar. The pipeline "
        "parses, chunks, embeds with voyage-law-2, retrieves per-question "
        "context, and grades 5 security compliance requirements with Claude "
        "Sonnet."
    )
    st.markdown("##### Requirements graded")
    cols = st.columns(len(QUESTION_TITLES))
    for col, title in zip(cols, QUESTION_TITLES.values()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")


def main() -> None:
    st.set_page_config(
        page_title="Contract Compliance Analyzer",
        page_icon="📑",
        layout="wide",
    )
    init_state()

    # Global CSS: stretch st.tabs across the full content width. Each tab
    # gets equal flex so Overview / 5 questions / Chat share the strip
    # evenly regardless of label length.
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] div[role="tablist"] {
            display: flex;
            width: 100%;
            gap: 0;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            flex: 1 1 0;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file, run_btn, same_file = render_sidebar()

    if run_btn and uploaded_file is not None:
        if same_file:
            run_analyze_only(st.session_state.upload_id)
        else:
            reset_document_state()
            st.session_state.current_file_id = uploaded_file.file_id
            run_pipeline(uploaded_file)

    if st.session_state.analysis is None:
        render_empty_state()
        return

    if st.session_state.session_stale:
        st.info(
            "Showing the last analysis rehydrated from cache. "
            "The backend's in-memory index is gone — re-upload to enable chat."
        )

    analysis = st.session_state.analysis
    results = analysis["results"]
    retrieval = analysis["metadata"]["retrieval"]

    # Tab labels carry the verdict emoji so you can scan compliance state
    # across the five questions from the tab strip alone.
    tab_labels = ["📊 Overview"]
    for result, stats in zip(results, retrieval):
        title = QUESTION_TITLES.get(
            stats["question_id"], stats["question_id"].replace("_", " ").title()
        )
        if result.get("error"):
            emoji = "⚫"
        else:
            emoji = STATE_STYLE.get(
                result.get("compliance_state", ""), {"emoji": "⚪"}
            )["emoji"]
        tab_labels.append(f"{emoji} {title}")
    tab_labels.append("💬 Chat")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview(analysis)
    for i, (result, stats) in enumerate(zip(results, retrieval)):
        with tabs[i + 1]:
            render_verdict_detail(result, stats)
    with tabs[-1]:
        render_chat(st.session_state.upload_id)

    # Tab-jump: consume the flag set by overview row buttons and simulate
    # a click on the matching tab button in the DOM. st.tabs has no
    # programmatic selection API, so this JS-nudge is the pragmatic path.
    if (jump := st.session_state.jump_to_tab) is not None:
        st.session_state.jump_to_tab = None
        st.components.v1.html(
            f"""
            <script>
              const tabs = window.parent.document.querySelectorAll(
                'div[data-testid="stTabs"] button[role="tab"]'
              );
              if (tabs[{jump}]) tabs[{jump}].click();
            </script>
            """,
            height=0,
        )


if __name__ == "__main__":
    main()
