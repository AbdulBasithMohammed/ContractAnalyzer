# Prompt templates + tool schema for the Stage 5 analyzer.
# The system prompt anchors the rubric; the tool schema forces structured output.

from .retriever import ComplianceQuestion
from .schemas import RetrievedChunk


NO_EVIDENCE_SENTINEL = "No relevant provisions found in the contract."


COMPLIANCE_SYSTEM_PROMPT = f"""You are a senior security and compliance analyst reviewing a vendor security contract (an information-security addendum to a master services agreement). Your job is to grade a single compliance requirement against the contract language provided.

## Evidence rules
- Base your verdict ONLY on the provided contract excerpts. Do not rely on outside knowledge, general industry practice, or assumptions about what is "standard."
- If the contract is silent on a sub-requirement, treat it as not addressed. Silence is not compliance.
- Do not fabricate, paraphrase, or summarize quotes. Quotes must be verbatim substrings of the provided excerpts.

## Compliance states
- **Fully Compliant** — every sub-requirement in the question is explicitly addressed by the contract with clear, unambiguous language.
- **Partially Compliant** — some sub-requirements are addressed explicitly; others are silent, ambiguous, or only inferable from adjacent clauses.
- **Non-Compliant** — the contract is silent on the requirement as a whole, or its language directly contradicts it.

## Confidence scale (0–100)
- **85–100** — all sub-requirements explicitly covered (or explicitly absent); language is clear.
- **60–84** — most sub-requirements covered; one or two inferred from adjacent clauses or partially stated.
- **30–59** — minimal coverage; significant gaps; heavy inference required.
- **0–29** — silence or direct contradiction; Non-Compliant with low ambiguity.

## Quote requirements
- Provide verbatim excerpts from the contract that support your verdict, each on its own line.
- Prefix each quote with its section header and page numbers as shown in the context, e.g. `[§6.6 Password Management Standard | p.7]: "<verbatim text>"`.
- If the verdict is Non-Compliant because the contract is silent on the requirement and no supporting or contradicting text exists, set `relevant_quotes` to exactly: `{NO_EVIDENCE_SENTINEL}`
- Do NOT invent quotes. If you cannot find a verbatim excerpt to support a partial verdict, reconsider the verdict.

## Rationale
- 2–4 sentences. Tie the verdict to the quoted evidence and call out which sub-requirements are covered vs. missing.

## Response format
Respond by invoking the `submit_compliance_verdict` tool. Do not emit any prose outside the tool call."""


# Tool schema for forced structured output. Mirrors ComplianceResult minus
# `compliance_question` — that field is filled server-side from the question
# object, so no reason to make the model echo it.
COMPLIANCE_TOOL = {
    "name": "submit_compliance_verdict",
    "description": (
        "Submit the final compliance verdict for the question under review. "
        "Must be called exactly once per question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "compliance_state": {
                "type": "string",
                "enum": ["Fully Compliant", "Partially Compliant", "Non-Compliant"],
                "description": "The compliance verdict per the rubric.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Confidence in the verdict, per the 0–100 scale.",
            },
            "relevant_quotes": {
                "type": "string",
                "description": (
                    "Verbatim contract excerpts supporting the verdict, each "
                    "prefixed with its section and page. Use the "
                    f"'{NO_EVIDENCE_SENTINEL}' sentinel when the contract is "
                    "silent and Non-Compliant."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "2–4 sentence explanation tying the verdict to the quoted "
                    "evidence; call out covered vs. missing sub-requirements."
                ),
            },
        },
        "required": [
            "compliance_state",
            "confidence",
            "relevant_quotes",
            "rationale",
        ],
    },
}


def _format_chunk(chunk: RetrievedChunk, index: int) -> str:
    header = chunk.section_header or "(preamble)"
    pages = ", ".join(f"p.{p}" for p in chunk.page_numbers)
    return f"[Chunk {index} | Section: {header} | Pages: {pages}]\n{chunk.text}"


def build_user_message(
    question: ComplianceQuestion,
    chunks: list[RetrievedChunk],
) -> str:
    """Assemble the user-turn message: context excerpts followed by the question."""
    if chunks:
        context = "\n\n---\n\n".join(
            _format_chunk(c, i + 1) for i, c in enumerate(chunks)
        )
    else:
        context = "(no contract excerpts retrieved for this question)"

    return (
        "## Contract excerpts\n"
        "The following excerpts were retrieved from the contract as the "
        "passages most relevant to the question. They are presented in "
        "document order.\n\n"
        f"{context}\n\n"
        "## Compliance question\n"
        f"{question.question}\n\n"
        "Respond via the `submit_compliance_verdict` tool."
    )
