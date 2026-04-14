# Stage 5: LLM Analysis
# One async Claude call per compliance question, fanned out via asyncio.gather.
# Structured output is enforced with a forced tool_use; the model must respond
# via the `submit_compliance_verdict` tool, which mirrors ComplianceResult.

from __future__ import annotations

import asyncio
import logging

import anthropic

from .config import settings
from .prompts import COMPLIANCE_SYSTEM_PROMPT, COMPLIANCE_TOOL, build_user_message
from .retriever import COMPLIANCE_QUESTIONS, ComplianceQuestion
from .schemas import ComplianceResult, ComplianceState, RetrievedChunk

logger = logging.getLogger(__name__)

# Process-wide async client — created on first use.
_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def analyze_question(
    question: ComplianceQuestion,
    chunks: list[RetrievedChunk],
) -> ComplianceResult:
    """Grade one compliance question against its retrieved context.

    Uses forced tool_use so the model must respond via the structured-output
    tool. The tool input is validated into a ComplianceResult; the
    `compliance_question` field is filled from the question's verbatim text.
    """
    client = _get_client()
    user_message = build_user_message(question, chunks)

    tool_input = await _call_with_retry(
        client=client,
        system=COMPLIANCE_SYSTEM_PROMPT,
        user_message=user_message,
        question_id=question.id,
    )

    result = ComplianceResult(
        compliance_question=question.question,
        compliance_state=ComplianceState(tool_input["compliance_state"]),
        confidence=float(tool_input["confidence"]),
        relevant_quotes=tool_input["relevant_quotes"],
        rationale=tool_input["rationale"],
    )
    logger.info(
        "analyze[%s]: state=%s confidence=%.1f",
        question.id, result.compliance_state.value, result.confidence,
    )
    return result


async def analyze_all(
    retrieved: dict[str, list[RetrievedChunk]],
) -> list[ComplianceResult]:
    """Analyze all 5 compliance questions in parallel.

    Results are returned in COMPLIANCE_QUESTIONS order, not dict-iteration
    order, so the downstream response is deterministic.
    """
    tasks = [
        analyze_question(q, retrieved.get(q.id, []))
        for q in COMPLIANCE_QUESTIONS
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_with_retry(
    *,
    client: anthropic.AsyncAnthropic,
    system: str,
    user_message: str,
    question_id: str,
) -> dict:
    """Call Sonnet with forced tool_use; retry on transient errors."""
    max_retries = settings.analysis_max_retries
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=settings.analysis_model,
                max_tokens=settings.analysis_max_tokens,
                system=system,
                tools=[COMPLIANCE_TOOL],
                tool_choice={"type": "tool", "name": COMPLIANCE_TOOL["name"]},
                messages=[{"role": "user", "content": user_message}],
            )
            return _extract_tool_input(response, question_id)
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as e:
            if attempt == max_retries - 1:
                logger.error("analyze[%s]: retries exhausted: %s", question_id, e)
                raise
            wait = 2.0 * (2 ** attempt)
            logger.warning(
                "analyze[%s]: transient error (attempt %d/%d); sleeping %.1fs",
                question_id, attempt + 1, max_retries, wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")  # defensive


def _extract_tool_input(response: anthropic.types.Message, question_id: str) -> dict:
    """Pull the forced tool_use block's input out of the response."""
    for block in response.content:
        if block.type == "tool_use" and block.name == COMPLIANCE_TOOL["name"]:
            return block.input
    raise RuntimeError(
        f"analyze[{question_id}]: model response missing tool_use "
        f"(stop_reason={response.stop_reason})"
    )
