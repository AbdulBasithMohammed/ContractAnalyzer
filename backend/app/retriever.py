# Stage 4: Retrieval
# Multi-query expansion over the per-upload Qdrant index. For each of the
# 5 fixed compliance questions, every hand-written sub-query is run, hits
# are deduped (max score wins), score-ranked under a token budget, and
# re-sorted into document order for stable analyzer input.

from __future__ import annotations

import logging
from dataclasses import dataclass

import tiktoken

from .config import settings
from .embedder import EmbeddingIndex
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplianceQuestion:
    id: str
    title: str
    question: str
    sub_queries: tuple[str, ...]


# Full requirement text is copied verbatim from the Contract Analyzer
# Assignment (Table 1) so the analyzer sees exactly what the rubric asks.
COMPLIANCE_QUESTIONS: tuple[ComplianceQuestion, ...] = (
    ComplianceQuestion(
        id="password_management",
        title="Password Management",
        question=(
            "Password Management. The contract must require a documented "
            "password standard covering password length/strength, prohibition "
            "of default and known-compromised passwords, secure storage (no "
            "plaintext; salted hashing if stored), brute-force protections "
            "(lockout/rate limiting), prohibition on password sharing, "
            "vaulting of privileged credentials/recovery codes, and time-based "
            "rotation for break-glass credentials. Based on the contract "
            "language and exhibits, what is the compliance state for Password "
            "Management?"
        ),
        sub_queries=(
            "password length complexity and strength requirements",
            "prohibition of default or known-compromised passwords",
            "secure password storage salted hashing no plaintext",
            "account lockout and brute-force rate limiting protections",
            "vaulting of privileged credentials and break-glass rotation",
        ),
    ),
    ComplianceQuestion(
        id="it_asset_management",
        title="IT Asset Management",
        question=(
            "IT Asset Management. The contract must require an in-scope asset "
            "inventory (including cloud accounts/subscriptions, workloads, "
            "databases, security tooling), define minimum inventory fields, "
            "require at least quarterly reconciliation/review, and require "
            "secure configuration baselines with drift remediation and "
            "prohibition of insecure defaults. Based on the contract language "
            "and exhibits, what is the compliance state for IT Asset "
            "Management?"
        ),
        sub_queries=(
            "asset inventory of cloud accounts workloads and databases",
            "minimum inventory fields and asset ownership attributes",
            "quarterly asset inventory reconciliation and review",
            "secure configuration baselines and hardening standards",
            "configuration drift remediation and prohibition of insecure defaults",
        ),
    ),
    ComplianceQuestion(
        id="security_training_background_checks",
        title="Security Training & Background Checks",
        question=(
            "Security Training & Background Checks. The contract must require "
            "security awareness training on hire and at least annually, and "
            "background screening for personnel with access to Company Data "
            "to the extent permitted by law, including maintaining a "
            "screening policy and attestation/evidence. Based on the contract "
            "language and exhibits, what is the compliance state for Security "
            "Training and Background Checks?"
        ),
        sub_queries=(
            "security awareness training on hire and annually",
            "background screening checks for personnel with data access",
            "screening policy and personnel attestation evidence",
            "training completion records and evidence of delivery",
            "employee onboarding security obligations and acknowledgements",
        ),
    ),
    ComplianceQuestion(
        id="data_in_transit_encryption",
        title="Data in Transit Encryption",
        question=(
            "Data in Transit Encryption. The contract must require encryption "
            "of Company Data in transit using TLS 1.2+ (preferably TLS 1.3 "
            "where feasible) for Company-to-Service traffic, administrative "
            "access pathways, and applicable Service-to-Subprocessor "
            "transfers, with certificate management and avoidance of insecure "
            "cipher suites. Based on the contract language and exhibits, what "
            "is the compliance state for Data in Transit Encryption?"
        ),
        sub_queries=(
            "encryption of data in transit using TLS 1.2 or TLS 1.3",
            "encrypted administrative access pathways and remote admin",
            "subprocessor transfers encrypted in transit",
            "certificate management and key rotation",
            "prohibition of insecure cipher suites and deprecated protocols",
        ),
    ),
    ComplianceQuestion(
        id="network_authn_authz",
        title="Network Authentication & Authorization Protocols",
        question=(
            "Network Authentication & Authorization Protocols. The contract "
            "must specify the authentication mechanisms (e.g., SAML SSO for "
            "users, OAuth/token-based for APIs), require MFA for privileged/"
            "production access, require secure admin pathways (bastion/secure "
            "gateway) with session logging, and require RBAC authorization. "
            "Based on the contract language and exhibits, what is the "
            "compliance state for Network Authentication and Authorization "
            "Protocols?"
        ),
        sub_queries=(
            "SAML SSO or federated identity for user authentication",
            "OAuth or token-based authentication for APIs",
            "multi-factor authentication for privileged or production access",
            "bastion host or secure admin gateway with session logging",
            "role-based access control RBAC authorization model",
        ),
    ),
)


# cl100k_base matches the chunker's tokenizer — keeps budget math consistent.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_tokenizer.encode(text, disallowed_special=()))


def _consolidate(
    per_sub_query_hits: list[list[RetrievedChunk]],
    budget: int,
    question_id: str,
) -> list[RetrievedChunk]:
    """Union hits across sub-queries, keep max score per chunk, pack under
    token budget by relevance, then present in document order.

    Rank-then-doc-order gives the analyzer the most relevant chunks while
    preserving reading order — matters for contracts where later sections
    amend earlier ones.
    """
    best: dict[int, RetrievedChunk] = {}
    for hits in per_sub_query_hits:
        for hit in hits:
            prior = best.get(hit.chunk_index)
            if prior is None or hit.score > prior.score:
                best[hit.chunk_index] = hit

    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    selected: list[RetrievedChunk] = []
    used = 0
    for chunk in ranked:
        cost = _token_len(chunk.text)
        # Always accept the top-ranked chunk even if it alone exceeds budget —
        # dropping it would leave the analyzer with nothing.
        if selected and used + cost > budget:
            continue
        selected.append(chunk)
        used += cost

    selected.sort(key=lambda c: c.chunk_index)
    logger.info(
        "retrieve[%s]: %d sub-queries -> %d unique hits -> %d kept (%d tokens)",
        question_id, len(per_sub_query_hits), len(best), len(selected), used,
    )
    return selected


def retrieve_for_question(
    index: EmbeddingIndex,
    question: ComplianceQuestion,
    top_k: int | None = None,
    max_context_tokens: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve context for a single compliance question. Embeds all
    sub-queries in one Voyage call via search_batch.
    """
    top_k = top_k or settings.retrieval_top_k
    budget = max_context_tokens or settings.retrieval_context_tokens
    hits = index.search_batch(list(question.sub_queries), top_k=top_k)
    return _consolidate(hits, budget, question.id)


def retrieve_all(
    index: EmbeddingIndex,
    top_k: int | None = None,
    max_context_tokens: int | None = None,
) -> dict[str, list[RetrievedChunk]]:
    """Retrieve context for all 5 compliance questions in ONE Voyage call.

    Flattens every sub-query across every question into a single batched
    embed, then slices the per-sub-query hit lists back out per question.
    Cuts Voyage round-trips from N_questions * N_sub_queries down to 1.
    """
    top_k = top_k or settings.retrieval_top_k
    budget = max_context_tokens or settings.retrieval_context_tokens

    flat: list[str] = []
    offsets: list[tuple[ComplianceQuestion, int, int]] = []
    for q in COMPLIANCE_QUESTIONS:
        start = len(flat)
        flat.extend(q.sub_queries)
        offsets.append((q, start, len(flat)))

    all_hits = index.search_batch(flat, top_k=top_k)

    return {
        q.id: _consolidate(all_hits[start:end], budget, q.id)
        for q, start, end in offsets
    }
