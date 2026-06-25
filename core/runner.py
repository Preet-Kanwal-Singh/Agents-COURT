"""
Sequential council runner — Phase 2/3/4/5/6.

Phase 2: run_sequential
Phase 3: run_review_round, run_aggregated_review
Phase 4: run_full_pipeline (non-streaming, CLI entry point)
Phase 5: run_full_pipeline updated — classification parameter removed, router
         called internally via classify_query.
Phase 6: stream_full_pipeline — same pipeline logic as run_full_pipeline
         but yields SSE-ready event dicts and streams the Archivist output
         token-by-token. Sole entry point for the FastAPI layer.

Phase 7 note: swap run_sequential for asyncio.gather here without touching
anything upstream.
"""
from __future__ import annotations

from statistics import mean
from typing import Any, AsyncGenerator

from .models import Role, RoleResponse
from .ollama_client import call_role
from .prompt_loader import get_system_prompt
from .synthesis import (
    build_context_packet,
    run_archivist,
    stream_archivist,
    strip_latex,
    format_synthesis_output,
)
from .review import (
    anonymize, build_review_prompt, extract_json, validate_review,
    normalize_scores, flag_quality_issues, deanonymize_scores,
    ReviewResult, AggregatedReview, ReviewParseError,
)

COUNCIL_ROLES: list[Role] = [Role.VIRGO, Role.PISCES, Role.CRITIC, Role.SEER]


async def run_sequential(
    query: str,
    roles: list[Role] | None = None,
) -> list[RoleResponse]:
    """
    Run roles sequentially. Returns all responses including errors.
    Never raises — failed roles are captured with .error set.
    """
    if roles is None:
        roles = COUNCIL_ROLES

    results: list[RoleResponse] = []
    for role in roles:
        system_prompt = get_system_prompt(role.value)
        response = await call_role(role, system_prompt, query)
        results.append(response)
    return results


async def run_review_round(
    query: str,
    council_responses: list[RoleResponse],
    classification: str = "unclassified",
) -> ReviewResult:
    """
    Single Reviewer pass. Returns ReviewResult with .ok=False on failure.
    Never raises.
    """
    _EMPTY = ReviewResult(
        strongest=None, strongest_reason="", weakest_blind_spot=None,
        blind_spot_detail="", collective_miss="", role_scores={},
        quality_flags=[], was_repaired=False,
    )

    try:
        anon = anonymize(council_responses)
    except ValueError as e:
        result = _EMPTY
        result.error = f"Anonymization failed: {e}"
        return result

    user_prompt = build_review_prompt(
        query=query,
        classification=classification,
        content_by_letter=anon.content_by_letter,
    )
    system_prompt = get_system_prompt("reviewer")
    response = await call_role(Role.REVIEWER, system_prompt, user_prompt)

    if not response.ok:
        result = _EMPTY
        result.error = f"Reviewer call failed: {response.error}"
        return result

    try:
        parsed, was_repaired = extract_json(response.content)
    except ReviewParseError as e:
        result = _EMPTY
        result.error = f"JSON extraction failed: {e}"
        return result

    problems = validate_review(parsed)
    if problems:
        result = _EMPTY
        result.error = f"Validation failed: {'; '.join(problems)}"
        return result

    raw_scores = {l: float(parsed["usefulness_scores"][l]) for l in ("A", "B", "C", "D")}
    normalized = normalize_scores(raw_scores)
    role_scores = deanonymize_scores(normalized, anon.anonymization_map)

    strongest_letter = parsed.get("strongest")
    weakest_letter = parsed.get("weakest_blind_spot")

    return ReviewResult(
        strongest=anon.anonymization_map[strongest_letter].value if strongest_letter else None,
        strongest_reason=parsed.get("strongest_reason", ""),
        weakest_blind_spot=anon.anonymization_map[weakest_letter].value if weakest_letter else None,
        blind_spot_detail=parsed.get("blind_spot_detail", ""),
        collective_miss=parsed.get("collective_miss", ""),
        role_scores=role_scores,
        quality_flags=flag_quality_issues(parsed),
        was_repaired=was_repaired,
    )


async def run_aggregated_review(
    query: str,
    council_responses: list[RoleResponse],
    num_reviewers: int = 4,
    classification: str = "unclassified",
) -> AggregatedReview:
    """
    Run num_reviewers independent Reviewer passes and aggregate results.
    Each pass uses a fresh anonymization shuffle to prevent positional bias.
    Failed passes are excluded from aggregation but counted in error_count.
    """
    role_names = [r.value for r in COUNCIL_ROLES]

    reviews: list[ReviewResult] = []
    for _ in range(num_reviewers):
        result = await run_review_round(query, council_responses, classification)
        reviews.append(result)

    ok_reviews = [r for r in reviews if r.ok]

    all_scores: dict[str, list[float]] = {role: [] for role in role_names}
    for review in ok_reviews:
        for role, score in review.role_scores.items():
            all_scores[role].append(score)

    review_votes = {
        role: mean(scores) if scores else 0.0
        for role, scores in all_scores.items()
    }

    strongest_counts: dict[str, int] = {role: 0 for role in role_names}
    blind_spot_counts: dict[str, int] = {role: 0 for role in role_names}
    for review in ok_reviews:
        if review.strongest and review.strongest in strongest_counts:
            strongest_counts[review.strongest] += 1
        if review.weakest_blind_spot and review.weakest_blind_spot in blind_spot_counts:
            blind_spot_counts[review.weakest_blind_spot] += 1

    return AggregatedReview(
        reviews=reviews,
        review_votes=review_votes,
        strongest_counts=strongest_counts,
        blind_spot_counts=blind_spot_counts,
        blind_spots=[r.blind_spot_detail for r in ok_reviews],
        collective_misses=[r.collective_miss for r in ok_reviews],
        quality_flag_count=sum(1 for r in ok_reviews if r.quality_flags),
        repair_count=sum(1 for r in ok_reviews if r.was_repaired),
        error_count=sum(1 for r in reviews if not r.ok),
    )


async def run_full_pipeline(
    query: str,
    num_reviewers: int = 4,
) -> Any:
    """
    Full pipeline: router -> sequential council -> aggregated review -> Archivist synthesis.
    Returns SynthesisResult. CLI entry point — uses non-streaming Archivist call.

    Phase 5: router is called internally; classification parameter removed.
    """
    from .router import run_router

    router_result = await run_router(query)

    role_responses = await run_sequential(query)

    aggregated_review = await run_aggregated_review(
        query=query,
        council_responses=role_responses,
        num_reviewers=num_reviewers,
        classification=router_result.classification,
    )

    role_outputs = {
        r.role.value.lower(): r.content if r.ok else f"[{r.role.value} failed: {r.error}]"
        for r in role_responses
    }

    packet = build_context_packet(
        query=query,
        role_outputs=role_outputs,
        review_votes=aggregated_review.review_votes,
        blind_spots=aggregated_review.blind_spots,
        collective_misses=aggregated_review.collective_misses,
        classification=router_result.classification,
        initial_weights=router_result.initial_weights,
        modifiers=router_result.modifiers,
    )

    return await run_archivist(packet)


async def stream_full_pipeline(
    query: str,
    num_reviewers: int = 4,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Full pipeline with streaming Archivist output. Phase 6 API entry point.

    Yields event dicts. The FastAPI handler wraps these in SSE formatting —
    no pipeline logic lives in api.py.

    Event types:
      {"type": "status",  "message": str}
        — emitted before and after each phase; keeps long-running requests
          visible to the client during council/review inference.
      {"type": "router",  "data": dict}
        — router classification + weights + modifiers; emitted once, after
          router runs, before council. Clients that don't need metadata can
          filter on type.
      {"type": "token",   "content": str}
        — one per Archivist generation token; raw (not LaTeX-stripped).
      {"type": "done",    "synthesis": str}
        — terminal success event; synthesis is the fully assembled and
          LaTeX-stripped Archivist response.
      {"type": "error",   "message": str}
        — terminal failure event; emitted and then the generator returns.
    """
    from .router import run_router

    # ------------------------------------------------------------------ #
    # Router                                                               #
    # ------------------------------------------------------------------ #
    yield {"type": "status", "message": "Classifying query..."}

    try:
        router_result = await run_router(query)
    except Exception as exc:
        yield {"type": "error", "message": f"Router failed: {type(exc).__name__}: {exc}"}
        return

    yield {
        "type": "router",
        "data": {
            "classification": router_result.classification,
            "stage":          router_result.stage,
            "initial_weights": router_result.initial_weights,
            "modifiers":      router_result.modifiers,
            "reasoning":      router_result.reasoning,
        },
    }

    # ------------------------------------------------------------------ #
    # Council roles                                                        #
    # ------------------------------------------------------------------ #
    yield {"type": "status", "message": "Running council roles..."}
    role_responses = await run_sequential(query)

    ok_count = sum(1 for r in role_responses if r.ok)
    failed = [r.role.value for r in role_responses if not r.ok]
    status_msg = f"Council complete ({ok_count}/4 roles succeeded)"
    if failed:
        status_msg += f" — failed: {', '.join(failed)}"
    yield {"type": "status", "message": status_msg}

    # ------------------------------------------------------------------ #
    # Review round                                                         #
    # ------------------------------------------------------------------ #
    yield {"type": "status", "message": "Running review round..."}
    aggregated_review = await run_aggregated_review(
        query=query,
        council_responses=role_responses,
        num_reviewers=num_reviewers,
        classification=router_result.classification,
    )

    review_passes = 4 - aggregated_review.error_count
    review_msg = f"Review complete ({review_passes}/4 passes)"
    if aggregated_review.quality_flag_count:
        review_msg += f", {aggregated_review.quality_flag_count} quality flags"
    if aggregated_review.repair_count:
        review_msg += f", {aggregated_review.repair_count} JSON repairs"
    yield {"type": "status", "message": review_msg}

    # ------------------------------------------------------------------ #
    # Context packet                                                       #
    # ------------------------------------------------------------------ #
    role_outputs = {
        r.role.value.lower(): r.content if r.ok else f"[{r.role.value} failed: {r.error}]"
        for r in role_responses
    }

    packet = build_context_packet(
        query=query,
        role_outputs=role_outputs,
        review_votes=aggregated_review.review_votes,
        blind_spots=aggregated_review.blind_spots,
        collective_misses=aggregated_review.collective_misses,
        classification=router_result.classification,
        initial_weights=router_result.initial_weights,
        modifiers=router_result.modifiers,
    )

    # ------------------------------------------------------------------ #
    # Archivist synthesis — streaming                                      #
    # ------------------------------------------------------------------ #
    yield {"type": "status", "message": "Archivist synthesizing..."}

    accumulated: list[str] = []
    async for token in stream_archivist(packet):
        accumulated.append(token)
        yield {"type": "token", "content": token}

    full_text = "".join(accumulated)

    if not full_text.strip():
        yield {"type": "error", "message": "Archivist returned empty output — check model load and timeout"}
        return

    yield {"type": "done", "synthesis": strip_latex(full_text)}