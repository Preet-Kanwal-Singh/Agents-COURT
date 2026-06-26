"""
Sequential council runner — Phase 2/3/4/5/6/7.

Phase 2: run_sequential
Phase 3: run_review_round, run_aggregated_review
Phase 4: run_full_pipeline (non-streaming, CLI entry point)
Phase 5: run_full_pipeline updated — router called internally.
Phase 6: stream_full_pipeline — same pipeline logic as run_full_pipeline
         but yields SSE-ready event dicts and streams the Archivist output
         token-by-token. Sole entry point for the FastAPI layer.
Phase 7: run_parallel (model-aware asyncio.gather), check_parallel_viable (RAM check).
         run_full_pipeline and stream_full_pipeline both accept parallel=True
         (default); fall back to sequential automatically if RAM check fails.
         run_parallel groups roles by model: same-model roles run sequentially
         within their group, groups gather concurrently across different models.
         Avoids same-model timeout failures from naive asyncio.gather.
"""
from __future__ import annotations

import asyncio
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

# Parallel council run loads qwen3:8b (~5GB) + llama3.2:3b (~2GB) + qwen2.5:3b (~2GB)
# simultaneously — ~9GB model footprint. 12GB available is the conservative threshold,
# consistent with the brief's "requires 16GB+ total system RAM".
# Virgo and Pisces share qwen3:8b; that model is loaded once regardless.
PARALLEL_RAM_THRESHOLD_GB: float = 12.0


def check_parallel_viable(threshold_gb: float = PARALLEL_RAM_THRESHOLD_GB) -> tuple[bool, str]:
    """
    Returns (viable, reason). Never raises.

    Falls back to sequential (viable=False) on ImportError or any exception.
    Sequential is always safe; parallel is opportunistic.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb >= threshold_gb:
            return True, f"{available_gb:.1f}GB available >= {threshold_gb}GB threshold"
        return False, f"{available_gb:.1f}GB available < {threshold_gb}GB threshold — using sequential"
    except ImportError:
        return False, "psutil not installed — using sequential"
    except Exception as e:
        return False, f"RAM check failed ({e}) — using sequential"


async def run_parallel(
    query: str,
    roles: list[Role] | None = None,
) -> list[RoleResponse]:
    """
    Run council roles concurrently, grouped by model.
    Roles sharing a model run sequentially within their group; groups are
    gathered concurrently across different models.

    This avoids the timeout failure that occurs when naive asyncio.gather fires
    concurrent requests at the same model: Ollama serializes same-model inference
    internally, so a role queued behind another can exhaust its timeout waiting.
    Model-aware grouping eliminates that queue while preserving cross-model concurrency.

    Returns responses in COUNCIL_ROLES order regardless of completion order.
    Never raises — errors are captured in RoleResponse.error.

    Speedup ceiling on the current model lineup: Virgo+Pisces (qwen3:8b) run
    sequentially within their group while Critic and Seer run concurrently alongside
    them. Wall-clock gain = time saved on Critic+Seer (~57s on HAMON hardware).
    """
    from collections import defaultdict
    from .models import ROLE_MODELS

    if roles is None:
        roles = COUNCIL_ROLES

    # Group roles by assigned model, preserving within-group order.
    model_groups: dict[str, list[Role]] = defaultdict(list)
    for role in roles:
        model_groups[ROLE_MODELS[role]].append(role)

    async def _run_group(group_roles: list[Role]) -> list[RoleResponse]:
        """Run one model group sequentially, returning results in group order."""
        results = []
        for role in group_roles:
            system_prompt = get_system_prompt(role.value)
            results.append(await call_role(role, system_prompt, query))
        return results

    # Gather across groups (different models run concurrently).
    group_results: list[list[RoleResponse]] = list(
        await asyncio.gather(*[_run_group(g) for g in model_groups.values()])
    )

    # Flatten and restore caller-specified order.
    by_role: dict[Role, RoleResponse] = {
        r.role: r for group in group_results for r in group
    }
    return [by_role[role] for role in roles]


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
    parallel: bool = True,
) -> Any:
    """
    Full pipeline: router -> council -> review -> Archivist synthesis.
    Returns SynthesisResult. CLI entry point — uses non-streaming Archivist call.

    parallel=True (default): attempts parallel council execution if RAM check passes,
    falls back to sequential automatically.
    parallel=False: forces sequential (useful for debugging or low-RAM machines).
    """
    from .router import run_router

    router_result = await run_router(query)

    if parallel:
        viable, reason = check_parallel_viable()
        if viable:
            role_responses = await run_parallel(query)
        else:
            role_responses = await run_sequential(query)
    else:
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
    parallel: bool = True,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Full pipeline with streaming Archivist output. Phase 6 API entry point.

    Yields event dicts. The FastAPI handler wraps these in SSE formatting —
    no pipeline logic lives in api.py.

    Event types:
      {"type": "status",  "message": str}
      {"type": "router",  "data": dict}
      {"type": "token",   "content": str}
      {"type": "done",    "synthesis": str}
      {"type": "error",   "message": str}

    parallel=True (default): attempts parallel council execution if RAM check passes,
    falls back to sequential automatically. The "status" event before the council
    run reports which path was taken.
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
    # Council roles — parallel if viable, sequential fallback             #
    # ------------------------------------------------------------------ #
    if parallel:
        viable, reason = check_parallel_viable()
        if viable:
            yield {"type": "status", "message": f"Running council roles (parallel — {reason})..."}
            role_responses = await run_parallel(query)
        else:
            yield {"type": "status", "message": f"Running council roles (sequential — {reason})..."}
            role_responses = await run_sequential(query)
    else:
        yield {"type": "status", "message": "Running council roles (sequential)..."}
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
