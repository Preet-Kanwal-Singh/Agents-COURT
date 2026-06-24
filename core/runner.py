"""
Sequential council runner — Phase 2/3.

Phase 2: run_sequential — loops Virgo → Pisces → Critic → Seer, returns
flat list of RoleResponse objects.

Phase 3: run_review_round — single Reviewer pass (primitive, used in
standalone testing). run_aggregated_review — four independent passes with
score aggregation, the full Phase 3 spec.

Phase 7 note: swap run_sequential for an asyncio.gather-based version here
without touching anything upstream.
"""
from statistics import mean

from .models import Role, RoleResponse
from .ollama_client import call_role
from .prompt_loader import get_system_prompt
from .synthesis import build_context_packet, run_archivist, format_synthesis_output
from .review import (
    anonymize, build_review_prompt, extract_json, validate_review,
    normalize_scores, flag_quality_issues, deanonymize_scores,
    ReviewResult, AggregatedReview, ReviewParseError,
)

# Canonical council order — Archivist and Reviewer are not part of the round
COUNCIL_ROLES: list[Role] = [Role.VIRGO, Role.PISCES, Role.CRITIC, Role.SEER]

async def run_sequential(
    query: str,
    roles: list[Role] | None = None,
) -> list[RoleResponse]:
    """
    Run roles sequentially. Returns all responses including errors.
    Never raises — failed roles are captured with .error set.

    Args:
        query:  The user query sent to every role.
        roles:  Roles to run. Defaults to COUNCIL_ROLES.
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
    Run a single Reviewer pass on the council's outputs.

    Anonymizes the four responses (shuffled), builds the review prompt,
    calls the Reviewer, parses and validates the JSON, normalizes scores,
    and deanonymizes back to role names before returning.

    Returns a ReviewResult with .ok=False and .error set if the Reviewer
    call fails or produces output that can't be recovered even with
    json-repair. Never raises — errors are captured in the result so the
    caller can decide how to handle a failed review round.

    Note: `classification` is a placeholder until Phase 5's router lands.
    Pass the router's classification string once that exists; for now
    "unclassified" is explicit about the gap rather than fabricating a
    classification that isn't real yet.
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

    Each pass calls anonymize() independently so the A/B/C/D shuffle is
    different every time — this prevents positional bias from accumulating
    across reviewers. A reviewer that always sees Pisces at position C
    might score consistently based on position rather than content.

    Failed passes are excluded from aggregation but counted in error_count.
    Partial aggregation (e.g. 3/4 passes succeeded) is still returned
    rather than raising — the Archivist can note the reduced sample in
    synthesis if needed.
    """
    role_names = [r.value for r in COUNCIL_ROLES]

    reviews: list[ReviewResult] = []
    for _ in range(num_reviewers):
        result = await run_review_round(query, council_responses, classification)
        reviews.append(result)

    ok_reviews = [r for r in reviews if r.ok]

    # Aggregate normalized scores: mean across all successful passes.
    # Each pass produces scores summing to 1.0, so the mean also sums to 1.0.
    all_scores: dict[str, list[float]] = {role: [] for role in role_names}
    for review in ok_reviews:
        for role, score in review.role_scores.items():
            all_scores[role].append(score)

    review_votes = {
        role: mean(scores) if scores else 0.0
        for role, scores in all_scores.items()
    }

    # Count how many reviewers named each role as strongest / weakest blind spot.
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
    classification: str = "unclassified",
):
    """
    Full pipeline: sequential council → aggregated review → Archivist synthesis.
    Returns SynthesisResult. Caller handles formatting and display.
    """
    role_responses = await run_sequential(query)

    aggregated_review = await run_aggregated_review(
        query=query,
        council_responses=role_responses,
        num_reviewers=num_reviewers,
        classification=classification,
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
    )

    return await run_archivist(packet)
