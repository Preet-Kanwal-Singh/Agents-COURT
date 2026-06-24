"""
Phase 1–3 test runner.

Usage:
    python main.py                          # phase 1, virgo, benchmark query
    python main.py --role pisces            # phase 1, different role
    python main.py --phase 2               # phase 2, all council roles
    python main.py --phase 3               # phase 3, council + review round
    python main.py --check                 # health check only, no query
    python main.py --query "..."           # custom query string
"""
import asyncio
import argparse
import sys
from core import (
    Role, ROLE_MODELS,
    call_role, ping_ollama, list_loaded_models,
    get_system_prompt,
    run_sequential, COUNCIL_ROLES,
)
from core.runner import run_aggregated_review
from core.runner import run_full_pipeline
from core.synthesis import format_synthesis_output

TEST_QUERY = (
    "If Hamon were to prove that deep compatibility correlates only with a set of "
    "universally measurable behavioral metrics (e.g., the capacity for non-violent "
    "conflict resolution; resilience in ambiguity), would the convergence between "
    "MBTI and Vedic astrology become redundant? If yes, which part of your synthesis "
    "— the empirical rigor or the symbolic depth — is doing all the heavy lifting "
    "simply to maintain structural weight?"
)

_SEP = "=" * 64
_SEP_THIN = "-" * 64


async def health_check() -> bool:
    print("— Ollama health check —")
    reachable = await ping_ollama()
    if not reachable:
        print("✗ Ollama not reachable. Run: ollama serve")
        return False
    print("✓ Ollama running")

    loaded = await list_loaded_models()
    if loaded:
        print(f"  Loaded in memory: {', '.join(loaded)}")
    else:
        print("  No models currently loaded (first call will load the model)")
    return True


async def run_single_role(role: Role, query: str) -> None:
    model = ROLE_MODELS[role]
    print(f"\n— Loading system prompt for: {role.value} —")
    try:
        system_prompt = get_system_prompt(role.value)
        preview = system_prompt[:120].replace("\n", " ")
        print(f"  Prompt preview: {preview}...")
        print(f"  Prompt length: {len(system_prompt)} chars")
    except Exception as e:
        print(f"✗ Failed to load system prompt: {e}")
        sys.exit(1)

    print(f"\n— Calling {role.value} ({model}) —")
    print(f"  Query length: {len(query)} chars")
    print("  Waiting for response...\n")

    response = await call_role(role, system_prompt, query)

    if not response.ok:
        print(f"✗ Error: {response.error}")
        sys.exit(1)

    print(f"✓ Response received ({response.duration_ms}ms)\n")
    print(_SEP)
    print(f"ROLE: {response.role.value.upper()}  |  MODEL: {response.model}  |  {response.duration_ms}ms")
    print(_SEP)
    print(response.content)
    print(_SEP)


async def run_all_roles(query: str) -> list:
    print(f"\n— Phase 2: sequential council run ({len(COUNCIL_ROLES)} roles) —")
    print(f"  Roles: {' → '.join(r.value for r in COUNCIL_ROLES)}")
    print(f"  Query length: {len(query)} chars\n")

    responses = await run_sequential(query)

    ok_count = sum(1 for r in responses if r.ok)
    err_count = len(responses) - ok_count
    total_ms = sum(r.duration_ms for r in responses)

    for resp in responses:
        print(_SEP)
        if resp.ok:
            print(f"ROLE: {resp.role.value.upper()}  |  MODEL: {resp.model}  |  {resp.duration_ms}ms")
            print(_SEP)
            print(resp.content)
        else:
            print(f"ROLE: {resp.role.value.upper()}  |  MODEL: {resp.model}  |  ERROR")
            print(_SEP)
            print(f"✗ {resp.error}")
        print()

    print(_SEP)
    status = f"{ok_count} ok" + (f"  |  {err_count} errors" if err_count else "")
    print(f"Council complete — {len(responses)} roles  |  {status}  |  {total_ms}ms total")
    print(_SEP)

    if ok_count == 0:
        sys.exit(1)

    return responses


async def run_with_review(query: str) -> None:
    # Step 1: full council run
    responses = await run_all_roles(query)

    ok_responses = [r for r in responses if r.ok]
    if len(ok_responses) < 4:
        print(f"\n✗ Review round requires 4 successful council responses, got {len(ok_responses)}. Skipping review.")
        sys.exit(1)

    # Step 2: four reviewer passes with aggregation
    print(f"\n— Phase 3: review round (4 passes, {ROLE_MODELS[Role.REVIEWER]}) —")
    print("  Each pass uses an independent shuffle to prevent positional bias.")
    print("  Running reviewer 1 of 4...\n")

    agg = await run_aggregated_review(query, responses)

    # Print per-pass progress summary (brief — full content shown in aggregate below)
    for i, review in enumerate(agg.reviews, 1):
        if review.ok:
            flags = " ⚠ collective_miss flagged" if review.quality_flags else ""
            repaired = " ⚠ JSON repaired" if review.was_repaired else ""
            print(f"  Pass {i}: ✓  strongest={review.strongest}  blind_spot={review.weakest_blind_spot}{flags}{repaired}")
        else:
            print(f"  Pass {i}: ✗  {review.error}")

    if agg.error_count > 0:
        print(f"\n  ⚠ {agg.error_count}/4 passes failed — review_votes computed from {4 - agg.error_count} samples")

    print(f"\n{_SEP}")
    print(f"AGGREGATED REVIEW  |  {4 - agg.error_count}/4 passes  |  MODEL: {ROLE_MODELS[Role.REVIEWER]}")
    print(_SEP)

    print("\nRole weights (review_votes — mean normalized score across passes):")
    for role_name, weight in sorted(agg.review_votes.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 20)
        print(f"  {role_name:<10} {weight:.3f}  {bar}")

    print(f"\nStrongest votes:    ", end="")
    print("  ".join(f"{r}×{c}" for r, c in sorted(agg.strongest_counts.items(), key=lambda x: -x[1]) if c > 0))

    print(f"Blind spot votes:   ", end="")
    print("  ".join(f"{r}×{c}" for r, c in sorted(agg.blind_spot_counts.items(), key=lambda x: -x[1]) if c > 0))

    print(f"\nBlind spots (one per reviewer):")
    for i, detail in enumerate(agg.blind_spots, 1):
        print(f"  [{i}] {detail}")

    print(f"\nCollective misses (one per reviewer):")
    for i, miss in enumerate(agg.collective_misses, 1):
        print(f"  [{i}] {miss}")

    if agg.quality_flag_count > 0:
        print(f"\n  ⚠ {agg.quality_flag_count}/4 passes returned a non-answer for collective_miss")
    if agg.repair_count > 0:
        print(f"  ⚠ {agg.repair_count}/4 passes required JSON repair")

    print(_SEP)

async def run_phase4(query: str) -> None:
    print(f"\n— Phase 4: full pipeline — council → review → Archivist synthesis —")
    print(f"  Query length: {len(query)} chars\n")

    result = await run_full_pipeline(query)

    if result.error:
        print(f"✗ Archivist failed: {result.error}")
        sys.exit(1)

    print(format_synthesis_output(result))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Archivist Council test runner")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Phase to run: 1 = single role, 2 = full council, 3 = council + review (default: 1)")
    parser.add_argument("--role", default="virgo",
                        choices=["virgo", "pisces", "critic", "seer", "archivist"],
                        help="Role to call (phase 1 only, ignored in phase 2/3)")
    parser.add_argument("--check", action="store_true",
                        help="Health check only, skip the query")
    parser.add_argument("--query", default=None,
                        help="Custom query string (defaults to HAMON benchmark query)")
    args = parser.parse_args()

    ok = await health_check()
    if not ok or args.check:
        sys.exit(0 if ok else 1)

    query = args.query or TEST_QUERY

    if args.phase == 4:
        await run_phase4(query)
    elif args.phase == 3:
        await run_with_review(query)
    elif args.phase == 2:
        await run_all_roles(query)
    else:
        role = Role(args.role)
        await run_single_role(role, query)


if __name__ == "__main__":
    asyncio.run(main())
