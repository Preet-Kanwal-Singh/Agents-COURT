"""
core/synthesis.py — Archivist synthesis layer.

Owns:
  build_context_packet()    — assembles the full prompt the Archivist receives
  run_archivist()           — calls gemma4:latest via Ollama
  format_synthesis_output() — separates Archivist prose from pipeline metadata

Phase 5 changes:
  - build_context_packet now accepts a RouterResult and pre-computed final_weights.
  - classification, initial_weights, modifiers, and reasoning are all live from the router.
  - final_weights = initial_weights * 0.6 + review_votes * 0.4 (computed in runner).
  - weight_delta computed dynamically; any role with |final - initial| > 0.10 is
    flagged and the Archivist is instructed to name the tension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from core.prompt_loader import get_system_prompt

from core.models import Role
from core.ollama_client import call_role


# Archivist gets generous timeout — synthesis is the longest call in the pipeline.
# If ROLE_TIMEOUTS in models.py includes Role.ARCHIVIST, call_role will use it.
# This constant is a fallback reference; adjust call_role invocation if its
# signature reads timeout from ROLE_TIMEOUTS internally.
_ARCHIVIST_TIMEOUT = 180


@dataclass
class SynthesisResult:
    raw_output: str
    context_packet: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Context packet assembler
# ---------------------------------------------------------------------------

def build_context_packet(
    query: str,
    role_outputs: dict[str, str],          # {"virgo": ..., "pisces": ..., ...}
    review_votes: dict[str, float],        # aggregated, deanonymized weights
    blind_spots: list[str],                # one per reviewer pass
    collective_misses: list[str],          # one per reviewer pass — mandatory content
    router,                                # RouterResult from core.router (duck-typed)
    final_weights: dict[str, float],       # computed by runner: initial*0.6 + review*0.4
) -> str:
    """
    Constructs the full context packet sent to the Archivist as the user prompt.
    The Archivist's system prompt (archivist.md) defines its voice and behavior.
    This packet is the synthesis task.

    Role output order is preserved as passed — caller controls ordering.
    Convention: virgo → pisces → critic → seer (matches pipeline execution order).
    """

    # --- Role outputs ---
    role_section = "\n\n".join(
        f"{role.upper()}:\n{output.strip()}"
        for role, output in role_outputs.items()
    )

    # --- Router section ---
    router_weight_lines = "\n".join(
        f"    {role}: {w:.3f}"
        for role, w in sorted(router.initial_weights.items(), key=lambda x: -x[1])
    )

    # --- Review votes (raw reviewer output) ---
    review_vote_lines = "\n".join(
        f"    {role}: {score:.3f}"
        for role, score in sorted(review_votes.items(), key=lambda x: -x[1])
    )

    # --- Final weights (blended) ---
    final_weight_lines = "\n".join(
        f"    {role}: {score:.3f}"
        for role, score in sorted(final_weights.items(), key=lambda x: -x[1])
    )

    # --- Weight deltas: any role where |final - initial| > 0.10 ---
    deltas = {
        role: abs(final_weights.get(role, 0.0) - router.initial_weights[role])
        for role in router.initial_weights
        if abs(final_weights.get(role, 0.0) - router.initial_weights[role]) > 0.10
    }

    if deltas:
        delta_lines = "\n".join(
            f"    {role}: initial={router.initial_weights[role]:.3f} -> "
            f"final={final_weights.get(role, 0.0):.3f} (delta={d:.3f})"
            for role, d in sorted(deltas.items(), key=lambda x: -x[1])
        )
        delta_instruction = (
            "Flagged weight deltas (|final - initial| > 0.10):\n"
            + "\n".join(
                f"  {role}: initial={router.initial_weights[role]:.3f} -> "
                f"final={final_weights.get(role, 0.0):.3f} (delta={d:.3f})"
                for role, d in sorted(deltas.items(), key=lambda x: -x[1])
            )
            + "\nFor each flagged role, name the tension: what does the review round's "
            + "deviation from the router's prior reveal about what this query actually needed?"
        )
    else:
        delta_lines = "    none"
        delta_instruction = ""

    # --- Review round: blind spots and collective misses ---
    blind_spot_lines = "\n".join(
        f"  Pass {i + 1}: {bs.strip()}"
        for i, bs in enumerate(blind_spots)
    )
    collective_miss_lines = "\n".join(
        f"  Pass {i + 1}: {cm.strip()}"
        for i, cm in enumerate(collective_misses)
    )

    packet = f"""QUERY:
{query.strip()}

ROUTER:
  classification: {router.classification}  stage: {router.stage}
  initial_weights (post-modifier):
{router_weight_lines}
  modifiers: conclusion_implied={router.modifiers.get('conclusion_implied', False)}  context_dense={router.modifiers.get('context_dense', False)}
  reasoning: {router.reasoning}

ROLE OUTPUTS:
{role_section}

REVIEW SUMMARY:
  review_votes (mean across reviewer passes):
{review_vote_lines}

  final_weights (initial_weights * 0.6 + review_votes * 0.4):
{final_weight_lines}

  weight_delta (|final - initial| > 0.10):
{delta_lines}

  blind_spots (one per reviewer pass):
{blind_spot_lines}

  collective_misses (one per reviewer pass):
{collective_miss_lines}

---
SYNTHESIS INSTRUCTIONS:

Synthesise in full Archivist voice.
Weight the role outputs according to final_weights above.
Address the collective misses directly — they are mandatory, not optional context.
{delta_instruction}
Do not summarise the four roles. Synthesise them — the output should say something
none of them said alone.
"""

    return packet.strip()


# ---------------------------------------------------------------------------
# Archivist model call
# ---------------------------------------------------------------------------

async def run_archivist(context_packet: str) -> SynthesisResult:
    try:
        system_prompt = get_system_prompt("archivist")

        response = await call_role(
            role=Role.ARCHIVIST,
            system_prompt=system_prompt,
            query=context_packet,
        )

        error = getattr(response, "error", None)
        content = response.content if not error else ""

        return SynthesisResult(
            raw_output=content,
            context_packet=context_packet,
            error=error,
        )

    except Exception as exc:
        return SynthesisResult(
            raw_output="",
            context_packet=context_packet,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Response formatter
# ---------------------------------------------------------------------------

def format_synthesis_output(result: SynthesisResult) -> str:
    """
    Formats the Archivist's prose output for display.
    Separates synthesis from pipeline metadata. No JSON parsing.
    """
    if result.error:
        return f"[Archivist error: {result.error}]"

    if not result.raw_output.strip():
        return "[Archivist returned empty output — check model load and timeout]"

    bar = "-" * 60
    return (
        f"\n{bar}\n"
        f"ARCHIVIST SYNTHESIS\n"
        f"{bar}\n\n"
        f"{result.raw_output.strip()}\n\n"
        f"{bar}\n"
    )
