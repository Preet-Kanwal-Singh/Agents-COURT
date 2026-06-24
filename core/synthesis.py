"""
core/synthesis.py — Phase 4: Archivist synthesis layer.

Owns:
  build_context_packet()    — assembles the full prompt the Archivist receives
  run_archivist()           — calls gemma4:latest via Ollama
  format_synthesis_output() — separates Archivist prose from pipeline metadata

Phase 4 constraints:
  - final_weights == review_votes (router not yet active — Phase 5 dependency)
  - classification defaults to "unclassified" — do not fabricate router output
  - weight_delta is not computed dynamically (no initial_weights without router),
    but the Virgo tension from Phase 3 is flagged explicitly: Virgo received 0.450
    on a pisces-signal query; expected weight under router is ~0.10. Delta exceeds
    the 0.10 threshold — the Archivist is instructed to name it.

Wiring into runner.py (Phase 4 addition):
  from core.synthesis import build_context_packet, run_archivist, format_synthesis_output

  packet = build_context_packet(
      query=query,
      role_outputs={r.role.value.lower(): r.content for r in role_responses},
      review_votes=aggregated_review.review_votes,
      blind_spots=[r.blind_spot_detail for r in aggregated_review.reviews],
      collective_misses=[r.collective_miss for r in aggregated_review.reviews],
  )
  synthesis = await run_archivist(packet)
  print(format_synthesis_output(synthesis))
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
    classification: str = "unclassified",  # placeholder until Phase 5 router
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

    # --- Weights (sorted descending for readability) ---
    weight_lines = "\n".join(
        f"  {role}: {score:.3f}"
        for role, score in sorted(review_votes.items(), key=lambda x: -x[1])
    )

    # --- Review round: blind spots and collective misses ---
    blind_spot_lines = "\n".join(
        f"  Pass {i + 1}: {bs.strip()}"
        for i, bs in enumerate(blind_spots)
    )
    collective_miss_lines = "\n".join(
        f"  Pass {i + 1}: {cm.strip()}"
        for i, cm in enumerate(collective_misses)
    )

    # --- Weight delta note (Phase 4 hardcoded — Phase 5 computes dynamically) ---
    # Virgo at 0.450 on a pisces-signal query. Under the router's pisces-signal
    # vector (virgo: 0.10), the delta would be 0.35 — well above the 0.10 threshold
    # that triggers mandatory tension-naming in the synthesis.
    weight_delta_note = (
        "Router not yet active — final_weights == review_votes (Phase 5 dependency).\n"
        "  Phase 3 finding: Virgo received 0.450 on a query expected to classify as\n"
        "  pisces-signal (expected router weight ~0.10). Delta ~0.35 exceeds the 0.10\n"
        "  threshold. The Archivist must name this tension in the synthesis."
    )

    packet = f"""QUERY:
{query.strip()}

ROUTER:
  classification: {classification}
  initial_weights: N/A ({weight_delta_note})

ROLE OUTPUTS:
{role_section}

REVIEW SUMMARY:
  final_weights (= review_votes, router not yet active):
{weight_lines}

  blind_spots (one per reviewer pass):
{blind_spot_lines}

  collective_misses (one per reviewer pass):
{collective_miss_lines}

---
SYNTHESIS INSTRUCTIONS:

Synthesise in full Archivist voice.
Weight the role outputs according to final_weights above.
Address the collective misses directly — they are mandatory, not optional context.
The Virgo weight (0.450) substantially exceeds its expected value on a pisces-signal
query. Name this tension: what does it mean that the most analytically precise response
scored highest on a question about meaning-making and symbolic frameworks?
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
