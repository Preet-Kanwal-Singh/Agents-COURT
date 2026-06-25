"""
core/synthesis.py — Archivist synthesis layer.

Phase 4: build_context_packet, run_archivist, format_synthesis_output.
Phase 5: build_context_packet updated to accept router output (initial_weights,
         modifiers) and compute final_weights dynamically. Weight delta detection
         is now live rather than hardcoded.
Phase 6: stream_archivist (streaming variant of run_archivist), strip_latex
         (LaTeX cleanup before API responses). strip_latex lives here because
         synthesis.py owns all Archivist output processing. It is NOT applied
         in format_synthesis_output — that function is CLI-only. The API
         response handler calls strip_latex on the assembled streaming output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from core.prompt_loader import get_system_prompt
from core.models import Role
from core.ollama_client import call_role, stream_role

_ARCHIVIST_TIMEOUT = 180  # reference constant; actual timeout is in ROLE_TIMEOUTS


@dataclass
class SynthesisResult:
    raw_output: str
    context_packet: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LaTeX strip (Phase 6)
# ---------------------------------------------------------------------------

# Patterns observed in gemma4 output: $X \rightarrow Y$, \text{...}, bare \cmd.
# Display math ($$...$$) stripped first to avoid the inline pattern consuming
# one of the dollar signs.
_LATEX_DISPLAY   = re.compile(r'\$\$.*?\$\$', re.DOTALL)
_LATEX_INLINE    = re.compile(r'\$([^$\n]+)\$')
_LATEX_TEXT_CMD  = re.compile(r'\\text\{([^}]*)\}')
_LATEX_CMD_ARG   = re.compile(r'\\[a-zA-Z]+\{([^}]*)\}')
_LATEX_CMD_BARE  = re.compile(r'\\([a-zA-Z]+)')

# Bare commands with useful Unicode equivalents.
# Arrow variants are the most common gemma4 LaTeX output; extend as needed.
_CMD_REPLACEMENTS: dict[str, str] = {
    "rightarrow":      "->",
    "leftarrow":       "<-",
    "Rightarrow":      "=>",
    "Leftarrow":       "<=",
    "leftrightarrow":  "<->",
    "to":              "->",
    "mapsto":          "|->",
    "cdot":            "*",
    "times":           "x",
    "ldots":           "...",
    "cdots":           "...",
    "implies":         "=>",
    "iff":             "<=>",
}


def strip_latex(text: str) -> str:
    """
    Remove LaTeX constructs from Archivist output before sending to API clients.

    Order matters:
      1. Display math ($$...$$) — strip delimiters, keep inner content.
      2. Inline math ($...$)    — strip delimiters, keep inner content.
      3. \\text{...}            — keep content (most readable part of math text).
      4. \\cmd{arg}             — keep arg.
      5. Bare \\cmd             — replace with ASCII equivalent or remove.
    """
    text = _LATEX_DISPLAY.sub(lambda m: m.group(0)[2:-2].strip(), text)
    text = _LATEX_INLINE.sub(lambda m: m.group(1).strip(), text)
    text = _LATEX_TEXT_CMD.sub(lambda m: m.group(1), text)
    text = _LATEX_CMD_ARG.sub(lambda m: m.group(1), text)
    text = _LATEX_CMD_BARE.sub(lambda m: _CMD_REPLACEMENTS.get(m.group(1), ""), text)
    return text


# ---------------------------------------------------------------------------
# Context packet assembler
# ---------------------------------------------------------------------------

def build_context_packet(
    query: str,
    role_outputs: dict[str, str],
    review_votes: dict[str, float],
    blind_spots: list[str],
    collective_misses: list[str],
    classification: str = "unclassified",
    initial_weights: dict[str, float] | None = None,
    modifiers: dict[str, bool] | None = None,
) -> str:
    """
    Constructs the full context packet sent to the Archivist as the user prompt.

    Phase 5 additions:
      - initial_weights: from the router's weight vector for the classification.
        If None (router not active), final_weights == review_votes.
      - modifiers: {"conclusion_implied": bool, "context_dense": bool}.
        Logged in the packet; weight adjustments are already reflected in
        initial_weights (applied by the router before calling here).
      - final_weights are computed dynamically:
          final_weight[role] = initial_weights[role] * 0.6 + review_votes[role] * 0.4
      - Weight delta flagging: any role where |final - initial| > 0.10 is listed
        explicitly — the Archivist is instructed to name the tension.

    Role output order is preserved as passed. Convention: virgo -> pisces -> critic -> seer.
    """
    # --- Final weight computation ---
    if initial_weights:
        final_weights = {
            role: round(initial_weights.get(role, 0.25) * 0.6 + score * 0.4, 4)
            for role, score in review_votes.items()
        }
    else:
        final_weights = dict(review_votes)

    # --- Weight delta flags ---
    delta_flags: list[str] = []
    if initial_weights:
        for role, final in final_weights.items():
            initial = initial_weights.get(role, 0.25)
            delta = abs(final - initial)
            if delta > 0.10:
                direction = "up" if final > initial else "down"
                delta_flags.append(
                    f"  {role}: initial={initial:.3f} -> final={final:.3f} "
                    f"(delta={delta:.3f}, shifted {direction})"
                )

    # --- Sections ---
    role_section = "\n\n".join(
        f"{role.upper()}:\n{output.strip()}"
        for role, output in role_outputs.items()
    )

    initial_weight_line = (
        "  " + ", ".join(f"{r}: {v:.3f}" for r, v in sorted(
            initial_weights.items(), key=lambda x: -x[1]
        ))
        if initial_weights else "  N/A (router not active)"
    )

    modifier_line = (
        "  " + ", ".join(f"{k}: {v}" for k, v in (modifiers or {}).items())
        if modifiers else "  none"
    )

    final_weight_lines = "\n".join(
        f"  {role}: {score:.3f}"
        for role, score in sorted(final_weights.items(), key=lambda x: -x[1])
    )

    review_vote_lines = "\n".join(
        f"  {role}: {score:.3f}"
        for role, score in sorted(review_votes.items(), key=lambda x: -x[1])
    )

    blind_spot_lines = "\n".join(
        f"  Pass {i + 1}: {bs.strip()}"
        for i, bs in enumerate(blind_spots)
    )
    collective_miss_lines = "\n".join(
        f"  Pass {i + 1}: {cm.strip()}"
        for i, cm in enumerate(collective_misses)
    )

    delta_section = (
        "\n".join(delta_flags)
        if delta_flags
        else "  none (all deltas within 0.10 threshold)"
    )

    # --- Synthesis instruction for weight tension ---
    tension_instruction = ""
    if delta_flags:
        tension_instruction = (
            "\nWeight tension detected on the following roles (delta > 0.10):\n"
            + "\n".join(delta_flags)
            + "\nName this tension in the synthesis — what does it mean that the "
            "review round moved these weights significantly from the router's initial "
            "classification? This is mandatory, not optional commentary."
        )

    packet = f"""QUERY:
{query.strip()}

ROUTER:
  classification: {classification}
  initial_weights: {initial_weight_line}
  modifiers: {modifier_line}

ROLE OUTPUTS:
{role_section}

REVIEW SUMMARY:
  review_votes (mean normalized score across reviewer passes):
{review_vote_lines}

  final_weights (router 0.6 + review 0.4):
{final_weight_lines}

  weight_delta (roles where |final - initial| > 0.10):
{delta_section}

  blind_spots (one per reviewer pass):
{blind_spot_lines}

  collective_misses (one per reviewer pass):
{collective_miss_lines}

---
SYNTHESIS INSTRUCTIONS:

Synthesise in full Archivist voice.
Weight the role outputs according to final_weights above.
Address the collective misses directly — they are mandatory, not optional context.
Do not summarise the four roles. Synthesise them — the output should say something none of them said alone.{tension_instruction}
"""

    return packet.strip()


# ---------------------------------------------------------------------------
# Archivist model call — non-streaming (CLI / Phase 4+)
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
# Archivist model call — streaming (Phase 6 / API)
# ---------------------------------------------------------------------------

async def stream_archivist(context_packet: str) -> AsyncGenerator[str, None]:
    """
    Streaming variant of run_archivist. Yields raw content tokens.

    think=False is enforced via _THINKING_DISABLED in ollama_client — Archivist
    is in that set. Errors are yielded as terminal error-marker tokens; the
    caller accumulates the full response and can inspect for the marker prefix.

    LaTeX stripping is NOT applied here — strip_latex() is called by the
    API response handler on the fully assembled output, not per-token.
    """
    system_prompt = get_system_prompt("archivist")
    async for token in stream_role(Role.ARCHIVIST, system_prompt, context_packet):
        yield token


# ---------------------------------------------------------------------------
# Response formatter — CLI only
# ---------------------------------------------------------------------------

def format_synthesis_output(result: SynthesisResult) -> str:
    """
    Formats the Archivist's prose output for CLI display.
    Not used by the API layer — LaTeX stripping and formatting happen there.
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
