"""
Standalone contract test for the Archivist synthesis step.
Mirrors test_reviewer.py pattern: hardcoded context, live model call, inspect output.

Run this before wiring synthesis into the pipeline.
Pass condition: gemma4:latest returns coherent prose synthesis that addresses
all four role outputs and the collective miss. No JSON validation needed.

Usage:
    python test_archivist.py
"""

import asyncio
from core.synthesis import build_context_packet, run_archivist, format_synthesis_output

# ---------------------------------------------------------------------------
# Hardcoded HAMON context
# Representative of Phase 3 output shape — not transcripts, stand-ins that
# match each role's structural character. Test is about gemma4:latest handling
# the context packet, not about producing a perfect synthesis on first call.
# ---------------------------------------------------------------------------

HAMON_QUERY = (
    "If Hamon were to prove that deep compatibility correlates only with a set of "
    "universally measurable behavioral metrics (e.g., the capacity for non-violent "
    "conflict resolution; resilience in ambiguity), would the convergence between "
    "MBTI and Vedic astrology become redundant? If yes, which part of your synthesis — "
    "the empirical rigor or the symbolic depth — is doing all the heavy lifting simply "
    "to maintain structural weight?"
)

DUMMY_ROLE_OUTPUTS = {
    "virgo": (
        "The question has an embedded assumption worth naming before proceeding: it treats "
        "'redundant' as binary — either the convergence survives empirical validation or it "
        "doesn't. The real question is what work each component is doing independently.\n\n"
        "If Hamon's metrics are descriptive (measuring what correlates with compatibility) "
        "rather than prescriptive (explaining why), then MBTI and Vedic astrology remain "
        "useful as navigational heuristics even in a world where Hamon is validated. "
        "The empirical rigor is load-bearing only if you're making predictive claims. "
        "The symbolic depth is load-bearing if the framework generates a shared vocabulary "
        "between partners — a different function entirely from prediction.\n\n"
        "The implied conclusion — that one must be doing the heavy lifting — assumes the "
        "two functions are in competition. They may not be."
    ),
    "pisces": (
        "The question assumes that if empirical metrics could fully account for compatibility, "
        "the symbolic frameworks would be exposed as decorative. But this assumes meaning is "
        "downstream of measurement — that the only valid function of a framework is predictive "
        "accuracy.\n\n"
        "Symbolic depth doesn't compete with empirical rigor. It operates in a different "
        "register entirely. MBTI and Vedic astrology — whatever their predictive failures — "
        "serve as containers for self-concept. They give people a language for interior "
        "experience that behavioral metrics don't touch. That's not a failure mode of "
        "the symbolic. It's its entire function.\n\n"
        "The synthesis isn't holding structure together by borrowing credibility from "
        "empirical rigor. It's generating something neither framework produces alone: a "
        "cross-domain vocabulary for compatibility that feels both legitimate and meaningful. "
        "Whether that's valid depends on what you think validity is for."
    ),
    "critic": (
        "Three assumptions in the framing need examination:\n\n"
        "1. The conditional 'if Hamon were to prove' is taken as stipulated. But the "
        "question then moves from 'behavioral metrics correlate with compatibility' to "
        "'therefore symbolic frameworks become redundant.' This doesn't follow without "
        "an additional premise: that MBTI and Vedic astrology's only valid function is "
        "predicting the same thing Hamon measures.\n\n"
        "2. 'Redundant' assumes the frameworks are competing for the same job. If one is "
        "a measurement instrument and the other is a meaning-making system, proving one "
        "doesn't falsify the other.\n\n"
        "3. 'Heavy lifting to maintain structural weight' implies the synthesis is "
        "architecturally fragile. This may be true, but it's asserted not demonstrated. "
        "What would evidence of structural fragility actually look like here?"
    ),
    "seer": (
        "I don't know what Hamon is. The question names it as if it's established.\n\n"
        "What I can see: the question asks whether, if a new measurement system worked, "
        "two older frameworks would become unnecessary. Then it asks which part of the "
        "older system — the scientific-sounding part or the symbolic part — is actually "
        "doing the work.\n\n"
        "What I can't see: what 'deep compatibility' means. The question uses it as if "
        "it's defined. It isn't. If 'deep compatibility' is defined as whatever Hamon "
        "measures, then yes — Hamon makes the others redundant by definition. If it means "
        "something the behavioral metrics don't capture, the question answers itself.\n\n"
        "The word 'synthesis' also does a lot of unnamed work. Synthesis of what, "
        "for what purpose, used by whom."
    ),
}

# Phase 3 aggregated weights — empirically confirmed on HAMON
DUMMY_REVIEW_VOTES = {
    "virgo": 0.450,
    "pisces": 0.305,
    "seer":   0.144,
    "critic": 0.100,
}

# One blind spot per reviewer pass (representative, not transcripts)
DUMMY_BLIND_SPOTS = [
    "Critic names three structural problems but doesn't engage with any of them — identifies gaps without filling them.",
    "Virgo treats the descriptive/prescriptive distinction as decisive but doesn't establish why. The claim needs more than assertion.",
    "Seer correctly names undefined terms but uses them as an exit — stops at the gap rather than asking what happens if it's filled.",
    "Pisces is the most complete response but sidesteps the redundancy question by reframing it as a category error without examining whether the reframe is valid.",
]

# One collective miss per reviewer pass — the highest-signal output of the review round
DUMMY_COLLECTIVE_MISSES = [
    "All four responses failed to engage with whether empirical rigor and symbolic depth could be genuinely complementary — mutually generative rather than parallel. Every response accepted the zero-sum frame rather than interrogating it.",
    "None of the responses addressed who benefits from maintaining the synthesis. The framing implies the synthesis is being defended for structural reasons rather than because it serves actual users.",
    "All four missed that the question's premise — if Hamon proves its case — contains a hidden assumption about what proof looks like in a domain about human compatibility.",
    "The council failed to name that 'heavy lifting to maintain structural weight' is a load-bearing metaphor the question never defines. What counts as structural weight in a compatibility framework?",
]


async def main():
    print("=" * 60)
    print("Archivist Contract Test")
    print("Model: gemma4:latest")
    print("Query: HAMON benchmark")
    print("=" * 60)
    print()
    print("Building context packet...")
    print()

    packet = build_context_packet(
        query=HAMON_QUERY,
        role_outputs=DUMMY_ROLE_OUTPUTS,
        review_votes=DUMMY_REVIEW_VOTES,
        blind_spots=DUMMY_BLIND_SPOTS,
        collective_misses=DUMMY_COLLECTIVE_MISSES,
    )

    print("--- Context Packet (sent to Archivist) ---")
    print(packet)
    print()
    print("--- Calling gemma4:latest (may take 60–120s) ---")
    print()

    result = await run_archivist(packet)

    if result.error:
        print(f"[FAIL] {result.error}")
        return

    print(format_synthesis_output(result))
    print("[Pass — inspect output above. Prose should address collective miss and name Virgo tension.]")


if __name__ == "__main__":
    asyncio.run(main())
