"""
Phase 5 contract test — router.
Stage 1/modifier/weight tests run without model calls.
Full router test requires Ollama: python test_router.py --stage2
"""

import argparse
import asyncio
import sys

from core.router import (
    classify_stage1,
    apply_modifiers,
    compute_final_weights,
    get_weight_deltas,
    run_router,
    INITIAL_WEIGHT_VECTORS,
)

HAMON = (
    "If Hamon were to prove that deep compatibility correlates only with a set of "
    "universally measurable behavioral metrics (e.g., the capacity for non-violent "
    "conflict resolution; resilience in ambiguity), would the convergence between "
    "MBTI and Vedic astrology become redundant? If yes, which part of your synthesis "
    "\u2014 the empirical rigor or the symbolic depth \u2014 is doing all the heavy "
    "lifting simply to maintain structural weight?"
)

# Phase 3 values from DEVIATIONS.md — used to verify the weight formula
_raw = {"virgo": 0.450, "pisces": 0.305, "seer": 0.144, "critic": 0.100}
PHASE3_REVIEW_VOTES = {k: v / sum(_raw.values()) for k, v in _raw.items()}

OK  = "[PASS]"
ERR = "[FAIL]"


def _fmt(w: dict) -> str:
    return "  ".join(f"{k}: {v:.3f}" for k, v in w.items())


def test_stage1() -> bool:
    print("=== Stage 1: classification ===\n")
    classification, modifiers, reasoning = classify_stage1(HAMON)

    class_ok = classification == "pisces"
    ci_ok    = modifiers["conclusion_implied"] is True
    cd_ok    = modifiers["context_dense"] is True
    passed   = class_ok and ci_ok and cd_ok

    print(f"{OK if passed else ERR}  HAMON")
    print(f"  classification:     {classification}  (expected: pisces)  {'ok' if class_ok else 'WRONG'}")
    print(f"  conclusion_implied: {modifiers['conclusion_implied']}  (expected: True)   {'ok' if ci_ok else 'WRONG'}")
    print(f"  context_dense:      {modifiers['context_dense']}  (expected: True)   {'ok' if cd_ok else 'WRONG'}")
    print(f"  reasoning: {reasoning}\n")
    return passed


def test_modifiers() -> bool:
    print("=== Modifier application ===\n")
    base = dict(INITIAL_WEIGHT_VECTORS["pisces"])
    after = apply_modifiers(dict(base), {"conclusion_implied": True, "context_dense": True})

    critic_up = after["critic"] > base["critic"]
    seer_up   = after["seer"]   > base["seer"]
    sum_ok    = abs(sum(after.values()) - 1.0) < 1e-6
    passed    = critic_up and seer_up and sum_ok

    print(f"{OK if passed else ERR}  pisces base + both modifiers")
    print(f"  critic: {base['critic']:.3f} -> {after['critic']:.4f}  {'ok' if critic_up else 'WRONG'}")
    print(f"  seer:   {base['seer']:.3f} -> {after['seer']:.4f}  {'ok' if seer_up else 'WRONG'}")
    print(f"  sum:    {sum(after.values()):.6f}  {'ok' if sum_ok else 'WRONG'}\n")
    return passed


def test_final_weights() -> bool:
    print("=== Final weight formula (HAMON + Phase 3 review_votes) ===\n")
    _, modifiers, _ = classify_stage1(HAMON)
    base    = dict(INITIAL_WEIGHT_VECTORS["pisces"])
    initial = apply_modifiers(base, modifiers)
    final   = compute_final_weights(initial, PHASE3_REVIEW_VOTES)
    deltas  = get_weight_deltas(initial, final)

    virgo_flagged = "virgo" in deltas
    sum_ok        = abs(sum(final.values()) - 1.0) < 1e-6
    passed        = virgo_flagged and sum_ok

    print(f"{OK if passed else ERR}  HAMON final weights")
    print(f"  initial:       {_fmt(initial)}")
    print(f"  final:         {_fmt(final)}")
    print(f"  deltas >0.10:  {deltas}")
    print(f"  virgo flagged: {virgo_flagged}  (expected: True)  {'ok' if virgo_flagged else 'WRONG'}")
    print(f"  sum: {sum(final.values()):.6f}  {'ok' if sum_ok else 'WRONG'}\n")
    return passed


async def test_full_router() -> bool:
    print("=== Full router — HAMON (Stage 2 fires only if Stage 1 returns ambiguous) ===\n")
    result = await run_router(HAMON)

    class_ok = result.classification == "pisces"
    ci_ok    = result.modifiers.get("conclusion_implied") is True
    passed   = class_ok and ci_ok

    print(f"{OK if passed else ERR}  HAMON full router")
    print(f"  classification:  {result.classification}  stage={result.stage}")
    print(f"  initial_weights: {_fmt(result.initial_weights)}")
    print(f"  modifiers: {result.modifiers}")
    print(f"  reasoning: {result.reasoning}\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", action="store_true",
                        help="Run full router test (requires Ollama)")
    args = parser.parse_args()

    results = [
        test_stage1(),
        test_modifiers(),
        test_final_weights(),
    ]

    if args.stage2:
        results.append(asyncio.run(test_full_router()))

    print("=" * 44)
    all_ok = all(results)
    print("ALL PASS" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)