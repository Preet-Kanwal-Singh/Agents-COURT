"""
Phase 5 — Query router.
Stage 1: rule-based keyword classifier (no model call).
Stage 2: LLM classifier (llama3.2:3b), runs only if Stage 1 returns 'ambiguous'.
final_weights are NOT computed here — call compute_final_weights() after review_votes are known.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import ollama
from json_repair import repair_json

from .models import ROUTER_MODEL, ROUTER_TIMEOUT


# ── Weight tables ─────────────────────────────────────────────────────────────

INITIAL_WEIGHT_VECTORS: dict[str, dict[str, float]] = {
    "virgo":    {"virgo": 0.40, "pisces": 0.10, "critic": 0.30, "seer": 0.20},
    "pisces":   {"virgo": 0.10, "pisces": 0.40, "critic": 0.20, "seer": 0.30},
    "ambiguous":{"virgo": 0.25, "pisces": 0.25, "critic": 0.25, "seer": 0.25},
    "decision": {"virgo": 0.25, "pisces": 0.10, "critic": 0.40, "seer": 0.25},
}

MODIFIER_BUMP: float = 0.05
VALID_CLASSIFICATIONS: frozenset[str] = frozenset({"virgo", "pisces", "ambiguous", "decision"})
MIN_SCORE: int = 2  # keyword hits required for a non-ambiguous Stage 1 result


# ── RouterResult ──────────────────────────────────────────────────────────────

@dataclass
class RouterResult:
    classification: str
    initial_weights: dict[str, float]  # base vector + modifiers applied
    modifiers: dict[str, bool]
    reasoning: str = ""
    stage: int = 1  # 1 = keyword-only, 2 = LLM used


# ── Weight helpers ────────────────────────────────────────────────────────────

def _bump_role(weights: dict[str, float], target: str, amount: float) -> dict[str, float]:
    """Bump target by amount; redistribute deficit proportionally from others. Sum-preserving."""
    weights = dict(weights)
    new_val = min(1.0, weights.get(target, 0.0) + amount)
    actual = new_val - weights.get(target, 0.0)
    if actual <= 0:
        return weights
    weights[target] = new_val
    others = {r: w for r, w in weights.items() if r != target}
    total = sum(others.values())
    if total <= 0:
        return weights
    for role, w in others.items():
        weights[role] = max(0.0, w - (w / total) * actual)
    return weights


def apply_modifiers(weights: dict[str, float], modifiers: dict[str, bool]) -> dict[str, float]:
    """Apply conclusion_implied (Critic bump) and context_dense (Seer bump)."""
    if modifiers.get("conclusion_implied"):
        weights = _bump_role(weights, "critic", MODIFIER_BUMP)
    if modifiers.get("context_dense"):
        weights = _bump_role(weights, "seer", MODIFIER_BUMP)
    return weights


def compute_final_weights(
    initial_weights: dict[str, float],
    review_votes: dict[str, float],
) -> dict[str, float]:
    """final[role] = initial[role] * 0.6 + review_votes[role] * 0.4"""
    raw = {
        role: initial_weights.get(role, 0.0) * 0.6 + review_votes.get(role, 0.0) * 0.4
        for role in initial_weights
    }
    total = sum(raw.values())
    return {role: v / total for role, v in raw.items()} if total > 0 else raw


def get_weight_deltas(
    initial_weights: dict[str, float],
    final_weights: dict[str, float],
    threshold: float = 0.10,
) -> dict[str, float]:
    """Returns {role: delta} for roles where |final - initial| > threshold."""
    return {
        role: round(abs(final_weights.get(role, 0.0) - w), 4)
        for role, w in initial_weights.items()
        if abs(final_weights.get(role, 0.0) - w) > threshold
    }


# ── Stage 1: keyword patterns ─────────────────────────────────────────────────

_VIRGO = [
    r"\bbuild\b", r"\bdebug\b", r"\bimplement\b", r"\bfix\b",
    r"\barchitecture\b", r"\bdeploy\b", r"\binstall\b", r"\bconfigure\b",
    r"\brefactor\b", r"\boptimize\b", r"\bperformance\b", r"\blatency\b",
    r"\berror\b", r"\bbug\b", r"\bcode\b", r"\bapi\b",
    r"\bfunction\b", r"\bmodule\b", r"\blibrary\b", r"\bsyntax\b",
    r"\bhow (do i|to)\b",
]

_PISCES = [
    r"\bwhy\b",
    r"\bwhat does (this|it) mean\b",
    r"\bmeaning\b",
    r"\breflect\w*\b",
    r"\bphilosoph\w*\b",
    r"\bidentity\b",
    r"\bcontradiction\b",
    r"\bparadox\b",
    r"\bsymbol\w*\b",       # catches "symbolic"
    r"\bconvergence\b",
    r"\bsynthesis\b",
    r"\brigou?r\b",          # catches "rigor" and "rigour"
    r"\bdepth\b",
    r"\bontolog\w*\b",
    r"\bepistemi\w*\b",
]

_DECISION = [
    r"\bshould i\b",
    r"\bdecide\b", r"\bdecision\b",
    r"\bchoose\b", r"\bchoice\b",
    r"\bwhich (option|approach|path|one)\b",
    r"\brecommend\b",
    r"\bpros and cons\b",
    r"\btradeoff\b", r"\btrade-off\b",
    r"\bbetter for\b", r"\bbest for\b",
]

_CONCLUSION_IMPLIED = [
    r"\bif (yes|so|true)\b",
    r"\bdoes(n'?t)? (this|that) mean\b",
    r"\bisn'?t (it|this|that)\b",
    r"\bheavy lifting\b",
    r"\bproves? that\b",
    r"\bwhich means\b",
    r"\btherefore\b",
    r"\bimplies? that\b",
    r"\bwould\b.{0,100}\bredundant\b",
    r"\bwould\b.{0,100}\bmoot\b",
    r"\bwhich (part|side|element)\b.{0,60}\bdoing\b",
]


def _score(q: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, q))


def _detect_conclusion_implied(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _CONCLUSION_IMPLIED)


def _detect_context_dense(query: str) -> bool:
    """True if query has >=2 proper nouns or acronyms an outsider wouldn't know."""
    candidates: set[str] = set()
    candidates.update(re.findall(r'\b[A-Z]{2,}\b', query))          # MBTI, API, etc.
    for sentence in re.split(r'(?<=[.!?])\s+', query):
        words = sentence.split()
        for word in words[1:]:                                        # skip sentence-initial cap
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() != clean:
                candidates.add(clean)
    return len(candidates) >= 2


def classify_stage1(query: str) -> tuple[str, dict[str, bool], str]:
    """
    Rule-based classifier. No model calls.
    Returns (classification, modifiers, reasoning).
    """
    q = query.lower()
    scores = {
        "virgo":    _score(q, _VIRGO),
        "pisces":   _score(q, _PISCES),
        "decision": _score(q, _DECISION),
    }
    modifiers = {
        "conclusion_implied": _detect_conclusion_implied(query),
        "context_dense":      _detect_context_dense(query),
    }

    top_score = max(scores.values())

    if top_score < MIN_SCORE:
        return "ambiguous", modifiers, f"no signal clears threshold — scores={scores}"

    top = [k for k, v in scores.items() if v == top_score]

    if len(top) > 1:
        # virgo wins a virgo/decision tie — technical context dominates
        if set(top) == {"virgo", "decision"}:
            return "virgo", modifiers, f"virgo wins tie over decision (score={top_score})"
        return "ambiguous", modifiers, f"tied: { {k: scores[k] for k in top} }"

    winner = top[0]
    second = sorted(scores.values(), reverse=True)[1]
    return winner, modifiers, f"stage1 {winner}={top_score} next={second}"


# ── Stage 2: LLM classifier ───────────────────────────────────────────────────

_S2_SYSTEM = "You are a query classifier. Output JSON only. No preamble. No explanation."


def _s2_user(query: str) -> str:
    return f"""\
Classify this query. Output exactly this JSON and nothing else:
{{"classification": "virgo|pisces|decision|ambiguous", "conclusion_implied": false, "context_dense": false, "reasoning": "one line"}}

Categories:
- virgo: technical, implementation, debugging, how-to, performance, system design
- pisces: philosophical, meaning, open reflection, framework or identity questions, no concrete deliverable
- decision: explicit choice between named options or a direct tradeoff request
- ambiguous: no clear dominant signal

conclusion_implied: true if the query framing already steers toward a specific answer
context_dense: true if the query uses project-specific terms, proper nouns, or acronyms an outsider would not know

QUERY: {query}"""


async def _classify_stage2(query: str) -> tuple[str, dict[str, bool], str]:
    client = ollama.AsyncClient(timeout=ROUTER_TIMEOUT)
    try:
        resp = await client.chat(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _S2_SYSTEM},
                {"role": "user",   "content": _s2_user(query)},
            ],
            options={"num_predict": 128, "temperature": 0},
            think=False,   # top-level, not inside options
        )
        raw = resp["message"]["content"].strip()
    except Exception as e:
        return "ambiguous", {"conclusion_implied": False, "context_dense": False}, f"stage2 error: {e}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(repair_json(raw))
        except Exception:
            return "ambiguous", {"conclusion_implied": False, "context_dense": False}, "stage2 parse failed"

    classification = data.get("classification", "ambiguous")
    if classification not in VALID_CLASSIFICATIONS:
        classification = "ambiguous"

    modifiers = {
        "conclusion_implied": bool(data.get("conclusion_implied", False)),
        "context_dense":      bool(data.get("context_dense", False)),
    }
    return classification, modifiers, f"stage2: {data.get('reasoning', '').strip()}"


# ── Public entry point ────────────────────────────────────────────────────────

async def run_router(query: str) -> RouterResult:
    stage = 1
    classification, modifiers, reasoning = classify_stage1(query)

    if classification == "ambiguous":
        classification, modifiers, reasoning = await _classify_stage2(query)
        stage = 2

    base = INITIAL_WEIGHT_VECTORS.get(classification, INITIAL_WEIGHT_VECTORS["ambiguous"]).copy()
    initial_weights = apply_modifiers(base, modifiers)

    return RouterResult(
        classification=classification,
        initial_weights=initial_weights,
        modifiers=modifiers,
        reasoning=reasoning,
        stage=stage,
    )