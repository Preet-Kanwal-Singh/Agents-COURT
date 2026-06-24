"""
Phase 3 — Review round building blocks.

Handles anonymization (role -> letter), review prompt construction, and
parsing/validation of Reviewer's JSON output.

Deliberately kept separate from runner.py: Phase 3 has not been wired into
the live council pipeline yet. This module is meant to be exercised
standalone first, against fixed dummy responses (see test_reviewer.py),
before anonymization runs on real council output. Reviewer has zero
execution history — a malformed JSON response breaks the round, not just
degrades it, so the contract gets checked before anything depends on it.
"""
import json
import random
from dataclasses import dataclass

from json_repair import repair_json

from .models import Role, RoleResponse

LETTERS = ["A", "B", "C", "D"]


@dataclass
class AnonymizedRound:
    anonymization_map: dict[str, Role]   # "A" -> Role.VIRGO
    content_by_letter: dict[str, str]    # "A" -> response text


def anonymize(responses: list[RoleResponse], shuffle: bool = True) -> AnonymizedRound:
    """
    Maps role responses to A/B/C/D. Shuffled by default so letter position
    isn't a stable proxy for role identity across runs.

    Requires exactly 4 successful responses. Partial council runs (a role
    timed out) aren't reviewable yet — that's a Phase 6 concern per the
    brief ("model timeout -> skip role, note in synthesis"), not handled
    here. This raises rather than silently reviewing 3.
    """
    ok_responses = [r for r in responses if r.ok]
    if len(ok_responses) != 4:
        raise ValueError(
            f"Expected exactly 4 successful responses to anonymize, got "
            f"{len(ok_responses)} of {len(responses)}. Decide a skip/fill "
            f"policy before this is allowed to proceed with a partial set."
        )

    order = ok_responses[:]
    if shuffle:
        random.shuffle(order)

    anonymization_map = {LETTERS[i]: r.role for i, r in enumerate(order)}
    content_by_letter = {LETTERS[i]: r.content for i, r in enumerate(order)}
    return AnonymizedRound(anonymization_map=anonymization_map, content_by_letter=content_by_letter)


def build_review_prompt(query: str, classification: str, content_by_letter: dict[str, str]) -> str:
    """
    Builds the Reviewer's user prompt per the COUNCIL.md template.

    `classification` is router output. Phase 5 (the router) doesn't exist
    yet, so callers should pass a placeholder like "unclassified" for now
    rather than a fabricated classification — the Reviewer prompt template
    expects the field to be present, but its real value is a later-phase
    dependency, not something to fake convincingly here.
    """
    blocks = "\n\n".join(
        f"RESPONSE {letter}:\n{content_by_letter[letter]}" for letter in LETTERS
    )
    return (
        f"Four advisors independently responded to the following query.\n"
        f"Router classification: {classification}\n\n"
        f"QUERY:\n{query}\n\n"
        f"{blocks}\n\n"
        f"Answer all four questions. Be specific.\n\n"
        f"1. Which response best serves this query? Why?\n"
        f"2. Which response has the most significant blind spot?\n"
        f"   What specifically is it missing or assuming?\n"
        f"3. What did ALL four responses fail to address? Four independent\n"
        f"   perspectives on the same query almost always share at least one\n"
        f"   real gap — find it. 'None' is not an acceptable answer unless you\n"
        f"   can name a specific reason these four are genuinely complete.\n"
        f"4. Rate each response's usefulness for this query independently,\n"
        f"   0.0 (not useful) to 1.0 (extremely useful). Rate each response\n"
        f"   on its own merits — scores do not need to add up to anything.\n\n"
        f'{{\n'
        f'  "strongest": "<exactly one letter: A, B, C, or D>",\n'
        f'  "strongest_reason": "",\n'
        f'  "weakest_blind_spot": "<exactly one letter: A, B, C, or D — never more than one>",\n'
        f'  "blind_spot_detail": "",\n'
        f'  "collective_miss": "",\n'
        f'  "usefulness_scores": {{ "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0 }}\n'
        f'}}'
    )


class ReviewParseError(Exception):
    pass


def extract_json(raw: str) -> tuple[dict, bool]:
    """
    Returns (parsed_dict, was_repaired).

    was_repaired=True means the standard parser failed and json-repair
    recovered the object — worth logging so you can see the rate over time
    rather than having syntax slips silently disappear. Three distinct
    malformation shapes have appeared so far across phi4-mini and
    llama3.2:3b: literal control characters in string values, multi-letter
    combined fields, and missing comma delimiters. Patching each one
    individually was whack-a-mole; repair_json handles all of them.

    Strategy: strip code fences, find the {...} blob, try standard parse
    with strict=False (tolerates raw control chars), fall back to
    repair_json only if that fails. Raises ReviewParseError only if repair
    also produces something unusable (empty or non-dict).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ReviewParseError(f"No JSON object found in output:\n{raw[:300]}")

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate, strict=False), False
    except json.JSONDecodeError:
        pass

    repaired = repair_json(candidate, return_objects=True)
    if not isinstance(repaired, dict) or not repaired:
        raise ReviewParseError(
            f"JSON repair failed to recover a usable object.\nCandidate:\n{candidate[:300]}"
        )
    return repaired, True


REQUIRED_KEYS = {
    "strongest", "strongest_reason", "weakest_blind_spot",
    "blind_spot_detail", "collective_miss", "usefulness_scores",
}


def validate_review(parsed: dict) -> list[str]:
    """
    Returns a list of validation problems. Empty list means valid.
    Does not raise — caller decides whether a failure here is fatal
    or just logged, since during standalone testing you want to see
    every problem in one pass rather than stopping at the first.

    Note: this does NOT require usefulness_scores to sum to 1.0. Standalone
    testing showed llama3.2:3b reliably produces well-differentiated
    independent 0-1 ratings but fails to also renormalize them to sum to
    1.0 in ~80% of runs. Asking the model to do that arithmetic in its head
    was the wrong place to enforce the constraint — see normalize_scores()
    below, which does it in code instead. This validator checks the model
    only for what it's actually good at: a real score in range for every
    letter.
    """
    problems: list[str] = []

    missing = REQUIRED_KEYS - parsed.keys()
    if missing:
        problems.append(f"Missing keys: {missing}")

    for letter_key in ("strongest", "weakest_blind_spot"):
        if letter_key in parsed and parsed[letter_key] not in LETTERS:
            problems.append(
                f"'{letter_key}' = {parsed[letter_key]!r}, expected exactly one of {LETTERS} "
                f"(if this contains a '|', the model combined multiple letters)"
            )

    scores = parsed.get("usefulness_scores")
    if not isinstance(scores, dict):
        problems.append(f"'usefulness_scores' is not a dict: {scores!r}")
    else:
        missing_letters = set(LETTERS) - scores.keys()
        if missing_letters:
            problems.append(f"usefulness_scores missing letters: {missing_letters}")
        else:
            try:
                values = {l: float(scores[l]) for l in LETTERS}
            except (TypeError, ValueError):
                problems.append(f"usefulness_scores contains non-numeric values: {scores!r}")
            else:
                out_of_range = {l: v for l, v in values.items() if not (0.0 <= v <= 1.0)}
                if out_of_range:
                    problems.append(f"usefulness_scores out of [0.0, 1.0] range: {out_of_range}")
                if all(v == 0.0 for v in values.values()):
                    problems.append("usefulness_scores are all 0.0 — nothing to normalize")

    return problems


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """
    Redistributes raw 0-1 usefulness ratings to sum to 1.0, preserving
    relative differentiation. This is the code-level replacement for the
    "scores must sum to 1.0" instruction that the model was failing to
    follow ~80% of the time — the model differentiates responses fine on
    an absolute scale, so dividing by the total here recovers the same
    forced-differentiation effect the brief wanted, without relying on
    the model to do the arithmetic itself.
    """
    total = sum(scores.values())
    if total == 0:
        # Already caught by validate_review, but guard here too in case
        # this is ever called without validation run first.
        return {letter: 1.0 / len(scores) for letter in scores}
    return {letter: value / total for letter, value in scores.items()}


_LAZY_NON_ANSWERS = {"none", "n/a", "na", "nothing", "no blind spots", "nothing significant"}


def flag_quality_issues(parsed: dict) -> list[str]:
    """
    Non-fatal content checks — these don't fail validation, because a
    syntactically valid 'None' is still schema-correct. They exist because
    9 of 10 standalone runs returned a literal non-answer for
    collective_miss, which is the field the brief calls the highest-signal
    output of the review round. Schema validation can't see this kind of
    failure; it has to be checked separately and reported as a warning so
    it doesn't get lost next to genuine passes.
    """
    flags: list[str] = []

    miss = str(parsed.get("collective_miss", "")).strip().lower()
    if miss in _LAZY_NON_ANSWERS or len(miss) < 15:
        flags.append(
            f"collective_miss looks like a non-answer ({parsed.get('collective_miss')!r}) "
            f"rather than a specific gap — worth a closer look before trusting this field."
        )

    return flags


def deanonymize_scores(scores: dict[str, float], anonymization_map: dict[str, Role]) -> dict[str, float]:
    return {anonymization_map[letter].value: score for letter, score in scores.items()}


@dataclass
class ReviewResult:
    """
    Everything produced by a single Reviewer pass, in one object.

    strongest/weakest_blind_spot are role names (deanonymized), not letters.
    role_scores maps role name -> normalized weight (sums to 1.0).
    quality_flags is non-empty when collective_miss looks like a non-answer.
    was_repaired=True means the model emitted malformed JSON that required
    json-repair to recover — valid to use, worth knowing about.
    error is set if the Reviewer call itself failed or produced unparseable
    output even after repair; in that case all other fields are empty/None.
    """
    strongest: str | None
    strongest_reason: str
    weakest_blind_spot: str | None
    blind_spot_detail: str
    collective_miss: str
    role_scores: dict[str, float]        # role name -> normalized weight
    quality_flags: list[str]
    was_repaired: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class AggregatedReview:
    """
    Aggregated output of four independent Reviewer passes.

    Each pass uses a fresh shuffle so positional bias can't accumulate
    across reviewers — if all four saw the same letter order, consistent
    score patterns could reflect position rather than content.

    review_votes: mean normalized score per role across all successful
    passes, summing to 1.0. This is the weight vector that feeds into
    the Archivist context packet. The formula
        final_weight[role] = router_weight * 0.6 + review_votes * 0.4
    is a Phase 5 concern — router_weight doesn't exist yet, so
    review_votes is the complete output for now.

    blind_spots / collective_misses: one entry per successful reviewer
    pass, in run order. The brief specifies these as lists, not summaries,
    so the Archivist receives all of them and decides what to weight.

    error_count > 0 means some passes failed entirely and review_votes
    is computed from fewer than four samples. Worth surfacing but not
    fatal — partial aggregation is better than no review round.
    """
    reviews: list[ReviewResult]
    review_votes: dict[str, float]      # role -> mean normalized score, sums to 1.0
    strongest_counts: dict[str, int]    # role -> times named strongest across passes
    blind_spot_counts: dict[str, int]   # role -> times named weakest blind spot
    blind_spots: list[str]              # blind_spot_detail per successful pass
    collective_misses: list[str]        # collective_miss per successful pass
    quality_flag_count: int             # passes where collective_miss was a non-answer
    repair_count: int                   # passes where json-repair was needed
    error_count: int                    # passes that failed entirely

    @property
    def ok(self) -> bool:
        return self.error_count < len(self.reviews)  # at least one pass succeeded
