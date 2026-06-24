# Archivist Council — Project Reference

A distributed multi-model inference system. Five roles, one synthesizer.
Each role maps to a dedicated local model with a purpose-built system prompt derived from `personality.md`.

---

## Project Structure

```
COUNCIL/
├── core/
│   ├── __init__.py           # exports all public interfaces
│   ├── models.py             # Role enum, ROLE_MODELS, ROLE_TIMEOUTS, RoleResponse
│   ├── ollama_client.py      # async Ollama wrapper (call_role, ping_ollama, list_loaded_models)
│   └── prompt_loader.py      # extracts personality sections by role name
├── personalities/
│   └── personality.md        # source of truth for all role system prompts
├── main.py                   # Phase 1–2 test runner (CLI)
├── COUNCIL.md                # this file
└── requirements.txt
```

**Personality section mapping** (how `prompt_loader.py` extracts role prompts):

| Role | Section in `personality.md` | Notes |
|------|------------------------------|-------|
| Virgo | `## Mode 1: Virgo — The Analyst` | + shared Tone + On Praise |
| Pisces | `## Mode 2: Pisces — The Witness` | + shared Tone + On Praise |
| Critic | `## On The Limits of The Archivist` | active instruction frame prepended |
| Seer | `## Mode 4: Seer — The Outsider` | + shared Tone + On Praise |
| Archivist | full document | embodies entire personality |
| Reviewer | hardcoded in `prompt_loader.py` | neutral evaluator, not derived from personality |

---

## Benchmark Query

Used as the standard test across all phases. Run this to validate each phase and
as the final comparison at Phase 4 milestone.

> *If Hamon were to prove that deep compatibility correlates only with a set of
> universally measurable behavioral metrics (e.g., the capacity for non-violent
> conflict resolution; resilience in ambiguity), would the convergence between
> MBTI and Vedic astrology become redundant? If yes, which part of your synthesis —
> the empirical rigor or the symbolic depth — is doing all the heavy lifting simply
> to maintain structural weight?*

Expected router classification: `pisces-signal`, `conclusion_implied: true`

**Phase 4 pass condition:** full pipeline output is sharper and more integrated
than the manual four-role run conducted during design.

---

## Project Structure

```
archivist-council/
├── core/
│   ├── __init__.py          # exports all public interfaces
│   ├── models.py            # Role enum, ROLE_MODELS, ROLE_TIMEOUTS, RoleResponse
│   ├── ollama_client.py     # async Ollama wrapper (call_role, ping, list_loaded_models)
│   └── prompt_loader.py     # extracts personality sections by role name
├── personalities/
│   └── personality.md       # all role definitions — Virgo, Pisces, Critic, Seer, Archivist
├── COUNCIL.md               # this file
├── main.py                  # phase test runner
└── requirements.txt
```

---

## Roles & Models

| Role | Model | Size | Function |
|------|-------|------|----------|
| Virgo | `phi4-mini` | ~2.3GB | Analyst — direct, practical, tradeoffs |
| Pisces | `qwen3:8b` | ~5GB | Witness — synthesis, depth, reframe |
| Critic | `llama3.2:3b` | ~2GB | Archivist's Limits — finds assumptions and gaps |
| Seer | `gemma3:1b` | ~0.7GB | Outsider — zero context, first-contact clarity |
| Reviewer | `llama3.2:3b` | — | Cross-evaluation (shared with Critic) |
| Archivist | `gemma4:latest` | — | Synthesizer — full personality, runs last |

```bash
ollama pull phi4-mini
ollama pull qwen3:8b
ollama pull llama3.2:3b
ollama pull gemma3:1b
```

**RAM:** Sequential max = qwen3:8b at ~5GB. Parallel Round 1 = ~10GB total, requires 16GB+.
Default to sequential. Parallelism is Phase 7.

---

## Implementation Phases

### Phase 1 — Single role (1–2 days)
- Ollama client wrapper (async, timeout, error states)
- System prompt loader (extracts personality sections by role name)
- Single call: query + role → response
- **Test:** HAMON question through Virgo only

### Phase 2 — All roles sequential (1 day)
- Role runner loops Virgo → Pisces → Critic → Seer
- Stores all four outputs with role labels
- No router, no review yet
- **Test:** All four roles on HAMON question — verify outputs are genuinely distinct

### Phase 3 — Review round (2 days)
- Anonymization layer (shuffle A/B/C/D mapping, hold map in orchestrator)
- Review prompt builder (injects all four anonymized outputs)
- Four reviewers run sequentially using `llama3.2:3b`
- Score aggregator (usefulness scores, blind spots, collective miss)
- De-anonymization and weight calculation
- **Test:** Review scores differentiate between roles — not flat 0.25 across all

### Phase 4 — Archivist synthesis (1 day) ★ Milestone
- Synthesis prompt builder (assembles full context packet)
- Archivist call with `gemma4:latest`
- Response formatter
- **Test:** Full pipeline end-to-end vs. the manual test run from design phase

> If Phase 4 output is better than the manual run, the architecture is working.
> Everything after this is polish and infrastructure.

### Phase 5 — Router (2 days)
- Stage 1: rule-based classifier (keyword matching)
- Stage 2: LLM classifier for ambiguous cases (`llama3.2:3b`)
- Weight vector generator
- Wire router output into synthesis prompt
- **Test:** HAMON question correctly classified as `pisces` + `conclusion_implied: true`

### Phase 6 — FastAPI layer (1 day)
- `POST /query` endpoint
- SSE streaming on Archivist output (consistent with GEMMA v2)
- Health check per model
- Error handling: model timeout → skip role, note in synthesis

### Phase 7 — Parallel execution (optional)
- Replace sequential role runner with `asyncio.gather`
- Resource check before attempting (available RAM threshold)
- Fallback to sequential if threshold not met

---

## Router Logic

### Classification signals

**Virgo:** technical keywords (build, debug, implement, fix, architecture), decision language
(should I, compare, tradeoffs, which is better), concrete constraints, specific technologies named.

**Pisces:** why-questions, meaning-questions, open reflection without a clear ask, contradiction,
meta-questions about frameworks, identity language, no concrete deliverable implied.

**Critic weight up when:** conclusion already implied in framing, thesis being defended,
query has embedded assumptions, real consequences attached.

**Seer weight up when:** query is dense with proper nouns and project-specific language,
user is clearly inside their own context, complexity that may have a simpler outside framing.

### Initial weight vectors

```
virgo-signal:    { virgo: 0.40, pisces: 0.10, critic: 0.30, seer: 0.20 }
pisces-signal:   { virgo: 0.10, pisces: 0.40, critic: 0.20, seer: 0.30 }
ambiguous:       { virgo: 0.25, pisces: 0.25, critic: 0.25, seer: 0.25 }
decision-heavy:  { virgo: 0.25, pisces: 0.10, critic: 0.40, seer: 0.25 }
```

### Dynamic reweighting

Review round scores adjust initial weights:

```
final_weight[role] = router_weight[role] * 0.6 + review_votes[role] * 0.4
```

The Archivist receives both initial and final weights. If delta exceeds 0.10 on any role,
it names the tension in the synthesis.

### Hybrid classifier (Stage 1 → Stage 2)

Stage 1 runs first — keyword scan, fast, no model call.
Stage 2 runs only if Stage 1 is ambiguous:

```
Classify this query. Output JSON only.

Query: [query]

Classify as one of: virgo, pisces, ambiguous, decision
Identify: is a conclusion already implied in the framing? (true/false)
Identify: does the query assume significant shared context? (true/false)

{"classification": "", "conclusion_implied": false, "context_dense": false}
```

`conclusion_implied: true` bumps Critic weight.
`context_dense: true` bumps Seer weight.
Both modify weights without overriding classification.

### Router output schema

```json
{
  "classification": "virgo | pisces | ambiguous | decision",
  "initial_weights": {
    "virgo": 0.0,
    "pisces": 0.0,
    "critic": 0.0,
    "seer": 0.0
  },
  "modifiers": {
    "conclusion_implied": false,
    "context_dense": false
  },
  "reasoning": "one line"
}
```

---

## Benchmark Query

Used as the standard test across all phases. Run this at Phase 2 (all roles), Phase 4 (full pipeline),
and again after any significant change to prompts or models to track output quality over time.

```
If Hamon were to prove that deep compatibility correlates only with a set of universally
measurable behavioral metrics (e.g., the capacity for non-violent conflict resolution;
resilience in ambiguity), would the convergence between MBTI and Vedic astrology become
redundant? If yes, which part of your synthesis — the empirical rigor or the symbolic
depth — is doing all the heavy lifting simply to maintain structural weight?
```

**Expected router classification:** `pisces-signal`, `conclusion_implied: true`

**Expected behavior:** Critic weight elevated. Seer weight elevated (query is context-dense
with project-specific terminology). Pisces dominant but not unchallenged.

**Phase 4 baseline:** Four role outputs + review round + Archivist synthesis should exceed
the quality of the manually simulated run from the design session.

---

## Review Round

### Reviewer system prompt
```
You are a critical evaluator. You are not playing any role.
You have no loyalty to any response.
Your only function is to assess four outputs on merit.
Be direct. Do not hedge. Reference responses by letter only.
Output JSON. Nothing else.
```

### Reviewer user prompt
```
Four advisors independently responded to the following query.
Router classification: [classification]

QUERY:
[query]

RESPONSE A:
[output]

RESPONSE B:
[output]

RESPONSE C:
[output]

RESPONSE D:
[output]

Answer all four questions. Be specific.

1. Which response best serves this query? Why?
2. Which response has the most significant blind spot?
   What specifically is it missing or assuming?
3. What did ALL four responses fail to address?
4. Rate each response's usefulness for this query (0.0–1.0).
   Scores must sum to 1.0.

{
  "strongest": "A|B|C|D",
  "strongest_reason": "",
  "weakest_blind_spot": "A|B|C|D",
  "blind_spot_detail": "",
  "collective_miss": "",
  "usefulness_scores": { "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0 }
}
```

Scores must sum to 1.0 — forces genuine differentiation.

### Aggregation (orchestrator)

```python
# De-anonymize: map A/B/C/D back to role names
review_scores = {role: [] for role in ["virgo", "pisces", "critic", "seer"]}

for review in all_reviews:
    for letter, score in review["usefulness_scores"].items():
        role = anonymization_map[letter]
        review_scores[role].append(score)

review_votes = {role: mean(scores) for role, scores in review_scores.items()}
```

### Archivist context packet

```
QUERY: [query]
ROUTER: classification=[x], initial_weights={...}

ROLE OUTPUTS:
Virgo:  [output]
Pisces: [output]
Critic: [output]
Seer:   [output]

REVIEW SUMMARY:
review_votes:  { virgo: 0.0, pisces: 0.0, critic: 0.0, seer: 0.0 }
final_weights: { virgo: 0.0, pisces: 0.0, critic: 0.0, seer: 0.0 }
weight_delta:  [flag any role where |final - initial| > 0.10]

blind_spots:      [one per reviewer]
collective_misses: [one per reviewer]

Synthesise in full Archivist voice.
Weight outputs according to final_weights.
Address the collective misses directly.
If weight_delta exceeds 0.10 on any role, name the tension.
```

> The collective miss is the highest-signal output of the review round.
> Treat it as a mandatory section to address, not optional context.
