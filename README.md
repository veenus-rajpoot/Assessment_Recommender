# SHL Assessment Recommender — scaffold

## What's here and what's been tested

Tested and working in this sandbox (no network access here, so this is as
far as I could verify):
- `build_catalog.py` — cleaned all 377 catalog items successfully.
- `retrieval.py` — TF-IDF + metadata-filter search, verified with
  `tests/test_retrieval_offline.py` (6/6 checks pass, zero deps beyond
  scikit-learn/numpy, no LLM or network needed).

Written but **not yet run**, because this sandbox has no internet access and
doesn't have `fastapi`/`pydantic`/`openai` installed (only stdlib + numpy +
scikit-learn were available):
- `schemas.py`, `llm.py`, `agent.py`, `main.py`

You'll need to actually run these yourself and fix whatever breaks on
first contact — treat this as a strong first draft of the conversation
logic, not a finished, battle-tested agent. In particular:
- I have not seen the 10 provided conversation traces (only the assignment
  PDF describes them) — you should read them and check the controller
  prompt's `has_enough_context` behavior against real examples, especially
  around what counts as "enough" to recommend vs. clarify.
- The controller LLM call's JSON output isn't validated against a strict
  schema before use (I do a best-effort `json.loads` and catch failures,
  but a model that returns almost-valid JSON with a stray field won't be
  caught structurally). If you want tighter guarantees, define a pydantic
  model for the controller's output and pass it through `client.chat.completions.create(response_format=...)` (Groq/OpenRouter support this for some models) instead of freeform JSON.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 build_catalog.py          # regenerates data/catalog_clean.json
cp .env.example .env              # fill in your LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
export $(cat .env | xargs)
```

## Run locally

```bash
uvicorn main:app --reload --port 8000
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

## Test

```bash
python3 tests/test_retrieval_offline.py   # no API key needed, run this first
# then, once traces are downloaded from the assignment link:
# write a small script that POSTs each trace's turns to /chat and checks
# the final recommendations against the trace's labeled expected shortlist
# using the Recall@K formula from the assignment (see appendix in the PDF).
```

## Deploy

Any ASGI-friendly free host works (Render, Fly, Railway, Modal, HF Spaces).
Render example: `uvicorn main:app --host 0.0.0.0 --port $PORT` as the start
command, `pip install -r requirements.txt` as the build command, and set
the three `LLM_*` env vars plus expect the documented cold-start delay.

## Architecture notes (for the approach doc)

- **Retrieval**: TF-IDF over catalog `search_text`, not embeddings — see
  the docstring in `retrieval.py` for the reasoning (corpus is tiny,
  keyword-exactness matters for product names, avoids an extra network hop
  inside the 30s budget). Metadata (duration, remote) are hard filters;
  test_type/job_level are soft boosts, since conversational signal on those
  is often implicit.
- **Hallucination defense**: the `recommendations` array is always built by
  looking up real `Catalog` objects (`_clamp_recommendations`), never
  parsed from LLM-generated text. The LLM only classifies intent, extracts
  constraints, and writes prose around a candidate list Python already
  chose. This is the main structural answer to the "hallucination" behavior
  probe.
- **Statelessness**: every `/chat` call re-derives constraints from the
  *entire* message history (not just the latest turn), which is what makes
  "refine" work correctly without any server-side session state — a later
  statement naturally overrides an earlier one because the controller
  prompt is instructed to resolve conflicts in favor of the later turn.
- **Turn cap**: `is_final_allowed_turn` in `agent.py` forces a best-effort
  recommendation instead of another clarifying question once the transcript
  is one exchange away from the 8-turn cap, so the agent can't get stuck
  clarifying forever and blow the hard eval.
- **Known gap**: no re-ranking beyond TF-IDF score + additive boosts. If
  Recall@10 comes in low on the provided traces, the highest-leverage next
  step is probably better constraint extraction (e.g. explicitly mapping
  colloquial seniority phrases to the catalog's `job_levels` vocabulary)
  before reaching for a heavier retrieval method.
# Assessment_Recommender
