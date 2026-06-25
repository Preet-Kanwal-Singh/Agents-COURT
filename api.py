"""
Phase 6 — FastAPI layer.

Endpoints:
  POST /query   full pipeline, SSE-streamed Archivist output
  GET  /health  Ollama reachability + per-model load status

Design constraints (from Phase 5 handoff):
  - stream_full_pipeline is the only pipeline entry point called here.
    Pipeline logic is not reconstructed in this file.
  - LaTeX stripping happens in synthesis.py (strip_latex called inside
    stream_full_pipeline on the assembled output). Nothing is stripped here.
  - The handler is a thin SSE formatter over the event generator.

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000
  uvicorn api:app --reload          # dev, auto-reloads on file save
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from core import ping_ollama, list_loaded_models, ROLE_MODELS
from core.runner import stream_full_pipeline


app = FastAPI(title="Archivist Council", version="0.6.0")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    """Serialize one event as an SSE message."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _pipeline_sse(query: str) -> AsyncGenerator[str, None]:
    """
    Wraps stream_full_pipeline in SSE wire format.
    Always emits a terminal stream_end event so clients can close cleanly
    even if an error event was the last meaningful output.
    """
    try:
        async for event in stream_full_pipeline(query):
            yield _sse(event)
    except Exception as exc:
        yield _sse({"type": "error", "message": f"Unhandled error: {exc}"})
    finally:
        yield _sse({"type": "stream_end"})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query")
async def query_endpoint(body: QueryRequest) -> StreamingResponse:
    """
    Full council pipeline with SSE streaming on Archivist output.

    Event sequence (in order):
      status     - before/after each phase (router, council, review, synthesis)
      router     - classification, initial_weights, modifiers, reasoning
      token      - one per Archivist generation token
      done       - LaTeX-stripped full synthesis text
      error      - terminal; pipeline could not complete
      stream_end - always last; signals the stream is closed

    Clients that only want the final synthesis can ignore all events except
    "done". Clients that want real-time display consume "token" events and
    then replace with "done" when it arrives (done has the cleaned version).
    """
    return StreamingResponse(
        _pipeline_sse(body.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx proxy buffering
        },
    )


@app.get("/health")
async def health() -> dict:
    """
    Ollama reachability + per-role model status.

    status:
      "ok"          - Ollama up, all role models loaded in memory
      "degraded"    - Ollama up, at least one model not yet loaded
                      (first call will load it; not an error condition)
      "unavailable" - Ollama not reachable

    loaded_models: what ollama ps currently shows in memory.
    roles: per-role model name + whether it's currently resident.
    """
    ollama_up = await ping_ollama()

    if not ollama_up:
        return {
            "status": "unavailable",
            "ollama": False,
            "loaded_models": [],
            "roles": {
                role.value: {"model": model, "loaded": False}
                for role, model in ROLE_MODELS.items()
            },
        }

    loaded = await list_loaded_models()
    loaded_set = set(loaded)

    role_status = {
        role.value: {"model": model, "loaded": model in loaded_set}
        for role, model in ROLE_MODELS.items()
    }

    all_loaded = all(s["loaded"] for s in role_status.values())

    return {
        "status": "ok" if all_loaded else "degraded",
        "ollama": True,
        "loaded_models": loaded,
        "roles": role_status,
    }
