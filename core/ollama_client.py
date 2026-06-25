import re
import time
from typing import AsyncGenerator

from ollama import AsyncClient, ResponseError
import httpx

from .models import Role, ROLE_MODELS, ROLE_TIMEOUTS, ROLE_NUM_PREDICT, ROLE_KEEP_ALIVE, RoleResponse

OLLAMA_BASE = "http://localhost:11434"

# Qwen3 and some Gemma models emit <think>...</think> blocks.
# Strip them before returning content — consistent with GEMMA v2 pipeline.
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# Roles where we explicitly pass think=False to the Ollama API.
# Gemma4 (Archivist) will burn its entire num_predict budget on <think> blocks
# when given a large context packet, producing zero visible output.
_THINKING_DISABLED: set[Role] = {Role.ARCHIVIST}


def _strip_thinking(text: str) -> str:
    stripped = _THINK_PATTERN.sub("", text).strip()

    if "<think>" in stripped:
        after = stripped.split("</think>", 1)
        if len(after) > 1:
            return after[1].strip()
        return ""

    return stripped


async def call_role(
    role: Role,
    system_prompt: str,
    query: str,
) -> RoleResponse:
    """
    Send a query to the model assigned to a given role.
    Returns a RoleResponse — never raises. Errors are captured in .error.
    """
    model = ROLE_MODELS[role]
    timeout = ROLE_TIMEOUTS[role]
    start = time.monotonic()

    try:
        client = AsyncClient(timeout=timeout)
        options = {"temperature": 0.7, "num_predict": ROLE_NUM_PREDICT[role]}

        chat_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            options=options,
            keep_alive=ROLE_KEEP_ALIVE[role],
        )

        if role in _THINKING_DISABLED:
            chat_kwargs["think"] = False

        response = await client.chat(**chat_kwargs)
        content = _strip_thinking(response.message.content)
        duration_ms = int((time.monotonic() - start) * 1000)

        if not content:
            return RoleResponse(
                role=role, model=model, content="", duration_ms=duration_ms,
                error="Model exhausted context window during thinking — no response generated."
            )

        return RoleResponse(role=role, model=model, content=content, duration_ms=duration_ms)

    except ConnectionError:
        return RoleResponse(role=role, model=model, content="", duration_ms=0,
            error="Cannot reach Ollama. Is it running? -> ollama serve")
    except ResponseError as e:
        return RoleResponse(role=role, model=model, content="", duration_ms=0,
            error=f"Ollama error {e.status_code}: {e.error}")
    except httpx.ConnectError:
        return RoleResponse(role=role, model=model, content="", duration_ms=0,
            error="Cannot reach Ollama. Is it running? -> ollama serve")
    except httpx.TimeoutException:
        return RoleResponse(role=role, model=model, content="", duration_ms=0,
            error=f"Timeout after {timeout}s — model may need more time or is not loaded.")
    except httpx.HTTPStatusError as e:
        return RoleResponse(role=role, model=model, content="", duration_ms=0,
            error=f"HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        return RoleResponse(role=role, model=model, content="", duration_ms=0,
            error=f"Unexpected error: {type(e).__name__}: {e}")


async def stream_role(
    role: Role,
    system_prompt: str,
    query: str,
) -> AsyncGenerator[str, None]:
    """
    Streaming variant of call_role. Yields raw content tokens as the model generates.

    Intended for Archivist use only — council and review roles run via call_role.
    think=False is applied for roles in _THINKING_DISABLED (Archivist is in this set),
    so <think> blocks won't appear in the stream. No think-stripping is applied
    per-token because the suppression happens at the model level.

    Never raises — errors are yielded as a terminal error-marker token so the
    accumulator in the caller still gets a non-empty response it can inspect.
    """
    model = ROLE_MODELS[role]
    timeout = ROLE_TIMEOUTS[role]

    try:
        client = AsyncClient(timeout=timeout)
        options = {"temperature": 0.7, "num_predict": ROLE_NUM_PREDICT[role]}

        chat_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            options=options,
            keep_alive=ROLE_KEEP_ALIVE[role],
            stream=True,
        )

        if role in _THINKING_DISABLED:
            chat_kwargs["think"] = False

        async for chunk in await client.chat(**chat_kwargs):
            token = chunk.message.content
            if token:
                yield token

    except httpx.TimeoutException:
        yield f"\n[stream timeout after {timeout}s — model may need a longer ROLE_TIMEOUTS value]"
    except httpx.ConnectError:
        yield "\n[stream error: cannot reach Ollama — is it running?]"
    except ResponseError as e:
        yield f"\n[stream error: Ollama {e.status_code}: {e.error}]"
    except Exception as exc:
        yield f"\n[stream error: {type(exc).__name__}: {exc}]"


async def ping_ollama() -> bool:
    try:
        await AsyncClient().list()
        return True
    except Exception:
        return False


async def list_loaded_models() -> list[str]:
    try:
        running = await AsyncClient().ps()
        return [m.model for m in running.models]
    except Exception:
        return []
