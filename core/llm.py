"""LiteLLM wrapper: model roles, retries, timeouts, client-side RPM limiting,
JSON-mode helper, and the optional env-gated Ollama fallback.

The sliding-window limiter enforces Gemini's own free-tier RPM limits
client-side so we rarely see upstream 429s at all.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from typing import Any, AsyncIterator, Literal

import litellm

from core.config import settings

logger = logging.getLogger("llm")

# LiteLLM reads the key from the process env; make sure a .env-file-only
# configuration still works.
if settings.gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

litellm.suppress_debug_info = True
# The aiohttp transport intermittently drops Gemini streaming connections
# ("Server disconnected"); the httpx transport is reliable.
litellm.disable_aiohttp_transport = True

Role = Literal["main", "cheap"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class QuotaExceededError(Exception):
    """LLM quota exhausted after retries (routes map this to a clean 503)."""


class SlidingWindowLimiter:
    """Per-model RPM limiter: asyncio lock + deque of monotonic timestamps.
    The lock is RELEASED before sleeping, then re-acquired and re-checked, so
    one sleeper never serializes concurrent requests."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._lock = asyncio.Lock()
        self._stamps: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] > 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self._rpm:
                    self._stamps.append(now)
                    return
                wait = 60.0 - (now - self._stamps[0]) + 0.05
            logger.info("rpm limiter: waiting %.1fs", wait)
            await asyncio.sleep(wait)


_limiters: dict[Role, SlidingWindowLimiter] = {
    "main": SlidingWindowLimiter(settings.rpm_main),
    "cheap": SlidingWindowLimiter(settings.rpm_cheap),
}

_models: dict[Role, str] = {
    # ⚠ VERIFY — confirmed against ai.google.dev model list + LiteLLM Gemini
    # provider docs (July 2026): both are stable model codes.
    "main": settings.model_main,
    "cheap": settings.model_cheap,
}


async def _acompletion(model: str, **kwargs: Any) -> Any:
    return await litellm.acompletion(model=model, num_retries=2, **kwargs)


async def complete(
    role: Role,
    messages: list[dict[str, str]],
    *,
    stream: bool = False,
    timeout: float = 30.0,
    **kwargs: Any,
) -> Any:
    """One chat completion through the rate limiter with graceful 429 handling.
    Returns a ModelResponse, or an async stream when stream=True."""
    await _limiters[role].acquire()
    model = _models[role]
    try:
        return await _acompletion(model, messages=messages, stream=stream, timeout=timeout, **kwargs)
    except litellm.RateLimitError:
        logger.warning("upstream 429 from %s despite limiter; backing off once", model)
        await asyncio.sleep(2.0)
        try:
            return await _acompletion(
                model, messages=messages, stream=stream, timeout=timeout, **kwargs
            )
        except litellm.RateLimitError:
            if settings.ollama_fallback_enabled:
                logger.warning("falling back to ollama/%s", settings.ollama_model)
                return await _acompletion(
                    f"ollama/{settings.ollama_model}",
                    messages=messages,
                    stream=stream,
                    timeout=max(timeout, 60.0),
                    api_base=settings.ollama_base_url,
                    **kwargs,
                )
            raise QuotaExceededError("LLM quota exceeded, retry shortly")


def response_text(response: Any) -> str:
    return response.choices[0].message.content or ""


async def stream_deltas(stream: Any) -> AsyncIterator[str]:
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


async def call_structured(
    role: Role,
    messages: list[dict[str, str]],
    *,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """JSON-mode call: response_format json_object, fence-stripping, and up to
    2 retries on JSONDecodeError. Never regex-extracts JSON from prose."""
    last_error: Exception | None = None
    for attempt in range(3):
        response = await complete(
            role,
            messages,
            timeout=timeout,
            response_format={"type": "json_object"},
        )
        text = _strip_fences(response_text(response))
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            last_error = ValueError(f"expected JSON object, got {type(parsed).__name__}")
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("structured call returned invalid JSON (attempt %d)", attempt + 1)
    raise last_error if last_error else RuntimeError("structured call failed")
