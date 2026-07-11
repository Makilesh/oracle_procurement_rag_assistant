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
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Literal

import litellm

from core import metrics
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

Role = Literal["main", "cheap", "judge"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class QuotaExceededError(Exception):
    """LLM quota exhausted after retries (routes map this to a clean 503)."""


class DailyBudgetExceeded(Exception):
    """Internal: a model's client-side RPD budget is spent — skip it, never wait."""


def _is_rate_limit(exc: Exception) -> bool:
    """Provider 429s don't always surface as litellm.RateLimitError (e.g.
    VertexAIError wrapping a raw 429), so also match on the error text."""
    if isinstance(exc, litellm.RateLimitError):
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "Too Many Requests" in text


class DailyCounter:
    """Durable RPD accounting in Redis, keyed (model#key, Google-reset date).
    Counts survive api restarts and are shared across replicas. If Redis is
    unreachable (unit tests, bare local runs) it degrades to an in-process
    rolling 24h window — same semantics, weaker durability."""

    def __init__(self) -> None:
        self._redis: Any = None
        self._disabled_until = 0.0
        self._fallback: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    def _client(self) -> Any:
        if self._redis is None:
            from redis.asyncio import from_url

            self._redis = from_url(settings.redis_url, decode_responses=True)
        return self._redis

    @staticmethod
    def _date() -> str:
        """Google free-tier quotas reset at midnight Pacific."""
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d")
        except Exception:
            return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-8))).strftime(
                "%Y%m%d"
            )

    async def within_budget(self, slot: str, limit: int) -> bool:
        """Atomically count one request against the slot's daily budget.
        INCR-first is deliberately conservative: rejected attempts inflate the
        count slightly rather than ever under-counting."""
        if time.monotonic() >= self._disabled_until:
            key = f"llm_rpd:{slot}:{self._date()}"
            try:
                client = self._client()
                count = await client.incr(key)
                if count == 1:
                    await client.expire(key, 26 * 3600)
                return count <= limit
            except Exception:
                logger.warning("redis unavailable for rpd accounting; using in-process fallback")
                self._disabled_until = time.monotonic() + 60.0
        async with self._lock:
            now = time.monotonic()
            window = self._fallback.setdefault(slot, deque())
            while window and now - window[0] > 86400.0:
                window.popleft()
            if len(window) >= limit:
                return False
            window.append(now)
            return True


_daily = DailyCounter()


class ModelBudget:
    """Per-(model, key) budget. RPM pacing is in-process (asyncio lock + deque
    of monotonic stamps; the lock is RELEASED before sleeping so one sleeper
    never serializes concurrent requests). RPD accounting is durable in Redis
    via DailyCounter; a spent RPD budget raises DailyBudgetExceeded immediately
    — waiting hours makes no sense when a fallback model is available."""

    def __init__(self, slot: str, rpm: int, rpd: int) -> None:
        self.slot = slot
        self.rpm = rpm
        self.rpd = rpd
        self._lock = asyncio.Lock()
        self._minute: deque[float] = deque()

    async def _try_acquire(self) -> float | None:
        """None = acquired; float = seconds to wait; raises when RPD is spent."""
        async with self._lock:
            now = time.monotonic()
            while self._minute and now - self._minute[0] > 60.0:
                self._minute.popleft()
            if len(self._minute) >= self.rpm:
                return 60.0 - (now - self._minute[0]) + 0.05
            self._minute.append(now)
        if await _daily.within_budget(self.slot, self.rpd):
            return None
        raise DailyBudgetExceeded

    async def acquire(self) -> None:
        while True:
            wait = await self._try_acquire()
            if wait is None:
                return
            logger.info("rpm limiter: waiting %.1fs", wait)
            await asyncio.sleep(wait)


# Free-tier budgets (RPM, RPD) verified in this project's AI Studio rate-limit
# dashboard on 2026-07-11. Unknown models get a conservative default.
_FREE_TIER_BUDGETS: dict[str, tuple[int, int]] = {
    "gemini/gemini-3.5-flash": (5, 20),
    "gemini/gemini-3.1-flash-lite": (15, 500),
    "gemini/gemini-3-flash-preview": (5, 20),
    "gemini/gemini-2.5-flash": (5, 20),
    "gemini/gemini-2.5-flash-lite": (10, 20),
}
_DEFAULT_BUDGET: tuple[int, int] = (5, 20)

_budgets: dict[str, ModelBudget] = {}

_models: dict[Role, str] = {
    # ⚠ VERIFY — confirmed against ai.google.dev model list + LiteLLM Gemini
    # provider docs (July 2026): all are live model codes.
    "main": settings.model_main,
    "cheap": settings.model_cheap,
    "judge": settings.model_judge,
}


def api_keys() -> list[str]:
    """Rotation list (first = primary). Each key is its own project/quota."""
    keys = [k.strip() for k in settings.gemini_api_keys.split(",") if k.strip()]
    return keys or [settings.gemini_api_key]


def budget_for(model: str, key_index: int) -> ModelBudget:
    """Budgets are per (model, key): every key is a separate quota project."""
    slot = f"{model}#{key_index}"
    if slot not in _budgets:
        rpm, rpd = _FREE_TIER_BUDGETS.get(model, _DEFAULT_BUDGET)
        if model == settings.model_main:
            rpm, rpd = settings.rpm_main, settings.rpd_main
        elif model == settings.model_cheap:
            rpm, rpd = settings.rpm_cheap, settings.rpd_cheap
        _budgets[slot] = ModelBudget(slot, rpm, rpd)
    return _budgets[slot]


def _chain(role: Role) -> list[str]:
    """Primary model for the role, then the shared fallback chain (best first)."""
    primary = _models[role]
    fallbacks = [m.strip() for m in settings.model_fallbacks.split(",") if m.strip()]
    return [primary] + [m for m in fallbacks if m != primary]


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
    """One chat completion through the per-(model, key) budgets with automatic
    fallback. A model is exhausted across EVERY api key (each key = its own
    quota project) before stepping down to the next model in the chain:
    primary(key1..N) → fallback1(key1..N) → ... → optional Ollama → clean 503.
    Returns a ModelResponse, or an async stream when stream=True."""
    rate_limited = False
    first_error: Exception | None = None
    keys = api_keys()
    for model in _chain(role):
        for key_index, key in enumerate(keys):
            try:
                await budget_for(model, key_index).acquire()
            except DailyBudgetExceeded:
                logger.warning("daily budget spent for %s key[%d]; rotating", model, key_index)
                metrics.LLM_CALLS.labels(model, "day_budget_spent").inc()
                rate_limited = True
                continue
            try:
                response = await _acompletion(
                    model,
                    messages=messages,
                    stream=stream,
                    timeout=timeout,
                    api_key=key or None,
                    **kwargs,
                )
                metrics.LLM_CALLS.labels(model, "ok").inc()
                return response
            except Exception as exc:
                if _is_rate_limit(exc):
                    logger.warning("upstream 429 from %s key[%d]; rotating", model, key_index)
                    metrics.LLM_CALLS.labels(model, "rate_limited").inc()
                    rate_limited = True
                    continue
                if first_error is None:
                    first_error = exc
                logger.warning("model %s failed (%s); next model", model, type(exc).__name__)
                metrics.LLM_CALLS.labels(model, "error").inc()
                break  # non-quota errors repeat on every key — skip to next model

    if settings.ollama_fallback_enabled:
        logger.warning("all Gemini models exhausted; falling back to ollama/%s", settings.ollama_model)
        return await _acompletion(
            f"ollama/{settings.ollama_model}",
            messages=messages,
            stream=stream,
            timeout=max(timeout, 60.0),
            api_base=settings.ollama_base_url,
            **kwargs,
        )
    if rate_limited or first_error is None:
        metrics.LLM_QUOTA_EXHAUSTED.inc()
        raise QuotaExceededError("LLM quota exceeded, retry shortly")
    raise first_error


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
