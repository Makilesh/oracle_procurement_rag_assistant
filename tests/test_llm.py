import asyncio
import json
from typing import Any

import litellm
import pytest

import core.llm as llm_module
from core.llm import QuotaExceededError, SlidingWindowLimiter, call_structured, complete


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


def make_fake_acompletion(responses: list[Any]):
    calls: list[dict[str, Any]] = []

    async def fake(model: str, **kwargs: Any) -> Any:
        calls.append({"model": model, **kwargs})
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return fake, calls


async def test_limiter_allows_up_to_rpm() -> None:
    limiter = SlidingWindowLimiter(rpm=3)
    for _ in range(3):
        await asyncio.wait_for(limiter.acquire(), timeout=0.5)


async def test_limiter_blocks_when_full() -> None:
    limiter = SlidingWindowLimiter(rpm=2)
    await limiter.acquire()
    await limiter.acquire()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(limiter.acquire(), timeout=0.2)


async def test_call_structured_strips_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = make_fake_acompletion([FakeResponse('```json\n{"score": 4}\n```')])
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    result = await call_structured("cheap", [{"role": "user", "content": "judge"}])
    assert result == {"score": 4}
    assert calls[0]["response_format"] == {"type": "json_object"}


async def test_call_structured_retries_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _ = make_fake_acompletion(
        [FakeResponse("not json at all"), FakeResponse('{"ok": true}')]
    )
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    result = await call_structured("cheap", [{"role": "user", "content": "judge"}])
    assert result == {"ok": True}


async def test_call_structured_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = make_fake_acompletion(
        [FakeResponse("bad"), FakeResponse("worse"), FakeResponse("nope")]
    )
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    with pytest.raises(json.JSONDecodeError):
        await call_structured("cheap", [{"role": "user", "content": "judge"}])
    assert len(calls) == 3


def _rate_limit_error() -> litellm.RateLimitError:
    return litellm.RateLimitError(message="quota", llm_provider="gemini", model="x")


async def test_complete_raises_quota_after_double_429(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = make_fake_acompletion([_rate_limit_error(), _rate_limit_error()])
    monkeypatch.setattr(llm_module, "_acompletion", fake)

    async def instant_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(llm_module.asyncio, "sleep", instant_sleep)
    with pytest.raises(QuotaExceededError):
        await complete("main", [{"role": "user", "content": "hi"}])
    assert len(calls) == 2


async def test_complete_recovers_after_single_429(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _ = make_fake_acompletion([_rate_limit_error(), FakeResponse("answer")])
    monkeypatch.setattr(llm_module, "_acompletion", fake)

    async def instant_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(llm_module.asyncio, "sleep", instant_sleep)
    response = await complete("main", [{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "answer"
