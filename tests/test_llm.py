import asyncio
import json
import time
from typing import Any

import litellm
import pytest

import core.llm as llm_module
from core.llm import (
    DailyBudgetExceeded,
    ModelBudget,
    QuotaExceededError,
    call_structured,
    complete,
)


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


@pytest.fixture(autouse=True)
def fresh_budgets(monkeypatch: pytest.MonkeyPatch):
    """Isolate budget state; pin one fallback model and one api key per test.
    Daily accounting is forced onto the in-process fallback (no Redis in unit
    tests) and its counters are reset so tests can't leak into each other."""
    llm_module._budgets.clear()
    monkeypatch.setattr(llm_module._daily, "_disabled_until", time.monotonic() + 3600)
    monkeypatch.setattr(llm_module._daily, "_fallback", {})
    monkeypatch.setattr(llm_module.settings, "model_fallbacks", "gemini/fallback-model")
    monkeypatch.setattr(llm_module.settings, "gemini_api_keys", "key-a")
    yield
    llm_module._budgets.clear()


async def test_budget_allows_up_to_rpm() -> None:
    budget = ModelBudget("test-a", rpm=3, rpd=100)
    for _ in range(3):
        await asyncio.wait_for(budget.acquire(), timeout=0.5)


async def test_budget_blocks_when_minute_full() -> None:
    budget = ModelBudget("test-b", rpm=2, rpd=100)
    await budget.acquire()
    await budget.acquire()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(budget.acquire(), timeout=0.2)


async def test_budget_raises_when_day_spent() -> None:
    budget = ModelBudget("test-c", rpm=10, rpd=2)
    await budget.acquire()
    await budget.acquire()
    with pytest.raises(DailyBudgetExceeded):
        await budget.acquire()


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


async def test_complete_falls_back_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = make_fake_acompletion([_rate_limit_error(), FakeResponse("answer")])
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    response = await complete("main", [{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "answer"
    assert calls[0]["model"] == llm_module.settings.model_main
    assert calls[1]["model"] == "gemini/fallback-model"


async def test_complete_quota_error_when_whole_chain_429s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, calls = make_fake_acompletion([_rate_limit_error(), _rate_limit_error()])
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    with pytest.raises(QuotaExceededError):
        await complete("main", [{"role": "user", "content": "hi"}])
    assert len(calls) == 2  # primary + one fallback, no endless retries


async def test_complete_skips_model_with_spent_daily_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, calls = make_fake_acompletion([FakeResponse("from fallback")])
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    # exhaust the primary's daily budget client-side (single key -> index 0)
    primary = f"{llm_module.settings.model_main}#0"
    llm_module._budgets[primary] = ModelBudget(primary, rpm=10, rpd=1)
    await llm_module._budgets[primary].acquire()

    response = await complete("main", [{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "from fallback"
    assert [c["model"] for c in calls] == ["gemini/fallback-model"]


async def test_complete_rotates_keys_before_next_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same model must be tried on every api key before falling back."""
    monkeypatch.setattr(llm_module.settings, "gemini_api_keys", "key-a,key-b")
    fake, calls = make_fake_acompletion([_rate_limit_error(), FakeResponse("second key")])
    monkeypatch.setattr(llm_module, "_acompletion", fake)

    response = await complete("main", [{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "second key"
    primary = llm_module.settings.model_main
    assert [(c["model"], c["api_key"]) for c in calls] == [
        (primary, "key-a"),
        (primary, "key-b"),
    ]


async def test_judge_role_uses_judge_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = make_fake_acompletion([FakeResponse('{"answer_relevance": 3, "faithfulness": 4}')])
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    await call_structured("judge", [{"role": "user", "content": "grade this"}])
    assert calls[0]["model"] == llm_module.settings.model_judge


async def test_complete_surfaces_non_quota_error(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = ValueError("bad request shape")
    fake, calls = make_fake_acompletion([boom, boom])
    monkeypatch.setattr(llm_module, "_acompletion", fake)
    with pytest.raises(ValueError, match="bad request shape"):
        await complete("main", [{"role": "user", "content": "hi"}])
    assert len(calls) == 2  # tried the chain, then surfaced the real error
