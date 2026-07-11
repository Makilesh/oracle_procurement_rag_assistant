"""Redis-backed session store.

Full history is stored unbounded per session (for /sessions/{id}/history);
the window() view applies the turn/token caps that gate what the LLM sees.
"""

import json
import time
from typing import Any

from redis.asyncio import Redis, from_url

from core.config import settings

Turn = dict[str, Any]


def scoped_session_id(user: str, session_id: str) -> str:
    """Namespace a client-supplied session id under the authenticated JWT
    subject, so no user can read, continue, or delete another user's
    conversation by guessing its id (IDOR guard). Routes translate ids at the
    boundary; clients keep seeing the raw id they sent."""
    return f"{user}:{session_id}"


def _estimate_tokens(text: str) -> int:
    """Fast approximation (~4 chars/token) for the history budget."""
    return max(1, len(text) // 4)


class SessionStore:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @classmethod
    def from_settings(cls) -> "SessionStore":
        return cls(from_url(settings.redis_url, decode_responses=True))

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    async def ping(self) -> bool:
        return bool(await self._redis.ping())

    async def exists(self, session_id: str) -> bool:
        return bool(await self._redis.exists(self._key(session_id)))

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        turn.setdefault("ts", time.time())
        await self._redis.rpush(self._key(session_id), json.dumps(turn, ensure_ascii=False))
        await self._redis.hset(
            self._meta_key(session_id), mapping={"updated_at": str(turn["ts"])}
        )
        await self._redis.hsetnx(self._meta_key(session_id), "created_at", str(turn["ts"]))

    async def history(self, session_id: str) -> list[Turn] | None:
        """Full unbounded history; None if the session is unknown."""
        if not await self.exists(session_id):
            return None
        raw = await self._redis.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    async def window(self, session_id: str) -> list[Turn]:
        """Most recent turns fed to the LLM: last HISTORY_WINDOW_TURNS turns,
        additionally capped by HISTORY_TOKEN_BUDGET (newest kept first)."""
        raw = await self._redis.lrange(self._key(session_id), -settings.history_window_turns, -1)
        turns = [json.loads(item) for item in raw]
        budget = settings.history_token_budget
        kept: list[Turn] = []
        for turn in reversed(turns):
            cost = _estimate_tokens(str(turn.get("content", "")))
            if kept and budget - cost < 0:
                break
            budget -= cost
            kept.append(turn)
        kept.reverse()
        return kept

    async def delete(self, session_id: str) -> bool:
        removed = await self._redis.delete(self._key(session_id), self._meta_key(session_id))
        return removed > 0
