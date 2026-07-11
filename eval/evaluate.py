"""Evaluation suite: hit rate + LLM-as-judge answer relevance & faithfulness.

Runs every question through the real chat pipeline components (small-talk
router, condensation, hybrid retrieval, answer generation) sequentially, so
the shared RPM limiter paces the LLM calls. Multi-turn entries run inside one
session so condensation is actually exercised. Judge failures mark a question
"unscored" instead of crashing the suite.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from core import prompts
from core.chat_pipeline import chat_once, persist_turn, prepare_turn
from core.index import IndexStore
from core.llm import call_structured, complete, response_text
from core.sessions import SessionStore

logger = logging.getLogger("eval")

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def _page_hit(sources: list[dict[str, Any]], expected_file: str, expected_pages: list[int]) -> bool:
    """Did any retrieved chunk come from expected_file within expected_pages ±1?"""
    allowed = {p + d for p in expected_pages for d in (-1, 0, 1)}
    return any(
        s["filename"] == expected_file and s["page"] in allowed for s in sources
    )


async def _judge(question: str, context: str, answer: str) -> dict[str, Any] | None:
    try:
        result = await call_structured(
            "cheap",
            [
                {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompts.JUDGE_USER_TEMPLATE.format(
                        question=question, context=context or "(no context retrieved)", answer=answer
                    ),
                },
            ],
        )
        relevance = int(result["answer_relevance"])
        faithfulness = int(result["faithfulness"])
        if not (1 <= relevance <= 5 and 1 <= faithfulness <= 5):
            raise ValueError(f"judge scores out of range: {result}")
        return {"answer_relevance": relevance, "faithfulness": faithfulness}
    except Exception:
        logger.warning("judge failed, marking question unscored", exc_info=True)
        return None


async def _answer_final_turn(
    index: IndexStore, store: SessionStore, session_id: str, message: str
) -> tuple[str, list[dict[str, Any]], str, int]:
    """Run one turn via the real pipeline components, also capturing the full
    context text the model saw (needed for faithfulness judging).
    Returns (answer, sources, context_text, llm_calls)."""
    calls = 0
    history = await store.window(session_id)
    prepared = await prepare_turn(index, store, session_id, message)
    if history:
        calls += 1  # condensation
    if prepared.kind in ("canned", "refusal"):
        answer = prepared.answer or ""
        await persist_turn(store, session_id, message, answer, [], prepared.condensed_query)
        return answer, [], "", calls
    response = await complete("main", prepared.messages, timeout=60.0)
    calls += 1
    answer = response_text(response)
    await persist_turn(store, session_id, message, answer, prepared.sources, prepared.condensed_query)
    # prepared.messages[1] holds the numbered context blocks + history + question
    context_text = prepared.messages[1]["content"].split("Recent conversation:")[0]
    return answer, prepared.sources, context_text, calls


async def run_evaluation(index: IndexStore, store: SessionStore) -> dict[str, Any]:
    testset: list[dict[str, Any]] = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    per_question: list[dict[str, Any]] = []
    total_calls = 0
    started = time.perf_counter()

    for i, entry in enumerate(testset, start=1):
        qid = entry["id"]
        turns: list[str] = entry["turns"]
        session_id = f"eval-{qid}-{uuid.uuid4().hex[:6]}"
        logger.info("eval %d/%d: %s", i, len(testset), qid)
        final_question = turns[-1]

        try:
            # prior turns (multi-turn pairs) run through the full chat pipeline
            for prior in turns[:-1]:
                await chat_once(index, store, session_id, prior)
                total_calls += 2  # condense (maybe) + answer; conservative count

            answer, sources, context_text, calls = await _answer_final_turn(
                index, store, session_id, final_question
            )
            total_calls += calls
        except Exception:
            # One bad question (quota blip, transient upstream error) must not
            # kill the suite — mark it unscored and keep going.
            logger.warning("eval question %s failed, marking unscored", qid, exc_info=True)
            per_question.append(
                {
                    "id": qid,
                    "question": final_question,
                    "multi_turn": len(turns) > 1,
                    "hit": None,
                    "answer_relevance": None,
                    "faithfulness": None,
                    "answer": "",
                    "notes": "unscored (question failed: quota or upstream error)",
                }
            )
            await store.delete(session_id)
            continue

        expected_file = entry.get("expected_source_filename")
        expect_refusal = entry.get("expect_refusal", False)
        hit: bool | None = None
        if expected_file:
            hit = _page_hit(sources, expected_file, entry.get("expected_pages", []))

        scores = await _judge(final_question, context_text, answer)
        total_calls += 1
        notes = ""
        if expect_refusal:
            refused = not sources
            notes = "refused correctly" if refused else "FAILED to refuse (hallucination risk)"
        keywords = [k.lower() for k in entry.get("expected_answer_keywords", [])]
        if keywords:
            missing = [k for k in keywords if k not in answer.lower()]
            if missing:
                notes = (notes + "; " if notes else "") + f"missing keywords: {missing}"

        per_question.append(
            {
                "id": qid,
                "question": final_question,
                "multi_turn": len(turns) > 1,
                "hit": hit,
                "answer_relevance": scores["answer_relevance"] if scores else None,
                "faithfulness": scores["faithfulness"] if scores else None,
                "answer": answer,
                "notes": notes or ("unscored (judge failed)" if scores is None else ""),
            }
        )
        # cleanup eval sessions
        await store.delete(session_id)

    hits = [q["hit"] for q in per_question if q["hit"] is not None]
    relevance = [q["answer_relevance"] for q in per_question if q["answer_relevance"] is not None]
    faithfulness = [q["faithfulness"] for q in per_question if q["faithfulness"] is not None]
    summary = {
        "hit_rate": round(sum(hits) / len(hits), 3) if hits else 0.0,
        "answer_relevance": round(sum(relevance) / len(relevance), 2) if relevance else 0.0,
        "faithfulness": round(sum(faithfulness) / len(faithfulness), 2) if faithfulness else 0.0,
        "llm_calls": total_calls,
        "duration_s": round(time.perf_counter() - started, 1),
        "per_question": per_question,
    }
    _write_markdown(summary)
    logger.info(
        "eval done: hit_rate=%s relevance=%s faithfulness=%s calls=%d",
        summary["hit_rate"],
        summary["answer_relevance"],
        summary["faithfulness"],
        total_calls,
    )
    return summary


def _write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Evaluation Results",
        "",
        f"- **Hit Rate:** {summary['hit_rate']:.0%}",
        f"- **Answer Relevance (1–5):** {summary['answer_relevance']}",
        f"- **Faithfulness (1–5):** {summary['faithfulness']}",
        f"- **LLM calls:** {summary['llm_calls']}  ·  **Duration:** {summary['duration_s']}s",
        "",
        "| # | Question | Multi-turn | Hit | Relevance | Faithfulness | Notes |",
        "|---|----------|------------|-----|-----------|--------------|-------|",
    ]
    for q in summary["per_question"]:
        hit = "—" if q["hit"] is None else ("✅" if q["hit"] else "❌")
        rel = q["answer_relevance"] if q["answer_relevance"] is not None else "—"
        faith = q["faithfulness"] if q["faithfulness"] is not None else "—"
        mt = "yes" if q["multi_turn"] else ""
        question = q["question"].replace("|", "\\|")
        notes = (q["notes"] or "").replace("|", "\\|")
        lines.append(f"| {q['id']} | {question} | {mt} | {hit} | {rel} | {faith} | {notes} |")
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


async def _main() -> None:
    from core.logging import setup_logging

    setup_logging()
    index = IndexStore()
    store = SessionStore.from_settings()
    await run_evaluation(index, store)


if __name__ == "__main__":
    asyncio.run(_main())
