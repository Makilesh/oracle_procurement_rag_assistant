from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import get_current_user
from api.schemas import EvalQuestionResult, EvalResponse
from core.llm import QuotaExceededError
from eval.evaluate import run_evaluation

router = APIRouter()


@router.get("/evaluate", response_model=EvalResponse)
async def evaluate(request: Request, user: str = Depends(get_current_user)) -> EvalResponse:
    """Runs the full suite sequentially through the RPM limiter — expect a few
    minutes on the free tier; progress is logged per question."""
    try:
        summary = await run_evaluation(request.app.state.index, request.app.state.sessions)
    except QuotaExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return EvalResponse(
        hit_rate=summary["hit_rate"],
        answer_relevance=summary["answer_relevance"],
        faithfulness=summary["faithfulness"],
        keyword_coverage=summary["keyword_coverage"],
        llm_calls=summary["llm_calls"],
        per_question=[
            EvalQuestionResult(
                id=q["id"],
                question=q["question"],
                hit=q["hit"],
                answer_relevance=q["answer_relevance"],
                faithfulness=q["faithfulness"],
                keyword_coverage=q["keyword_coverage"],
                notes=q["notes"],
            )
            for q in summary["per_question"]
        ],
        extra={"duration_s": summary["duration_s"]},
    )
