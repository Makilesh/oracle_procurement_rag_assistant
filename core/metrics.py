"""Prometheus metrics: request-level, RAG-stage, and LLM-budget observability.

Scraped from GET /metrics (unauthenticated — meant for infra-side scraping;
lock it down at the ingress in a real deployment).
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests by route template and status code",
    ["method", "route", "status"],
)

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency by route template",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds",
    "Latency of each RAG pipeline stage",
    ["stage"],  # condense | embed | search | rerank | first_token | generate
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

LLM_CALLS = Counter(
    "llm_calls_total",
    "LLM completion attempts by model and outcome",
    ["model", "outcome"],  # ok | rate_limited | day_budget_spent | error
)

LLM_QUOTA_EXHAUSTED = Counter(
    "llm_quota_exhausted_total",
    "Requests that exhausted the entire model/key fallback chain",
)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
