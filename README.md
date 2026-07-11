# Opkey Procurement RAG Chatbot

A Dockerized, session-aware RAG chatbot for procurement questions. FastAPI backend with JWT
auth, Redis-persisted multi-turn sessions, hybrid retrieval (dense + BM25 + cross-encoder
reranking) over ChromaDB, Gemini (via LiteLLM) for generation, and a Gradio chat UI with
streaming answers and source citations.

**Demo video:** _TODO — add link (YouTube unlisted / Loom / Drive)_

## Why these documents?

The knowledge base pairs a **software usage manual** with an **organizational policy** — a
realistic enterprise combination with deliberate ambiguity (e.g. "what is the approval
limit?" means different things in each):

| Document | What it covers |
|---|---|
| `data/oracle_fusion_using_procurement_26b.pdf` (670 pages) | Oracle Fusion Cloud Procurement "Using Procurement" guide — requisitions, purchase orders, agreements, approvals, supplier management |
| `data/richmond_procurement_policy.pdf` (9 pages) | University of Richmond procurement policy — competitive bidding thresholds, purchase methods, signature authority |

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
                │                    docker-compose                    │
                │                                                      │
 Browser ──────▶│  Gradio UI (7860)                                    │
                │      │  HTTP (login → JWT, SSE chat stream)          │
                │      ▼                                               │
                │  FastAPI api (8000)                                  │
                │   ├─ /auth/token ── JWT (HS256, python-jose)         │
                │   ├─ /chat ─┬─ small-talk router (rule-based)        │
                │   │         ├─ condense follow-up (Gemini flash-lite)│
                │   │         ├─ retrieve: bge-m3 dense (Chroma)       │
                │   │         │           + BM25 sparse → RRF fusion   │
                │   │         │           → bge-reranker → gate        │
                │   │         └─ answer: Gemini 3.5 Flash (streamed)   │
                │   ├─ /ingest, /documents ── PyMuPDF → heading-aware  │
                │   │         chunking → embed → Chroma + BM25         │
                │   └─ /sessions, /evaluate, /health                   │
                │      │                    │                          │
                │      ▼                    ▼                          │
                │  Redis (sessions,     ChromaDB (embedded,            │
                │   AOF, volume)         chroma_data volume)           │
                └─────────────────────────────────────────────────────┘
   Named volumes: redis_data, chroma_data, hf_cache (model cache)
   Bind mount: ./prebuilt_index (read-only bootstrap, see below)
```

## Prerequisites

- Docker + Docker Compose (that's all for the containerized path)
- A **Gemini API key** (free tier works): create one at https://aistudio.google.com/apikey
- Python 3.12 only if you want to run tests / develop locally

## Setup (5 steps)

```bash
# 1. clone
git clone <repo-url>
cd oracle_procurement_rag_assistant

# 2. configure
cp .env.example .env
# edit .env: set GEMINI_API_KEY and a random JWT_SECRET

# 3. start everything
docker-compose up --build

# 4. check the API
curl http://localhost:8000/health
# → {"status":"ok","docs_indexed":2,"chunks":1963}

# 5. chat
# open http://localhost:7860, log in with demo / demo123 (from .env), ask away
# Swagger docs: http://localhost:8000/docs (click Authorize, log in with demo credentials)
```

> **First boot** downloads the embedding + reranker models (~4 GB) into the `hf_cache`
> volume — one-time cost, persists across rebuilds. The knowledge-base index itself loads
> instantly from `prebuilt_index/` (see *Cold-start bootstrap* below).

## API Reference

Get a token first (only `/health` and `/auth/token` are unauthenticated):

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
```

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | no | Service liveness + index counts |
| POST | `/auth/token` | no | Exchange username/password for a JWT (60 min TTL) |
| POST | `/ingest` | 🔒 | Upload + index a PDF/TXT (no restart needed) |
| POST | `/chat` | 🔒 | Session-aware chat; SSE stream or JSON |
| GET | `/sessions/{id}/history` | 🔒 | Full turn-by-turn history |
| DELETE | `/sessions/{id}` | 🔒 | Delete a session |
| GET | `/documents` | 🔒 | List ingested documents |
| DELETE | `/documents/{id}` | 🔒 | Remove a document from the index |
| GET | `/evaluate` | 🔒 | Run the evaluation suite (takes minutes on free tier) |

```bash
# Health (no auth)
curl http://localhost:8000/health

# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./data/richmond_procurement_policy.pdf"

# Chat (non-streaming JSON)
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"session_id":"abc123","message":"What is the PO approval workflow?"}'

# Chat (SSE streaming)
curl -N -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"session_id":"abc123","message":"What about its thresholds?","stream":true}'

# Session history
curl http://localhost:8000/sessions/abc123/history -H "Authorization: Bearer $TOKEN"

# Delete session
curl -X DELETE http://localhost:8000/sessions/abc123 -H "Authorization: Bearer $TOKEN"

# List documents
curl http://localhost:8000/documents -H "Authorization: Bearer $TOKEN"

# Delete a document (id from /documents)
curl -X DELETE http://localhost:8000/documents/<doc_id> -H "Authorization: Bearer $TOKEN"

# Run evaluation (sequential, rate-limited — expect a few minutes)
curl http://localhost:8000/evaluate -H "Authorization: Bearer $TOKEN"
```

Error contract: `401` missing/invalid/expired token · `404` unknown session or document ·
`422` malformed body (Pydantic) · `429` rate limited (30/min chat, 5/min ingest per user)
with `Retry-After` · `503` LLM quota exhausted (clean JSON detail, never a stack trace).

## Design decisions

- **Sessions — Redis lists + AOF volume.** Each turn is an append-only JSON entry under
  `session:{id}`; full history is unbounded (for `/sessions/{id}/history`) while the LLM
  sees a window capped by `HISTORY_WINDOW_TURNS` (6) *and* a ~2000-token budget. AOF +
  named volume means multi-turn context survives `docker-compose down && up`.
- **Auth — JWT over API key.** `/auth/token` issues HS256 JWTs (python-jose) checked by a
  FastAPI dependency; integrates with Swagger's Authorize button via the OAuth2 password
  flow (the endpoint accepts both JSON and form bodies). Secrets only via `.env`.
- **Follow-up handling — query condensation.** With non-empty history, the cheap model
  rewrites "what about its approval limit?" into a standalone query ("purchase requisition
  approval limit") before retrieval. 3s budget, graceful fallback to the raw message.
- **Chunking — heading-aware, sentence-safe.** PyMuPDF parses page-by-page; spans larger
  than 1.2× the page's median font size become headings, maintaining a section path
  ("Purchase Orders > Approvals") that is prepended to every chunk. Chunks target ~450
  tokens (counted with the real bge-m3 tokenizer — `len(text.split())` is off by 20–30%)
  with 15% sentence-boundary overlap. bge-m3's 8192-token window means chunks are never
  truncated at embed time. Standalone acronyms (PO, PR, BPA, RFQ, RFP) are expanded in
  chunk text only, so BM25 matches both forms.
- **Retrieval — hybrid + rerank + confidence gate.** Dense (bge-m3/Chroma) top-12 and BM25
  top-12 are fused with rank-only RRF (k=60; raw scores are never mixed — scale mismatch),
  then the top-10 are reranked by a cross-encoder instantiated with a sigmoid activation
  (bge rerankers emit raw logits by default, which silently breaks thresholds). Top-4
  chunks with score ≥ 0.25 survive. If none do, the chatbot **refuses honestly** instead
  of calling the LLM with empty context. The reranker model is env-swappable
  (`RERANKER_MODEL`) with zero code changes.
- **LLM — Gemini via LiteLLM.** Answers use `gemini/gemini-3.5-flash`; condensation and
  judging use `gemini/gemini-3.1-flash-lite`. LiteLLM abstracts the provider, so switching
  to OpenAI/Anthropic/Ollama is a one-line env change. The wrapper enforces free-tier RPM
  limits client-side (sliding-window limiter per model), retries with backoff, and returns
  a clean 503 on quota exhaustion. Expected cost: **$0 on the free tier**, and well under
  the $3 guideline on paid keys (a full eval run is ~40 small calls).

  > **Observed free-tier reality:** the actual quota for `gemini-3.5-flash` is
  > **5 RPM / 20 requests per day per project** (verified in the AI Studio rate-limit
  > dashboard — far below the commonly cited 1,500 RPD). The architecture absorbs this
  > with **per-(model, key) budgets and a two-dimensional fallback**: every model has a
  > client-side RPM *and* RPD budget matched to the dashboard, optional key rotation
  > (`GEMINI_API_KEYS` — each key is its own quota project) exhausts the best model
  > across **all** keys first, and only then does the wrapper step down the model chain
  > (`3.5-flash → 3-flash-preview → 2.5-flash → 3.1-flash-lite → 2.5-flash-lite`, all
  > ids probe-verified) before returning a clean 503. Mid-stream disconnects salvage
  > the partial answer, and the eval suite marks quota-hit questions "unscored"
  > instead of crashing.
- **CPU-only Docker.** The api image installs the CPU torch wheel explicitly; no GPU is
  assumed anywhere in the compose path (GPU is a local-dev-only option via
  `EMBEDDING_DEVICE=cuda`). Models are cached in the `hf_cache` volume across rebuilds.
- **UI — pure HTTP client.** Gradio Blocks app with zero RAG logic: login panel → JWT in
  state → SSE streaming into the chat window → collapsible per-answer Sources panel →
  session badge + New Conversation button. 401/429/503 surface as friendly messages.

### Cold-start bootstrap (`prebuilt_index/`)

Embedding the 670-page Oracle guide on an unknown CPU takes several minutes and would look
like a hang on first run. The repo ships the prebuilt Chroma + BM25 artifacts; on startup
the api uses (in order): existing volume → copy of `prebuilt_index/` → full ingestion of
`/data` (background, logged). The prebuilt index is **only** a first-boot optimization —
the full lifecycle works through the API alone: delete all documents, re-upload via
`/ingest`, and the rebuilt index behaves identically (covered by an integration test).

## Evaluation

12 questions over both documents: 8 single-turn factual, 2 multi-turn pairs (follow-up
depends on the prior turn), 1 cross-document ambiguity probe, and 1 out-of-scope question
that must be refused. Metrics:

- **Hit Rate** — retrieved chunk from the expected file within expected pages ±1 (objective).
- **Keyword Coverage** — fraction of expected answer keywords present (objective,
  deterministic — no LLM involved, immune to judge leniency).
- **Answer Relevance & Faithfulness** — LLM-as-judge (1–5) with a strict calibrated
  rubric ("5 is rare", claim-by-claim grounding check, unsupported claims listed).
  The judge runs on a **held-out model** (`MODEL_JUDGE=gemini/gemini-2.5-flash`) that is
  never the answer model, so the system isn't grading its own output.

The suite runs sequentially through the per-model budgets (~40 LLM calls, a few minutes
on the free tier).

```bash
curl http://localhost:8000/evaluate -H "Authorization: Bearer $TOKEN"
```

### Results

Final run of 2026-07-11 — **15 questions**, answers on the default `gemini-3.5-flash`
chain, judged by the held-out `gemini-2.5-flash` with the strict rubric (35 LLM calls,
558s through the free-tier rate limiter):

- **Hit Rate:** 93% (13/14 scored questions; the refusal question is excluded by design)
- **Answer Relevance (1–5, strict judge):** 4.67
- **Faithfulness (1–5, strict judge):** 4.8
- **Keyword Coverage (objective, no LLM):** 100%

| # | Question | Multi-turn | Hit | Relevance | Faithfulness | Keywords | Notes |
|---|----------|------------|-----|-----------|--------------|----------|-------|
| richmond-thresholds | What are the competitive bidding thresholds at the University of Richmond? |  | ✅ | 5 | 5 | 100% |  |
| richmond-capital-equipment | What qualifies as capital equipment and what is its minimum purchase price? |  | ✅ | 5 | 5 | 100% |  |
| richmond-card-invoice | Can a University of Richmond purchase card be used to pay an invoice? |  | ✅ | 5 | 5 | 100% |  |
| richmond-tech-purchases | Which department manages technology purchases such as hardware and software? |  | ✅ | 5 | 5 | 100% |  |
| oracle-order-vs-requisition | What is the difference between an order and a requisition in Oracle Procurement? |  | ✅ | 5 | 5 | 100% |  |
| oracle-po-types | What purchase order types does Oracle Purchasing provide? |  | ✅ | 5 | 5 | 100% |  |
| oracle-requisition-lifecycle | What does the requisition life cycle refer to in Oracle Procurement? |  | ✅ | 5 | 5 | 100% |  |
| oracle-reassign-requisition | Can I reassign a requisition created by someone else in Oracle Procurement? |  | ✅ | 5 | 5 | 100% |  |
| richmond-sole-vs-single-source | What is the difference between sole source and single source? |  | ✅ | 5 | 5 | 100% |  |
| oracle-blanket-purchase-agreement | What is a blanket purchase agreement and when would you use one? |  | ✅ | 5 | 5 | 100% |  |
| oracle-supplier-registration | How does the supplier registration process work in Oracle Procurement? |  | ✅ | 5 | 5 | 100% |  |
| multiturn-oracle-requisition | What statuses can it have during approval? | yes | ✅ | 5 | 5 | 100% | fixed by the condensation constraint — see failure #1 |
| multiturn-richmond-thresholds | What is required for purchases above the highest threshold? | yes | ✅ | 3 | 4 | 100% | judge caught "must contact" where the policy says "encouraged to contact" |
| cross-doc-approval-limit | What is the approval limit for purchases? |  | ❌ | 5 | 5 | — | attribution fixed by the diversity guard — see failure #2 |
| out-of-scope-refusal | What is the capital of France? |  | — | 2 | 3 | — | refused correctly (a refusal *should* score low on relevance; the judge even counts the canned refusal wording as "claims") |

The score variance (3s, 4s, a 2) is the strict held-out judge working as intended — an
earlier run with a lenient same-model judge returned a uniform 5.0 across the board,
which is why the rubric was calibrated ("5 is rare", claim-by-claim grounding check)
and the judge moved to a model that never generates the answers. Its precision shows in
`multiturn-richmond-thresholds`: the answer said departments *"must contact"* the
Procurement Office for purchases over $125,000 where the policy says they are
*"encouraged to contact"* — exactly the modality drift a faithfulness judge exists to
catch.

### Failure analysis

**1. Session/context failure — condensation-induced retrieval drift (multiturn-oracle-requisition).**
Turn 1: *"What is a purchase requisition in Oracle Procurement?"* → answered correctly from
the requisitions chapter. Turn 2: *"What statuses can it have during approval?"* The
condenser resolved the pronoun correctly but rewrote the query as *"Oracle Procurement
purchase requisition approval **workflow** statuses"* — injecting the word "workflow",
which the user never said. In a 670-page manual dense with approval-workflow content, that
one word steered hybrid retrieval toward the generic *Transaction Console* /
*purchasing-document approval* sections (rerank top score 0.923 — confidently wrong) instead
of the requisition-status tables around pages 202–206. The answer was fluent, grounded, and
plausible — which is what makes this failure mode dangerous: faithfulness scored 5 because
the answer matched the *retrieved* chunks, but they were the wrong chunks. The failure is
**run-dependent**: in the strict-judge rerun the condensed query landed in the right pages
(hit ✅), yet the judge still scored relevance only **3/5** — the answer still mixed generic
document-approval statuses into what should have been a requisition-specific list, i.e. the
drift persists in softer form even when page-level retrieval succeeds.
**Hypothesis:** condensation should paraphrase minimally; every token it invents becomes a
high-weight retrieval term. **Fix shipped and validated:** the condensation prompt now
forbids introducing topic words — it may only reuse conversation vocabulary plus the noun
a pronoun resolves to. In the final run the condensed query came out as *"purchase
requisition statuses during approval Oracle Procurement"* (no invented "workflow"), the
right pages were retrieved, and the question scored **5/5 relevance** (previously ❌ hit /
3/5). A section-path filter biasing follow-ups toward the previous turn's chapter remains
the next-step hardening.

**2. Retrieval failure — cross-document ambiguity (cross-doc-approval-limit).**
*"What is the approval limit for purchases?"* has no single answer: Richmond defines
dollar thresholds ($10k/$125k) while the Oracle guide discusses approval routing and
signature authority. Retrieval confidence collapsed (top rerank score 0.452 vs ~0.99 on
well-posed questions; only 3 of 10 candidates cleared the 0.25 gate) and the kept chunks
missed the expected Richmond threshold pages. The retriever mixed both documents rather
than surfacing the policy table, and in one run the strict judge flagged an unsupported
cross-document attribution (faithfulness 4/5) — mixing sources invites the model to blur
which document says what. **Hypothesis:** ambiguous, entity-free queries dilute both dense
and sparse signals across documents. **Partial fix shipped:** a document-diversity guard —
when both documents clear the confidence gate, the second document is guaranteed a slot in
the context instead of one document monopolizing it. In the final run the answer correctly
presented both perspectives and faithfulness recovered to **5/5**; the page-level hit still
misses (the specific threshold table isn't in the top-4), so the question stays ❌ by the
strict hit metric. **Remaining fix I'd ship:** a document-scope clarifier — when top-1
rerank confidence is low and results span both documents, ask the user which context they
mean (Oracle software workflow vs. university policy).

**Bonus observation — the confidence gate works:** the out-of-scope question ("capital of
France") retrieved 0 chunks above the 0.25 threshold (top score 0.0) and produced the
honest refusal path with zero LLM calls wasted on hallucination.

## Tests

```bash
.venv/Scripts/python -m pytest tests -q -m "not slow"   # unit tests (no API needed)
.venv/Scripts/python -m pytest tests -q -m slow          # live document-lifecycle test (needs compose up)
```

Covers: auth (issuance, expiry, 401s), session store (window/token budget/delete), RRF
fusion, the LLM wrapper (limiter, JSON-mode retries, 429→503), the chat pipeline with
mocked LLM+retrieval (condensation skipped on first turn, invoked on follow-ups, refusal
below threshold), and the delete-all → re-ingest lifecycle.

## What I'd improve with more time

- **Per-user document scoping** — multi-tenant indexes under the same JWT auth.
- **Entity-aware condensation** — track discussed entities to resolve pronouns when two
  subjects are in play (the classic condensation failure mode).
- **Semantic caching** — cache condensed-query → answer pairs to cut repeat latency/cost.
- **Async batch evaluation** — parallelize the eval suite under a token-bucket budget.
- **Document-type filters** — let the user (or a router) restrict retrieval to the policy
  or the manual when the question implies one.
