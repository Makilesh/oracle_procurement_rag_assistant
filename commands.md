# Commands

A quick reference for running and exercising the Opkey Procurement RAG Chatbot.

**Services:** `api` (port 8000), `ui` (port 7860), `chroma`, `redis`, `index-init`.
**Login:** `demo` / `demo123`.

> Commands are shown for **PowerShell** (Windows) with a **bash** equivalent where the
> syntax differs. On PowerShell, use `curl.exe` (not the `curl` alias).

---

## 1. Run the app

```bash
docker compose up --build -d     # first time: build images + start (downloads models ~2GB)
docker compose up -d             # subsequent starts
docker compose ps                # check services are healthy
docker compose down              # stop (data persists in volumes)
```

Then open:
- **http://localhost:7860** — chat UI
- **http://localhost:8000/docs** — Swagger API docs

---

## 2. Health check

```bash
curl http://localhost:8000/health
# → {"status":"ok","docs_indexed":2,"chunks":1963}
```

---

## 3. Authenticate

**PowerShell**
```powershell
$TOKEN = (Invoke-RestMethod -Uri http://localhost:8000/auth/token -Method Post `
  -ContentType "application/json" -Body '{"username":"demo","password":"demo123"}').access_token
```

**bash**
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
```

Every endpoint below (except `/health` and `/auth/token`) needs `-H "Authorization: Bearer $TOKEN"`.

---

## 4. Core endpoints

```bash
# Ingest a document (admin)
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./data/richmond_procurement_policy.pdf"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"What are the competitive bidding thresholds?"}'

# List documents
curl http://localhost:8000/documents -H "Authorization: Bearer $TOKEN"

# Session history
curl http://localhost:8000/sessions/s1/history -H "Authorization: Bearer $TOKEN"

# Delete a session
curl -X DELETE http://localhost:8000/sessions/s1 -H "Authorization: Bearer $TOKEN"

# Run the evaluation suite
curl http://localhost:8000/evaluate -H "Authorization: Bearer $TOKEN"
```

On PowerShell, prefix with `curl.exe` and escape inner quotes in the `-d` body as `\"`
(or use Swagger at `/docs`, which needs no quoting).

---

## 5. Tests

```bash
pip install -r requirements-dev.txt        # test/lint tooling
pytest tests -q -m "not slow"              # 70 unit tests (~4s, no network)
pytest tests -q -m slow                    # live integration test (stack must be up)
ruff check .                               # lint
```

---

## 6. Inspect running state

```bash
docker compose logs -f api                 # follow api logs
curl http://localhost:8000/metrics         # Prometheus metrics

# Sessions stored in Redis
docker exec oracle_procurement_rag_assistant-redis-1 redis-cli --scan --pattern "session:*"
docker exec oracle_procurement_rag_assistant-redis-1 redis-cli LRANGE "session:demo:s1" 0 -1
```