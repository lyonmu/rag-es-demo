# AGENTS.md — RAG-ES-Demo

## Environment

- **Python 3.10+** in conda env `rag-es-demo`: `conda activate rag-es-demo`
- **Poetry** for dependency management. After changing `pyproject.toml`: `poetry lock && poetry install`
- No separate virtualenv needed — conda handles it.

## Quick Commands

```bash
# Run tests (all, no ES required)
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_chunkers/test_markdown_chunker.py::test_single_heading -v

# Start dev server
poetry run uvicorn app.main:app --reload

# Start prod server
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start ES (Docker Compose)
docker compose up -d
```

## Prerequisites (must be running)

1. **Elasticsearch 7.10.0** at `localhost:9200`, user `elastic`, password `Cass@123456`
   - Requires `analysis-ik` plugin for Chinese BM25 tokenization
   - ES 7.10 has **no native kNN** — vector search uses `script_score` + `cosineSimilarity` (brute-force scan)
   - elasticsearch-py **v7** client only (`elasticsearch[async]`), v8 is **incompatible**
     - Uses `http_auth` not `basic_auth`; `timeout` not `request_timeout`; `body=` not `operations=`
     - Health check uses `info()` not `ping()` (ping() unreliable against ES 7.x)
   - `dense_vector` does NOT support `index: true` — must use `script_score`
2. **Local embedding model** at `data/models/bge-small-zh-v1.5/` (已提交到仓库)
3. LLM 使用 OpenAI 兼容协议 — 通过 `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` 配置
4. **IK 分词器安装**: `curl -L -o plugins/analysis-ik.zip https://get.infini.cloud/elasticsearch/analysis-ik/7.10.0 && unzip plugins/analysis-ik.zip -d plugins/analysis-ik/ && docker compose restart es`

## Architecture & Key Decisions

- **API prefix**: all routes under `/rag/api/v1` (set via `include_router(prefix=...)` in `main.py`)
- **Unified response**: all endpoints return HTTP 200 + `{code, message, data}`, never raw dicts
  - Global exception handler in `register_exception_handler(app)` converts HTTPException → ApiResponse
  - Error codes: `100xx` system, `101xx` KB, `102xx` search, `103xx` chat
- **SQLite KB store** (`app/core/kb_store.py`): stores name, description, index_name, doc_count. DB at `data/kb_store.db`.
- **SSE event order**: token (stream) → done (full answer + sources) → sources (references at end)
- **LangChain usage**: only `ChatOpenAI` for LLM streaming — no LCEL/chain orchestration, RAG pipeline is handwritten

## Testing

- `pytest-asyncio` mode = `auto` (configured in pyproject.toml)
- **Tests skip ES/lifespan** — `tests/conftest.py` replaces `lifespan_context` with a no-op via `app.router.lifespan_context = _null_lifespan`
- When writing new router tests, use `from tests.conftest import create_test_app` to get an app without ES connection
- Pure logic tests (chunker, RRF) need no mocking
- Service tests mock `get_es_client` and `encode_texts`
- Router test paths must include `/rag/api/v1/` prefix

## .env

- **Not tracked in git** (`.gitignore`). Contains ES password and LLM API key.
- All config flows through `app/config.py` → `Settings` class with `pydantic-settings`
- New environment variables must be added to both `.env` and `Settings` class

## .gitignore

Excludes: `__pycache__`, `data/output/`, `data/output1/`, `data/uploads/*.md`, `.env`, `data/kb_store.db`
