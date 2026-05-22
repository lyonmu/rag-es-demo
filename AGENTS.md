# Repository Guidelines

## Project Structure & Module Organization
- `app/` contains production code for the FastAPI service.
- `app/main.py` is the API entrypoint; routers live in `app/routers/` (`chat.py`, `search.py`, `knowledge_base.py`, `upload.py`).
- Core infrastructure is under `app/core/` (Elasticsearch client, embedder, error codes, unified response, SQLite KB store).
- Retrieval logic is split in `app/retrievers/` (BM25, vector, hybrid/RRF), and document chunking is in `app/chunkers/`.
- Business orchestration is in `app/services/`.
- Tests mirror app modules in `tests/` (for example, `tests/test_routers/`, `tests/test_services/`, `tests/test_chunkers/`).
- Runtime assets and local model files are under `data/`; design notes and plans are in `docs/`.

## Build, Test, and Development Commands
- `poetry install`: install project and dev dependencies.
- `docker compose up -d`: start Elasticsearch 7.10 required by retrieval features.
- `poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`: run API locally with auto-reload.
- `poetry run pytest`: run full test suite.
- `poetry run pytest --cov=app --cov-report=term-missing`: run tests with coverage details.

## Coding Style & Naming Conventions
- Target Python `3.10+`; follow PEP 8 with 4-space indentation.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and explicit, domain-focused names (for example `hybrid_retriever.py`).
- Keep API responses aligned with the repository’s unified `{code, message, data}` contract.
- Keep routers thin; place retrieval/ingest/chat logic in `services/` and `retrievers/`.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` (`asyncio_mode = auto`).
- Add tests beside matching domain folders and name files `test_<module>.py`.
- Prefer deterministic unit tests with mocks for Elasticsearch/LLM dependencies.
- Validate both success and error-code paths for API endpoints.
- Router tests should use `httpx.AsyncClient` + `httpx.ASGITransport` (avoid `TestClient` hangs in this repo’s async/lifespan setup).
- For route tests, create app via `tests.conftest.create_test_app()` so lifespan is replaced and no real ES/model init is triggered.

## Commit & Pull Request Guidelines
- Follow conventional prefixes seen in history: `feat:`, `fix:`, `docs:`.
- Keep commit messages imperative and scoped (example: `fix: include total in search response`).
- PRs should include: purpose, key changes, test evidence (`pytest` output), and API examples when behavior changes.
- Link related issues/tasks and call out config or migration impacts (`.env`, ES plugin/data requirements).

## Security & Configuration Tips
- Do not commit real secrets; copy `.env.example` to `.env` for local setup.
- Elasticsearch credentials and plugin setup are required for end-to-end runs; verify service health before debugging app logic.
