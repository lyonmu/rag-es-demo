# RAG ES Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement low-risk RAG retrieval, ingest, indexing, and chat-context optimizations while keeping Elasticsearch 7.10 and BGE-Small-ZH-v1.5 unchanged.

**Architecture:** Keep the existing FastAPI/service/retriever layout. Add compatibility-safe fields and settings, make new KB mappings richer, keep old indexes searchable, and improve retrieval through better chunking, BM25 query structure, query embedding cache, RRF chunk identity, graceful degradation, and chat context budgeting.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic Settings, elasticsearch-py 7.x async client, SentenceTransformers, LangChain OpenAI, pytest, pytest-asyncio.

---

## File Structure

- Modify `app/config.py`: add optimization settings with safe defaults.
- Modify `app/chunkers/markdown_chunker.py`: add second-stage size-aware splitting while preserving heading paths.
- Modify `app/core/es_client.py`: update `KB_INDEX_MAPPING` with new fields and mapping version support.
- Modify `app/services/ingest_service.py`: write new chunk metadata fields, configurable refresh policy, and bulk failure accounting.
- Modify `app/routers/upload.py`: increment `doc_count` only when at least one chunk was indexed successfully.
- Modify `app/retrievers/bm25_retriever.py`: replace single-field match query with heading-aware bool query.
- Modify `app/core/embedder.py`: add cached query embedding helper while keeping batch document embedding unchanged.
- Modify `app/retrievers/vector_retriever.py`: use cached query embedding and keep ES 7.10 `script_score` compatibility.
- Modify `app/retrievers/hybrid_retriever.py`: prefer `chunk_id` for fusion identity, preserve old-index fallback, and degrade when one retrieval branch fails.
- Modify `app/services/chat_service.py`: add context budgeting helper and use it when building prompts.
- Modify `app/schemas/response.py`: add optional chunk-level and ingest failure fields if response models are used directly.
- Add `tests/test_retrievers/test_bm25_retriever.py`: validate BM25 query body.
- Add `tests/test_retrievers/test_vector_retriever.py`: validate cached query embedding and script-score body.
- Add or modify existing tests under `tests/test_chunkers/`, `tests/test_services/`, `tests/test_retrievers/`, and `tests/test_services/`.

---

### Task 1: Add Optimization Settings

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing settings test**

Create `tests/test_config.py`:

```python
"""Tests for application settings defaults."""

from app.config import Settings


def test_optimization_settings_defaults():
    settings = Settings()

    assert settings.chunk_max_chars == 1200
    assert settings.chunk_overlap_chars == 150
    assert settings.chunk_min_chars == 80
    assert settings.query_embedding_cache_size == 256
    assert settings.ingest_refresh_policy == "false"
    assert settings.bm25_heading_boost == 2.0
    assert settings.bm25_content_boost == 1.0
    assert settings.bm25_phrase_boost == 1.5
    assert settings.chat_context_max_chars == 6000
    assert settings.vector_candidate_mode is False
    assert settings.vector_candidate_top_k == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
poetry run pytest tests/test_config.py -v
```

Expected: FAIL with `AttributeError` for `chunk_max_chars`.

- [ ] **Step 3: Add settings**

In `app/config.py`, add these fields to `Settings` after the existing retrieval settings:

```python
    # Chunking
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 80

    # Retrieval optimization
    query_embedding_cache_size: int = 256
    bm25_heading_boost: float = 2.0
    bm25_content_boost: float = 1.0
    bm25_phrase_boost: float = 1.5
    vector_candidate_mode: bool = False
    vector_candidate_top_k: int = 200

    # Ingest
    ingest_refresh_policy: str = "false"

    # Chat
    chat_context_max_chars: int = 6000
```

- [ ] **Step 4: Run settings test**

Run:

```bash
poetry run pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add rag optimization settings"
```

---

### Task 2: Add Size-Aware Markdown Chunking

**Files:**
- Modify: `app/chunkers/markdown_chunker.py`
- Test: `tests/test_chunkers/test_markdown_chunker.py`

- [ ] **Step 1: Add failing chunk split tests**

Append to `tests/test_chunkers/test_markdown_chunker.py`:

```python

def test_long_heading_section_splits_with_overlap(monkeypatch):
    from app.chunkers import markdown_chunker

    monkeypatch.setattr(markdown_chunker.settings, "chunk_max_chars", 80)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_overlap_chars", 10)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_min_chars", 20)

    text = "# 长章节\n\n" + "甲" * 150
    chunks = chunk_markdown(text)

    assert len(chunks) >= 2
    assert all(chunk.heading_path == "长章节" for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].content[-10:] in chunks[1].content


def test_small_tail_merges_into_previous_chunk(monkeypatch):
    from app.chunkers import markdown_chunker

    monkeypatch.setattr(markdown_chunker.settings, "chunk_max_chars", 80)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_overlap_chars", 10)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_min_chars", 30)

    text = "# 长章节\n\n" + "乙" * 95
    chunks = chunk_markdown(text)

    assert len(chunks) == 1
    assert chunks[0].heading_path == "长章节"
    assert "乙" * 95 in chunks[0].content
```

- [ ] **Step 2: Run chunker tests to verify failure**

Run:

```bash
poetry run pytest tests/test_chunkers/test_markdown_chunker.py -v
```

Expected: FAIL because long sections are not split.

- [ ] **Step 3: Implement second-stage splitting**

In `app/chunkers/markdown_chunker.py`, import settings:

```python
from app.config import settings
```

Replace direct `chunks.append(...)` calls in `chunk_markdown` with collection of base sections and a final normalization pass. Add these helpers below `_build_heading_path`:

```python
def _split_oversized_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Split oversized chunks while preserving heading paths."""
    result: list[Chunk] = []
    for chunk in chunks:
        parts = _split_text_with_overlap(
            chunk.content,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
            min_chars=settings.chunk_min_chars,
        )
        for part in parts:
            result.append(
                Chunk(
                    content=part,
                    heading_path=chunk.heading_path,
                    chunk_index=len(result),
                )
            )
    return result


def _split_text_with_overlap(
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[str]:
    """Split text by character budget with deterministic overlap."""
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)

    if len(parts) >= 2 and len(parts[-1]) < min_chars:
        parts[-2] = f"{parts[-2]}{parts[-1]}"
        parts.pop()

    return parts
```

At the end of `chunk_markdown`, return `_split_oversized_chunks(chunks)` instead of `chunks`. For the no-heading branch, build the single chunk and return `_split_oversized_chunks(chunks)`.

- [ ] **Step 4: Run chunker tests**

Run:

```bash
poetry run pytest tests/test_chunkers/test_markdown_chunker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/chunkers/markdown_chunker.py tests/test_chunkers/test_markdown_chunker.py
git commit -m "feat: split oversized markdown chunks"
```

---

### Task 3: Update ES Mapping for New KB Indexes

**Files:**
- Modify: `app/core/es_client.py`
- Test: `tests/test_core/test_es_client.py`

- [ ] **Step 1: Write failing mapping test**

Create directory and file `tests/test_core/test_es_client.py`:

```python
"""Tests for Elasticsearch index mapping."""

from app.core.es_client import KB_INDEX_MAPPING


def test_kb_index_mapping_contains_optimization_fields():
    props = KB_INDEX_MAPPING["mappings"]["properties"]

    assert props["chunk_id"] == {"type": "keyword"}
    assert props["content_with_heading"] == {"type": "text", "analyzer": "ik_max_word"}
    assert props["content_length"] == {"type": "integer"}
    assert props["chunk_text_hash"] == {"type": "keyword"}
    assert props["mapping_version"] == {"type": "keyword"}
    assert props["embedding"] == {"type": "dense_vector", "dims": 512}
```

- [ ] **Step 2: Run mapping test to verify failure**

Run:

```bash
poetry run pytest tests/test_core/test_es_client.py -v
```

Expected: FAIL with `KeyError: 'chunk_id'`.

- [ ] **Step 3: Add mapping fields**

In `app/core/es_client.py`, update `KB_INDEX_MAPPING["mappings"]["properties"]` to include:

```python
            "chunk_id": {"type": "keyword"},
            "content_with_heading": {"type": "text", "analyzer": "ik_max_word"},
            "content_length": {"type": "integer"},
            "chunk_text_hash": {"type": "keyword"},
            "mapping_version": {"type": "keyword"},
```

Keep all existing fields unchanged.

- [ ] **Step 4: Run mapping test**

Run:

```bash
poetry run pytest tests/test_core/test_es_client.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/es_client.py tests/test_core/test_es_client.py
git commit -m "feat: add optimized kb index mapping fields"
```

---

### Task 4: Enrich Ingest Documents and Handle Bulk Failures

**Files:**
- Modify: `app/services/ingest_service.py`
- Modify: `app/routers/upload.py`
- Test: `tests/test_services/test_ingest_service.py`
- Test: `tests/test_routers/test_upload.py`

- [ ] **Step 1: Add failing ingest metadata test**

Append to `tests/test_services/test_ingest_service.py`:

```python

@pytest.mark.asyncio
async def test_ingest_markdown_writes_optimized_chunk_fields(monkeypatch):
    content = "# 标题\n\n内容"
    mock_bulk_response = {"errors": False, "items": [{"index": {"_id": "x", "status": 201}}]}

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512]

        result = await ingest_markdown("test-kb-id", "test.md", content)

        body = mock_client.bulk.call_args.kwargs["body"]
        document = body[1]
        assert document["chunk_id"] == body[0]["index"]["_id"]
        assert document["content_with_heading"].startswith("标题\n")
        assert document["content_length"] == len(document["content"])
        assert len(document["chunk_text_hash"]) == 64
        assert document["mapping_version"] == "rag-es-optimization-v1"
        assert result["failed_chunks_count"] == 0
```

- [ ] **Step 2: Add failing partial and all-failed bulk tests**

Append to `tests/test_services/test_ingest_service.py`:

```python

@pytest.mark.asyncio
async def test_ingest_markdown_reports_partial_bulk_failures():
    content = "# A\n\n一\n\n# B\n\n二"
    mock_bulk_response = {
        "errors": True,
        "items": [
            {"index": {"_id": "ok", "status": 201}},
            {"index": {"_id": "bad", "status": 500, "error": {"reason": "boom"}}},
        ],
    }

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512, [0.2] * 512]

        result = await ingest_markdown("test-kb-id", "test.md", content)

        assert result["chunks_count"] == 2
        assert result["indexed_chunks_count"] == 1
        assert result["failed_chunks_count"] == 1
        assert result["failed_chunk_ids"] == ["bad"]


@pytest.mark.asyncio
async def test_ingest_markdown_raises_when_all_bulk_items_fail():
    content = "# A\n\n一"
    mock_bulk_response = {
        "errors": True,
        "items": [{"index": {"_id": "bad", "status": 500, "error": {"reason": "boom"}}}],
    }

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:
        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512]

        with pytest.raises(RuntimeError, match="All chunks failed"):
            await ingest_markdown("test-kb-id", "test.md", content)
```

- [ ] **Step 3: Add failing upload doc_count test**

Append to `tests/test_routers/test_upload.py`:

```python

def test_upload_does_not_increment_doc_count_when_no_chunks_indexed():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.upload.get_kb") as mock_get, \
         mock.patch("app.routers.upload.ingest_markdown") as mock_ingest, \
         mock.patch("app.routers.upload.update_doc_count") as mock_update:
        mock_get.return_value = {
            "kb_id": "testkb",
            "name": "Test",
            "description": "",
            "index_name": "rag_kb_testkb",
            "doc_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        mock_ingest.return_value = {
            "doc_id": "abc123",
            "filename": "test.md",
            "chunks_count": 2,
            "indexed_chunks_count": 0,
            "failed_chunks_count": 2,
        }

        resp = client.post(
            "/rag/api/v1/kb/testkb/upload",
            files={"file": ("test.md", b"# Hello", "text/markdown")},
        )

        assert resp.status_code == 200
        assert resp.json()["code"] == 10000
        mock_update.assert_not_called()
```

- [ ] **Step 4: Run ingest and upload tests to verify failure**

Run:

```bash
poetry run pytest tests/test_services/test_ingest_service.py tests/test_routers/test_upload.py -v
```

Expected: FAIL because optimized fields and indexed-count handling do not exist.

- [ ] **Step 5: Implement ingest metadata and failure accounting**

In `app/services/ingest_service.py`, add imports:

```python
import hashlib
```

Add constant near logger:

```python
MAPPING_VERSION = "rag-es-optimization-v1"
```

Add helpers:

```python
def _chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}_{chunk_index}"


def _content_with_heading(heading_path: str, content: str) -> str:
    return f"{heading_path}\n{content}" if heading_path else content


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bulk_failures(resp: dict) -> list[str]:
    failures: list[str] = []
    for item in resp.get("items", []):
        index_result = item.get("index", {})
        if "error" in index_result:
            failures.append(index_result.get("_id", ""))
    return [item for item in failures if item]
```

When building each bulk document, compute `chunk_id = _chunk_id(doc_id, chunk.chunk_index)`, use it as `_id`, and add these fields:

```python
                "chunk_id": chunk_id,
                "content_with_heading": _content_with_heading(chunk.heading_path, chunk.content),
                "content_length": len(chunk.content),
                "chunk_text_hash": _sha256_text(chunk.content),
                "mapping_version": MAPPING_VERSION,
```

Change bulk call to:

```python
    resp = await client.bulk(body=actions, refresh=settings.ingest_refresh_policy)
```

Import `settings` from `app.config`.

After bulk response, calculate:

```python
    failed_chunk_ids = _bulk_failures(resp)
    failed_chunks_count = len(failed_chunk_ids)
    indexed_chunks_count = len(chunks) - failed_chunks_count

    if failed_chunks_count:
        logger.error("Bulk indexing failed for chunks: %s", failed_chunk_ids)

    if indexed_chunks_count == 0:
        raise RuntimeError("All chunks failed during bulk indexing")
```

Return:

```python
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
        "indexed_chunks_count": indexed_chunks_count,
        "failed_chunks_count": failed_chunks_count,
        "failed_chunk_ids": failed_chunk_ids,
    }
```

For the empty-content branch, return `indexed_chunks_count=0`, `failed_chunks_count=0`, and `failed_chunk_ids=[]`.

- [ ] **Step 6: Update upload doc_count condition**

In `app/routers/upload.py`, replace:

```python
    if result.get("chunks_count", 0) > 0:
        update_doc_count(kb_id, increment=1)
```

with:

```python
    if result.get("indexed_chunks_count", result.get("chunks_count", 0)) > 0:
        update_doc_count(kb_id, increment=1)
```

- [ ] **Step 7: Run ingest and upload tests**

Run:

```bash
poetry run pytest tests/test_services/test_ingest_service.py tests/test_routers/test_upload.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/services/ingest_service.py app/routers/upload.py tests/test_services/test_ingest_service.py tests/test_routers/test_upload.py
git commit -m "feat: enrich ingest chunks and report bulk failures"
```

---

### Task 5: Improve BM25 Query Structure

**Files:**
- Modify: `app/retrievers/bm25_retriever.py`
- Test: `tests/test_retrievers/test_bm25_retriever.py`

- [ ] **Step 1: Write failing BM25 query test**

Create `tests/test_retrievers/test_bm25_retriever.py`:

```python
"""Tests for BM25 retriever query body."""

import unittest.mock as mock

import pytest

from app.retrievers.bm25_retriever import bm25_search


@pytest.mark.asyncio
async def test_bm25_search_uses_heading_aware_multi_match():
    with mock.patch("app.retrievers.bm25_retriever.get_es_client") as mock_get_es:
        mock_client = mock.AsyncMock()
        mock_client.search = mock.AsyncMock(return_value={"hits": {"hits": []}})
        mock_get_es.return_value = mock_client

        await bm25_search("kb1", "混合检索", top_k=7)

        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["index"] == "rag_kb_kb1"
        assert kwargs["body"]["size"] == 7

        query = kwargs["body"]["query"]
        assert "bool" in query
        should = query["bool"]["should"]
        multi_match = should[0]["multi_match"]
        assert multi_match["query"] == "混合检索"
        assert "content^1.0" in multi_match["fields"]
        assert "heading_path^2.0" in multi_match["fields"]
        assert "content_with_heading^1.5" in multi_match["fields"]
        assert should[1]["match_phrase"]["content"]["boost"] == 1.5
        assert query["bool"]["minimum_should_match"] == 1
```

- [ ] **Step 2: Run BM25 test to verify failure**

Run:

```bash
poetry run pytest tests/test_retrievers/test_bm25_retriever.py -v
```

Expected: FAIL because the current body uses a single `match` query.

- [ ] **Step 3: Implement heading-aware BM25 body**

In `app/retrievers/bm25_retriever.py`, import settings:

```python
from app.config import settings
```

Replace `body` with:

```python
    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                f"content^{settings.bm25_content_boost}",
                                f"heading_path^{settings.bm25_heading_boost}",
                                f"content_with_heading^{settings.bm25_phrase_boost}",
                            ],
                            "operator": "or",
                        }
                    },
                    {
                        "match_phrase": {
                            "content": {
                                "query": query,
                                "boost": settings.bm25_phrase_boost,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "size": top_k,
    }
```

- [ ] **Step 4: Run BM25 test**

Run:

```bash
poetry run pytest tests/test_retrievers/test_bm25_retriever.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/retrievers/bm25_retriever.py tests/test_retrievers/test_bm25_retriever.py
git commit -m "feat: improve bm25 retrieval query"
```

---

### Task 6: Add Cached Query Embeddings for Vector Search

**Files:**
- Modify: `app/core/embedder.py`
- Modify: `app/core/__init__.py`
- Modify: `app/retrievers/vector_retriever.py`
- Test: `tests/test_retrievers/test_vector_retriever.py`

- [ ] **Step 1: Write failing vector cache tests**

Create `tests/test_retrievers/test_vector_retriever.py`:

```python
"""Tests for vector retriever."""

import unittest.mock as mock

import pytest

from app.core import embedder
from app.retrievers.vector_retriever import vector_search


@pytest.mark.asyncio
async def test_vector_search_uses_cached_query_embedding():
    embedder.encode_query.cache_clear()

    with mock.patch("app.core.embedder.encode_texts") as mock_encode, \
         mock.patch("app.retrievers.vector_retriever.get_es_client") as mock_get_es:
        mock_encode.return_value = [[0.1] * 512]
        mock_client = mock.AsyncMock()
        mock_client.search = mock.AsyncMock(return_value={"hits": {"hits": []}})
        mock_get_es.return_value = mock_client

        await vector_search("kb1", "重复问题", top_k=3)
        await vector_search("kb1", "重复问题", top_k=3)

        mock_encode.assert_called_once_with(["重复问题"])
        assert mock_client.search.call_count == 2


@pytest.mark.asyncio
async def test_vector_search_keeps_script_score_query():
    embedder.encode_query.cache_clear()

    with mock.patch("app.core.embedder.encode_texts") as mock_encode, \
         mock.patch("app.retrievers.vector_retriever.get_es_client") as mock_get_es:
        mock_encode.return_value = [[0.2] * 512]
        mock_client = mock.AsyncMock()
        mock_client.search = mock.AsyncMock(return_value={"hits": {"hits": []}})
        mock_get_es.return_value = mock_client

        await vector_search("kb1", "向量检索", top_k=4)

        body = mock_client.search.call_args.kwargs["body"]
        script_score = body["query"]["script_score"]
        assert script_score["query"] == {"match_all": {}}
        assert "cosineSimilarity" in script_score["script"]["source"]
        assert script_score["script"]["params"]["query_vector"] == [0.2] * 512
        assert body["size"] == 4
```

- [ ] **Step 2: Run vector tests to verify failure**

Run:

```bash
poetry run pytest tests/test_retrievers/test_vector_retriever.py -v
```

Expected: FAIL because `encode_query` does not exist.

- [ ] **Step 3: Implement cached query embedding helper**

In `app/core/embedder.py`, import `lru_cache`:

```python
from functools import lru_cache
```

Add below `encode_texts`:

```python
@lru_cache(maxsize=settings.query_embedding_cache_size)
def encode_query(text: str) -> list[float]:
    """Encode a single query with an in-process LRU cache."""
    return encode_texts([text])[0]
```

In `app/core/__init__.py`, update import:

```python
from .embedder import encode_query, encode_texts, get_embedder
```

Add `"encode_query"` to `__all__`.

- [ ] **Step 4: Use cached query embedding in vector retriever**

In `app/retrievers/vector_retriever.py`, change import:

```python
from app.core import encode_query, get_es_client
```

Replace:

```python
    query_vector = encode_texts([query])[0]
```

with:

```python
    query_vector = encode_query(query)
```

- [ ] **Step 5: Run vector tests**

Run:

```bash
poetry run pytest tests/test_retrievers/test_vector_retriever.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/core/embedder.py app/core/__init__.py app/retrievers/vector_retriever.py tests/test_retrievers/test_vector_retriever.py
git commit -m "feat: cache query embeddings for vector search"
```

---

### Task 7: Make Hybrid Fusion Chunk-Aware and Degradable

**Files:**
- Modify: `app/retrievers/hybrid_retriever.py`
- Test: `tests/test_retrievers/test_hybrid_retriever.py`

- [ ] **Step 1: Add failing chunk identity test**

Append to `tests/test_retrievers/test_hybrid_retriever.py`:

```python

def test_rrf_prefers_chunk_id_over_es_id():
    bm25 = [
        {
            "_id": "legacy_id_a",
            "_score": 5.0,
            "_source": {
                "chunk_id": "chunk_a",
                "doc_id": "doc_a",
                "content": "A",
                "filename": "a.md",
                "heading_path": "A",
            },
        }
    ]
    vector = [
        {
            "_id": "different_es_id",
            "_score": 0.9,
            "_source": {
                "chunk_id": "chunk_a",
                "doc_id": "doc_a",
                "content": "A",
                "filename": "a.md",
                "heading_path": "A",
            },
        }
    ]

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    assert len(results) == 1
    assert results[0].doc_id == "doc_a"
    assert results[0].bm25_score == 5.0
    assert results[0].vector_score == 0.9
```

- [ ] **Step 2: Add failing degradation tests**

Append to `tests/test_retrievers/test_hybrid_retriever.py`:

```python
import pytest

from app.retrievers.hybrid_retriever import hybrid_search


@pytest.mark.asyncio
async def test_hybrid_search_returns_vector_results_when_bm25_fails(monkeypatch):
    async def failing_bm25(*args, **kwargs):
        raise RuntimeError("bm25 down")

    async def working_vector(*args, **kwargs):
        return [_make_hit("doc_v", "V", 0.9)]

    monkeypatch.setattr("app.retrievers.hybrid_retriever.bm25_search", failing_bm25)
    monkeypatch.setattr("app.retrievers.hybrid_retriever.vector_search", working_vector)

    results = await hybrid_search("kb1", "query", top_k=5)

    assert len(results) == 1
    assert results[0].doc_id == "doc_v"
    assert results[0].vector_score == 0.9


@pytest.mark.asyncio
async def test_hybrid_search_raises_when_both_branches_fail(monkeypatch):
    async def failing_bm25(*args, **kwargs):
        raise RuntimeError("bm25 down")

    async def failing_vector(*args, **kwargs):
        raise RuntimeError("vector down")

    monkeypatch.setattr("app.retrievers.hybrid_retriever.bm25_search", failing_bm25)
    monkeypatch.setattr("app.retrievers.hybrid_retriever.vector_search", failing_vector)

    with pytest.raises(RuntimeError, match="Both retrieval branches failed"):
        await hybrid_search("kb1", "query", top_k=5)
```

If `pytest` is already imported at the top after this edit, keep only one `import pytest`.

- [ ] **Step 3: Run hybrid tests to verify failure**

Run:

```bash
poetry run pytest tests/test_retrievers/test_hybrid_retriever.py -v
```

Expected: FAIL because identity uses `_id` and `asyncio.gather` fails fast.

- [ ] **Step 4: Implement chunk identity helper**

In `app/retrievers/hybrid_retriever.py`, add:

```python
def _hit_identity(hit: dict) -> str:
    source = hit.get("_source", {})
    return source.get("chunk_id") or hit.get("_id", "")
```

In both BM25 and vector loops in `_rrf_merge`, replace:

```python
        doc_id = hit["_id"]
```

with:

```python
        doc_id = _hit_identity(hit)
```

Keep `source.get("doc_id", doc_id)` for the `SearchResultItem.doc_id` value by changing each `RankedDoc` construction to:

```python
                doc_id=source.get("doc_id", doc_id),
```

- [ ] **Step 5: Implement retrieval degradation**

In `hybrid_search`, replace the direct `asyncio.gather(...)` call with:

```python
    bm25_result, vector_result = await asyncio.gather(
        bm25_search(kb_id, query, top_k=pre_k),
        vector_search(kb_id, query, top_k=pre_k),
        return_exceptions=True,
    )

    bm25_hits: list[dict] = []
    vector_hits: list[dict] = []
    failures = []

    if isinstance(bm25_result, Exception):
        failures.append(f"bm25: {bm25_result}")
        logger.exception("BM25 retrieval failed for kb=%s query=%s", kb_id, query)
    else:
        bm25_hits = bm25_result

    if isinstance(vector_result, Exception):
        failures.append(f"vector: {vector_result}")
        logger.exception("Vector retrieval failed for kb=%s query=%s", kb_id, query)
    else:
        vector_hits = vector_result

    if failures and not bm25_hits and not vector_hits:
        raise RuntimeError(f"Both retrieval branches failed: {'; '.join(failures)}")
```

Then keep:

```python
    return _rrf_merge(bm25_hits, vector_hits, k=settings.rrf_k, top_k=top_k)
```

- [ ] **Step 6: Run hybrid tests**

Run:

```bash
poetry run pytest tests/test_retrievers/test_hybrid_retriever.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/retrievers/hybrid_retriever.py tests/test_retrievers/test_hybrid_retriever.py
git commit -m "feat: make hybrid retrieval chunk aware"
```

---

### Task 8: Add Chat Context Budgeting

**Files:**
- Modify: `app/services/chat_service.py`
- Test: `tests/test_services/test_chat_service.py`

- [ ] **Step 1: Write failing chat context tests**

Create `tests/test_services/test_chat_service.py`:

```python
"""Tests for chat service context construction."""

from app.schemas.response import SearchResultItem
from app.services import chat_service
from app.services.chat_service import _build_context


def _source(name: str, content: str, score: float) -> SearchResultItem:
    return SearchResultItem(
        doc_id=name,
        filename=f"{name}.md",
        heading_path=name,
        content=content,
        bm25_score=0.0,
        vector_score=0.0,
        rrf_score=score,
    )


def test_build_context_respects_character_budget(monkeypatch):
    monkeypatch.setattr(chat_service.settings, "chat_context_max_chars", 40)
    sources = [
        _source("low", "低分内容" * 20, 0.1),
        _source("high", "高分内容" * 20, 0.9),
    ]

    context = _build_context(sources)

    assert len(context) <= 40
    assert "high" in context
    assert "low" not in context


def test_build_context_handles_empty_sources():
    assert _build_context([]) == "参考资料为空。"
```

- [ ] **Step 2: Run chat service tests to verify failure**

Run:

```bash
poetry run pytest tests/test_services/test_chat_service.py -v
```

Expected: FAIL because `_build_context` does not budget or return empty-reference text.

- [ ] **Step 3: Implement context budgeting**

In `app/services/chat_service.py`, replace `_build_context` with:

```python
def _build_context(sources: list[SearchResultItem]) -> str:
    """Join search results into prompt context within a character budget."""
    if not sources:
        return "参考资料为空。"

    budget = settings.chat_context_max_chars
    parts: list[str] = []
    used = 0

    for i, source in enumerate(sorted(sources, key=lambda item: item.rrf_score, reverse=True), 1):
        heading = source.heading_path or "无标题"
        part = f"[{i}] {heading}\n{source.content}"
        separator = "\n\n---\n\n" if parts else ""
        remaining = budget - used - len(separator)
        if remaining <= 0:
            break
        if len(part) > remaining:
            part = part[:remaining]
        parts.append(f"{separator}{part}" if separator else part)
        used += len(separator) + len(part)

    return "".join(parts) if parts else "参考资料为空。"
```

- [ ] **Step 4: Run chat service tests**

Run:

```bash
poetry run pytest tests/test_services/test_chat_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/chat_service.py tests/test_services/test_chat_service.py
git commit -m "feat: budget chat retrieval context"
```

---

### Task 9: Update Response Schemas for New Fields

**Files:**
- Modify: `app/schemas/response.py`
- Test: `tests/test_schemas/test_response.py`

- [ ] **Step 1: Write failing schema test**

Create directory and file `tests/test_schemas/test_response.py`:

```python
"""Tests for response schemas."""

from app.schemas.response import SearchResultItem, UploadResponse


def test_search_result_item_accepts_chunk_id():
    item = SearchResultItem(
        chunk_id="chunk1",
        doc_id="doc1",
        filename="a.md",
        heading_path="A",
        content="content",
        bm25_score=1.0,
        vector_score=2.0,
        rrf_score=0.3,
    )

    assert item.chunk_id == "chunk1"


def test_upload_response_accepts_partial_failure_counts():
    response = UploadResponse(
        doc_id="doc1",
        filename="a.md",
        chunks_count=2,
        indexed_chunks_count=1,
        failed_chunks_count=1,
        failed_chunk_ids=["doc1_1"],
    )

    assert response.indexed_chunks_count == 1
    assert response.failed_chunks_count == 1
    assert response.failed_chunk_ids == ["doc1_1"]
```

- [ ] **Step 2: Run schema test to verify failure**

Run:

```bash
poetry run pytest tests/test_schemas/test_response.py -v
```

Expected: FAIL because schema fields do not exist.

- [ ] **Step 3: Add schema fields**

In `app/schemas/response.py`, update the import and models:

```python
from pydantic import BaseModel, Field
```

```python
class SearchResultItem(BaseModel):
    chunk_id: str = ""
    doc_id: str
    filename: str
    heading_path: str
    content: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
```

```python
class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int
    indexed_chunks_count: int = 0
    failed_chunks_count: int = 0
    failed_chunk_ids: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Include chunk_id in RRF output**

In `app/retrievers/hybrid_retriever.py`, add `chunk_id: str = ""` to `RankedDoc`. When constructing `RankedDoc`, set `chunk_id=doc_id` where `doc_id` is the identity from `_hit_identity`. When constructing `SearchResultItem`, pass `chunk_id=d.chunk_id`.

- [ ] **Step 5: Run schema and hybrid tests**

Run:

```bash
poetry run pytest tests/test_schemas/test_response.py tests/test_retrievers/test_hybrid_retriever.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/schemas/response.py app/retrievers/hybrid_retriever.py tests/test_schemas/test_response.py
git commit -m "feat: expose chunk and ingest failure metadata"
```

---

### Task 10: Final Regression and Documentation Check

**Files:**
- Modify: `README.md` if API response examples are stale after implementation.
- Verify: all changed files.

- [ ] **Step 1: Run full test suite**

Run:

```bash
poetry run pytest -v
```

Expected: PASS for the full suite.

- [ ] **Step 2: Run coverage command**

Run:

```bash
poetry run pytest --cov=app --cov-report=term-missing
```

Expected: PASS and coverage report printed. Review uncovered new branches only if they indicate untested error behavior from this plan.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff --stat HEAD~9..HEAD
```

Expected: output includes only app code and tests for this optimization work, plus optional README updates.

- [ ] **Step 4: Update README if response examples changed**

If `UploadResponse` examples or search result fields in `README.md` are stale, update the API examples to include these fields:

```json
{
  "doc_id": "abc123",
  "filename": "test.md",
  "chunks_count": 2,
  "indexed_chunks_count": 2,
  "failed_chunks_count": 0,
  "failed_chunk_ids": []
}
```

Search result example should include:

```json
{
  "chunk_id": "abc123_0",
  "doc_id": "abc123",
  "filename": "test.md",
  "heading_path": "概述",
  "content": "...",
  "bm25_score": 1.2,
  "vector_score": 1.8,
  "rrf_score": 0.03
}
```

- [ ] **Step 5: Commit final docs if changed**

If README changed:

```bash
git add README.md
git commit -m "docs: update optimized rag response examples"
```

If README did not change, do not create an empty commit.

- [ ] **Step 6: Capture final status**

Run:

```bash
git status --short
```

Expected: no unstaged or uncommitted files except user-owned unrelated changes.

---

## Self-Review Checklist

Spec coverage:

- Configuration defaults are covered by Task 1.
- Size-aware chunking is covered by Task 2.
- Mapping fields and mapping version are covered by Task 3.
- Ingest metadata, refresh policy, and bulk failure behavior are covered by Task 4.
- BM25 heading-aware query is covered by Task 5.
- Query embedding cache and ES 7.10 script-score compatibility are covered by Task 6.
- Chunk-level RRF identity and retrieval degradation are covered by Task 7.
- Chat context budget and empty-reference prompt path are covered by Task 8.
- Schema exposure for new response fields is covered by Task 9.
- Full-suite verification and docs sync are covered by Task 10.

Implementation boundaries:

- This plan does not upgrade ES.
- This plan does not change the embedding model.
- This plan does not introduce an external vector database.
- This plan does not add automatic old-index migration.
- Candidate vector mode remains configured but disabled by default; implementation is limited to interfaces that do not block a future candidate-mode task.
