# RAG-ES-Demo 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 ES 7.10 的中文 RAG 问答系统，支持多知识库、Markdown 分块入库、BM25+Vector 两阶段召回+RRF 融合检索、SSE 流式智能问答。

**Architecture:** LangChain 主导 + 自定义 ES Retriever。FastAPI 提供 REST API。多知识库 = 多 ES 索引。Embedding 本地加载，LLM 云端 OpenAI 标准协议。

**Tech Stack:** Python 3.10+, FastAPI, LangChain 0.3, elasticsearch-py v8 (async), sentence-transformers, pydantic-settings, Poetry

---

## 文件地图

| 文件 | 职责 | 新建/修改 |
|------|------|----------|
| `pyproject.toml` | Poetry 依赖管理 | 新建 |
| `.env` | 环境变量配置 | 新建 |
| `app/__init__.py` | 包初始化 | 新建 |
| `app/config.py` | Settings 配置类 | 新建 |
| `app/main.py` | FastAPI 入口、生命周期 | 新建 |
| `app/core/es_client.py` | ES 异步客户端单例、健康检查、索引管理 | 新建 |
| `app/core/embedder.py` | 本地 Embedding 模型单例 | 新建 |
| `app/chunkers/__init__.py` | 包初始化 | 新建 |
| `app/chunkers/markdown_chunker.py` | 按标题层级分块 | 新建 |
| `app/retrievers/__init__.py` | 包初始化 | 新建 |
| `app/retrievers/bm25_retriever.py` | BM25 文本检索 | 新建 |
| `app/retrievers/vector_retriever.py` | 向量 script_score 检索 | 新建 |
| `app/retrievers/hybrid_retriever.py` | RRF 融合 + 去重排序 | 新建 |
| `app/services/__init__.py` | 包初始化 | 新建 |
| `app/services/ingest_service.py` | 文件入库（分块→向量化→bulk） | 新建 |
| `app/services/chat_service.py` | SSE 问答（检索→Prompt→LLM 流式） | 新建 |
| `app/schemas/__init__.py` | 包初始化 | 新建 |
| `app/schemas/request.py` | 请求体 Pydantic 模型 | 新建 |
| `app/schemas/response.py` | 响应体 + SSE 事件模型 | 新建 |
| `app/routers/__init__.py` | 包初始化 | 新建 |
| `app/routers/knowledge_base.py` | /kb CRUD 路由 | 新建 |
| `app/routers/upload.py` | /upload 路由 | 新建 |
| `app/routers/search.py` | /search 路由 | 新建 |
| `app/routers/chat.py` | /chat SSE 路由 | 新建 |
| `tests/conftest.py` | pytest fixtures | 新建 |
| `tests/test_chunkers/test_markdown_chunker.py` | 分块单元测试 | 新建 |
| `tests/test_retrievers/test_hybrid_retriever.py` | RRF 单元测试 | 新建 |
| `tests/test_services/test_ingest_service.py` | 入库服务测试 | 新建 |
| `tests/test_routers/test_knowledge_base.py` | 知识库 API 测试 | 新建 |
| `tests/test_routers/test_upload.py` | 上传 API 测试 | 新建 |
| `tests/test_routers/test_search.py` | 检索 API 测试 | 新建 |
| `tests/test_routers/test_chat.py` | SSE 问答 API 测试 | 新建 |
| `README.md` | 项目文档 | 新建 |

---

## Task 1: 项目初始化 — 依赖、配置、目录结构

**Files:**
- Create: `pyproject.toml`
- Create: `.env`
- Create: `app/__init__.py`
- Create: `data/uploads/.gitkeep`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[tool.poetry]
name = "rag-es-demo"
version = "0.1.0"
description = "RAG demo with Elasticsearch and LangChain"
authors = []

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.115"
uvicorn = { extras = ["standard"], version = "^0.34" }
python-dotenv = "^1.0"
pydantic-settings = "^2.0"
elasticsearch = { extras = ["async"], version = "^8.0" }
langchain = "^0.3"
langchain-openai = "^0.3"
langchain-core = "^0.3"
sentence-transformers = "^3.0"
python-multipart = "^0.0.18"
httpx = "^0.28"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.24"
pytest-cov = "^6.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: 创建 .env**

```env
# ES 配置
ES_HOST=localhost
ES_PORT=9200
ES_USER=elastic
ES_PASSWORD=Cass@123456

# Embedding 配置
EMBEDDING_MODEL_PATH=data/models/bge-small-zh-v1.5
EMBEDDING_DIM=512
EMBEDDING_BATCH_SIZE=32

# LLM 配置 (OpenAI 标准协议)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=30

# RRF 配置
RRF_K=60
RETRIEVER_TOP_K=5
RETRIEVER_PRE_MULTIPLIER=2

# 服务配置
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# 上传配置
UPLOAD_DIR=data/uploads
MAX_UPLOAD_SIZE_MB=50
```

- [ ] **Step 3: 创建空包文件**

```python
# app/__init__.py
# app package
```

```
# data/uploads/.gitkeep
(空文件，仅用于 git 跟踪)
```

- [ ] **Step 4: 安装依赖**

```bash
cd /root/workspace/code/mu/github/rag-es-demo
poetry install
```

Expected: Dependencies installed successfully. `poetry show` lists all packages.

---

## Task 2: 配置与数据模型

**Files:**
- Create: `app/config.py`
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/request.py`
- Create: `app/schemas/response.py`
- Test: `tests/test_schemas/test_request_validation.py` (implicit, tested via router tests)

- [ ] **Step 1: 创建 app/config.py**

```python
"""Application configuration loaded from .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ES
    es_host: str = "localhost"
    es_port: int = 9200
    es_user: str = "elastic"
    es_password: str = "Cass@123456"

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"

    # Embedding
    embedding_model_path: str = "data/models/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 32

    # LLM (OpenAI 标准协议)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_timeout: int = 30

    # RRF
    rrf_k: int = 60
    retriever_top_k: int = 5
    retriever_pre_multiplier: int = 2

    # Service
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # Upload
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# 全局单例
settings = Settings()
```

- [ ] **Step 2: 创建 app/schemas/request.py**

```python
"""Request body models for API endpoints."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="搜索查询文本")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="用户提问")
    top_k: int = Field(default=5, ge=1, le=20, description="检索参考文档数")


class CreateKbRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="知识库名称")
    description: str = Field(default="", max_length=512, description="知识库描述")
```

- [ ] **Step 3: 创建 app/schemas/response.py**

```python
"""Response body and SSE event models."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    doc_id: str
    filename: str
    heading_path: str
    content: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


class KbInfo(BaseModel):
    kb_id: str
    name: str
    description: str
    index_name: str
    doc_count: int = 0
    created_at: str = ""


class KbCreateResponse(BaseModel):
    kb_id: str
    index_name: str
    created_at: str


class KbListResponse(BaseModel):
    knowledge_bases: list[KbInfo]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int


class SourceDoc(BaseModel):
    """引用文档，用于 SSE events 和问答响应。"""
    doc_id: str
    filename: str
    heading_path: str
    content: str
    score: float


class ChatDoneData(BaseModel):
    answer: str
    sources: list[SourceDoc]


class ChatErrorData(BaseModel):
    message: str
```

- [ ] **Step 4: 创建 app/schemas/__init__.py**

```python
from .request import ChatRequest, CreateKbRequest, SearchRequest
from .response import (
    ChatDoneData,
    ChatErrorData,
    KbCreateResponse,
    KbInfo,
    KbListResponse,
    SearchResultItem,
    SearchResponse,
    SourceDoc,
    UploadResponse,
)

__all__ = [
    "ChatDoneData",
    "ChatErrorData",
    "ChatRequest",
    "CreateKbRequest",
    "KbCreateResponse",
    "KbInfo",
    "KbListResponse",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "SourceDoc",
    "UploadResponse",
]
```

---

## Task 3: ES 客户端单例

**Files:**
- Create: `app/core/es_client.py`
- Test: 集成在 router 测试中

- [ ] **Step 1: 创建 app/core/es_client.py**

```python
"""Elasticsearch async client singleton with lifecycle management."""

import logging
from uuid import uuid4

from elasticsearch import AsyncElasticsearch

from app.config import settings
from app.schemas.response import KbInfo

logger = logging.getLogger(__name__)

# 全局 ES 客户端实例
_es_client: AsyncElasticsearch | None = None

# ES 索引 Mapping 模板
KB_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "heading_path": {"type": "text", "analyzer": "ik_max_word"},
            "chunk_index": {"type": "integer"},
            "content": {"type": "text", "analyzer": "ik_max_word"},
            "embedding": {"type": "dense_vector", "dims": 512},
            "created_at": {"type": "date"},
        }
    }
}

INDEX_PREFIX = "rag_kb_"


def _index_name(kb_id: str) -> str:
    return f"{INDEX_PREFIX}{kb_id}"


def _kb_id_from_index(index_name: str) -> str:
    return index_name.removeprefix(INDEX_PREFIX)


async def init_es_client() -> AsyncElasticsearch:
    """初始化 ES 连接并执行健康检查。启动时调用。"""
    global _es_client
    if _es_client is not None:
        return _es_client

    _es_client = AsyncElasticsearch(
        hosts=[settings.es_url],
        basic_auth=(settings.es_user, settings.es_password),
        request_timeout=10,
    )

    if not await _es_client.ping():
        logger.error("Cannot connect to Elasticsearch at %s", settings.es_url)
        await _es_client.close()
        _es_client = None
        raise ConnectionError(f"Cannot connect to Elasticsearch at {settings.es_url}")

    logger.info("Connected to Elasticsearch at %s", settings.es_url)
    return _es_client


async def get_es_client() -> AsyncElasticsearch:
    """获取 ES 客户端实例。"""
    if _es_client is None:
        raise ConnectionError("Elasticsearch client not initialized. Call init_es_client() first.")
    return _es_client


async def close_es_client() -> None:
    """关闭 ES 连接。"""
    global _es_client
    if _es_client is not None:
        await _es_client.close()
        _es_client = None
        logger.info("Elasticsearch connection closed")


async def create_kb_index(name: str, description: str) -> KbInfo:
    """创建知识库索引。返回 KbInfo。"""
    client = await get_es_client()
    kb_id = uuid4().hex
    index = _index_name(kb_id)

    await client.indices.create(
        index=index,
        body=KB_INDEX_MAPPING,
    )

    return KbInfo(
        kb_id=kb_id,
        name=name,
        description=description,
        index_name=index,
        doc_count=0,
    )


async def list_kb_indices() -> list[KbInfo]:
    """列出所有 rag_kb_* 索引。"""
    client = await get_es_client()
    indices = await client.cat.indices(
        index=f"{INDEX_PREFIX}*",
        format="json",
        h="index,docs.count",
    )

    result = []
    for idx in indices:
        idx_name = idx.get("index", "")
        doc_count = int(idx.get("docs.count", 0) or 0)
        result.append(
            KbInfo(
                kb_id=_kb_id_from_index(idx_name),
                name="",  # name/description 可后续扩展为 metadata 存储
                description="",
                index_name=idx_name,
                doc_count=doc_count,
            )
        )
    return result


async def get_kb_info(kb_id: str) -> KbInfo | None:
    """获取单个知识库信息。不存在时返回 None。"""
    client = await get_es_client()
    index = _index_name(kb_id)

    exists = await client.indices.exists(index=index)
    if not exists:
        return None

    stats = await client.cat.indices(index=index, format="json", h="index,docs.count")
    idx = stats[0] if stats else {}
    doc_count = int(idx.get("docs.count", 0) or 0)

    return KbInfo(
        kb_id=kb_id,
        name="",
        description="",
        index_name=index,
        doc_count=doc_count,
    )


async def delete_kb_index(kb_id: str) -> bool:
    """删除知识库索引。成功返回 True。"""
    client = await get_es_client()
    index = _index_name(kb_id)

    exists = await client.indices.exists(index=index)
    if not exists:
        return False

    await client.indices.delete(index=index)
    return True
```

---

## Task 4: Embedding 单例

**Files:**
- Create: `app/core/embedder.py`
- Test: 集成在 service 测试中

- [ ] **Step 1: 创建 app/core/embedder.py**

```python
"""Local embedding model singleton using sentence-transformers."""

import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """获取 Embedding 模型实例。启动时预加载。"""
    global _model
    if _model is None:
        logger.info("Loading embedding model from %s", settings.embedding_model_path)
        _model = SentenceTransformer(settings.embedding_model_path)
        logger.info("Embedding model loaded successfully")
    return _model


def encode_texts(texts: list[str]) -> list[list[float]]:
    """将文本列表批量编码为向量。"""
    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
```

- [ ] **Step 2: 创建 app/core/__init__.py**

```python
from .embedder import encode_texts, get_embedder
from .es_client import (
    close_es_client,
    create_kb_index,
    delete_kb_index,
    get_es_client,
    get_kb_info,
    init_es_client,
    list_kb_indices,
)

__all__ = [
    "close_es_client",
    "create_kb_index",
    "delete_kb_index",
    "encode_texts",
    "get_embedder",
    "get_es_client",
    "get_kb_info",
    "init_es_client",
    "list_kb_indices",
]
```

---

## Task 5: Markdown 分块器 + 测试

**Files:**
- Create: `app/chunkers/__init__.py`
- Create: `app/chunkers/markdown_chunker.py`
- Create: `tests/__init__.py`
- Create: `tests/test_chunkers/__init__.py`
- Create: `tests/test_chunkers/test_markdown_chunker.py`

- [ ] **Step 1: 创建 app/chunkers/markdown_chunker.py**

```python
"""Markdown chunker that splits by heading hierarchy (# / ## / ###)."""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    heading_path: str
    chunk_index: int


# 匹配 1-6 级标题
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_markdown(text: str) -> list[Chunk]:
    """按 Markdown 标题层级分块。

    每个标题及其下方内容（直到下一个同级或上级标题）作为一个 chunk。
    heading_path 记录层级路径，如 "一级标题 > 二级标题"。
    """
    lines = text.split("\n")
    chunks: list[Chunk] = []

    # 收集所有标题行及其位置
    headings: list[tuple[int, int, str]] = []  # (line_index, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((i, level, title))

    if not headings:
        # 无标题的文档，整个内容作为一个 chunk
        content = text.strip()
        if content:
            chunks.append(Chunk(content=content, heading_path="", chunk_index=0))
        return chunks

    # 按标题分割内容
    for idx, (line_idx, level, title) in enumerate(headings):
        # 确定当前标题的范围
        start = line_idx
        if idx + 1 < len(headings):
            end = headings[idx + 1][0]
        else:
            end = len(lines)

        section_lines = lines[start:end]
        content = "\n".join(section_lines).strip()

        # 构建 heading_path：收集所有父级标题
        path_parts = _build_heading_path(headings, idx)
        heading_path = " > ".join(path_parts)

        chunks.append(Chunk(content=content, heading_path=heading_path, chunk_index=idx))

    return chunks


def _build_heading_path(
    headings: list[tuple[int, int, str]], current_idx: int
) -> list[str]:
    """构建从根到当前标题的层级路径。"""
    _, current_level, current_title = headings[current_idx]
    path = [current_title]

    # 向上查找父级标题（level 更大的标题可能是子标题，我们要找 level 更小的）
    for i in range(current_idx - 1, -1, -1):
        _, level, title = headings[i]
        if level < current_level:
            path.insert(0, title)
            current_level = level

    return path
```

- [ ] **Step 2: 创建 app/chunkers/__init__.py**

```python
from .markdown_chunker import Chunk, chunk_markdown

__all__ = ["Chunk", "chunk_markdown"]
```

- [ ] **Step 3: 创建 tests/__init__.py**

```python
# tests package
```

- [ ] **Step 4: 创建 tests/test_chunkers/__init__.py**

```python
# test_chunkers package
```

- [ ] **Step 5: 创建 tests/test_chunkers/test_markdown_chunker.py**

```python
"""Tests for Markdown heading-based chunker."""

from app.chunkers import chunk_markdown


def test_single_heading():
    text = """# 概述

这是概述内容。
"""
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].content == "# 概述\n\n这是概述内容。"
    assert chunks[0].heading_path == "概述"
    assert chunks[0].chunk_index == 0


def test_nested_headings():
    text = """# 第一章

第一章引言。

## 第一节

第一节内容。

### 小节

小节内容。

## 第二节

第二节内容。
"""
    chunks = chunk_markdown(text)
    assert len(chunks) == 4

    # 第一章
    assert chunks[0].heading_path == "第一章"
    assert "第一章引言" in chunks[0].content

    # 第一节
    assert chunks[1].heading_path == "第一章 > 第一节"
    assert "第一节内容" in chunks[1].content

    # 小节
    assert chunks[2].heading_path == "第一章 > 第一节 > 小节"
    assert "小节内容" in chunks[2].content

    # 第二节
    assert chunks[3].heading_path == "第一章 > 第二节"
    assert "第二节内容" in chunks[3].content


def test_no_headings():
    text = "这是一段没有标题的文本。"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].content == "这是一段没有标题的文本。"
    assert chunks[0].heading_path == ""


def test_empty_text():
    chunks = chunk_markdown("")
    assert len(chunks) == 0


def test_deep_nesting():
    text = """# A

## B

### C

#### D

D 的内容。
"""
    chunks = chunk_markdown(text)
    assert len(chunks) == 4
    assert chunks[3].heading_path == "A > B > C > D"
```

- [ ] **Step 6: 运行测试验证**

```bash
cd /root/workspace/code/mu/github/rag-es-demo
poetry run pytest tests/test_chunkers/ -v
```

Expected: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env app/ tests/ data/
git commit -m "feat: add project setup, config, schemas, ES client, embedder, chunker"
```

---

## Task 6: BM25 检索器

**Files:**
- Create: `app/retrievers/__init__.py`
- Create: `app/retrievers/bm25_retriever.py`

- [ ] **Step 1: 创建 app/retrievers/bm25_retriever.py**

```python
"""BM25 text retriever for Elasticsearch."""

import logging

from app.core import get_es_client

logger = logging.getLogger(__name__)


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


async def bm25_search(
    kb_id: str,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """执行 BM25 文本检索。

    返回 ES 原始 hits 列表，每个 hit 包含 _source 和 _score。
    """
    client = await get_es_client()
    index = _index_name(kb_id)

    body = {
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or",
                }
            }
        },
        "size": top_k,
    }

    resp = await client.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    logger.debug("BM25 search for kb=%s query=%s returned %d results", kb_id, query, len(hits))
    return hits
```

- [ ] **Step 2: 创建 app/retrievers/__init__.py**

```python
from .bm25_retriever import bm25_search
from .hybrid_retriever import hybrid_search
from .vector_retriever import vector_search

__all__ = ["bm25_search", "hybrid_search", "vector_search"]
```

---

## Task 7: 向量检索器

**Files:**
- Create: `app/retrievers/vector_retriever.py`

- [ ] **Step 1: 创建 app/retrievers/vector_retriever.py**

```python
"""Vector retriever using ES 7.10 script_score for cosine similarity."""

import logging

from app.config import settings
from app.core import encode_texts, get_es_client

logger = logging.getLogger(__name__)

# script_score 模板：cosineSimilarity + 1.0 确保分数为正
SCRIPT_SOURCE = "cosineSimilarity(params.query_vector, 'embedding') + 1.0"


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


async def vector_search(
    kb_id: str,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """执行向量检索（script_score cosine similarity）。

    1. 将查询文本向量化
    2. 使用 script_score 在 ES 中暴力扫描
    3. 返回 hits 列表

    注意：ES 7.10 不支持原生 knn，必须用 script_score。
    """
    # 向量化查询文本
    query_vector = encode_texts([query])[0]

    client = await get_es_client()
    index = _index_name(kb_id)

    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": SCRIPT_SOURCE,
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "size": top_k,
    }

    resp = await client.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    logger.debug(
        "Vector search for kb=%s query=%s returned %d results", kb_id, query, len(hits)
    )
    return hits
```

---

## Task 8: RRF 混合检索器 + 测试

**Files:**
- Create: `app/retrievers/hybrid_retriever.py`
- Create: `tests/test_retrievers/__init__.py`
- Create: `tests/test_retrievers/test_hybrid_retriever.py`

- [ ] **Step 1: 创建 app/retrievers/hybrid_retriever.py**

```python
"""Hybrid retriever with RRF (Reciprocal Rank Fusion) ranking."""

import logging
from dataclasses import dataclass

from app.config import settings
from app.schemas.response import SearchResultItem

from .bm25_retriever import bm25_search
from .vector_retriever import vector_search

logger = logging.getLogger(__name__)


@dataclass
class RankedDoc:
    doc_id: str
    filename: str
    heading_path: str
    content: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0


async def hybrid_search(
    kb_id: str,
    query: str,
    top_k: int | None = None,
) -> list[SearchResultItem]:
    """两阶段召回 + RRF 排名融合。

    1. BM25 和向量检索并行，各取 pre_k 条
    2. 按 doc_id 去重
    3. RRF 分数融合: 1/(k+bm25_rank) + 1/(k+vector_rank)
    4. 按 RRF 分数降序取 top_k
    """
    if top_k is None:
        top_k = settings.retriever_top_k
    pre_k = top_k * settings.retriever_pre_multiplier

    # 并行执行两路检索
    import asyncio

    bm25_hits, vector_hits = await asyncio.gather(
        bm25_search(kb_id, query, top_k=pre_k),
        vector_search(kb_id, query, top_k=pre_k),
    )

    return _rrf_merge(bm25_hits, vector_hits, k=settings.rrf_k, top_k=top_k)


def _rrf_merge(
    bm25_hits: list[dict],
    vector_hits: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResultItem]:
    """RRF 融合两个检索结果列表。"""
    docs: dict[str, RankedDoc] = {}

    # 处理 BM25 结果
    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["_id"]
        source = hit.get("_source", {})
        if doc_id not in docs:
            docs[doc_id] = RankedDoc(
                doc_id=doc_id,
                filename=source.get("filename", ""),
                heading_path=source.get("heading_path", ""),
                content=source.get("content", ""),
                bm25_score=hit.get("_score", 0.0),
                bm25_rank=rank,
            )
        else:
            docs[doc_id].bm25_rank = rank
            docs[doc_id].bm25_score = hit.get("_score", 0.0)

    # 处理 Vector 结果
    for rank, hit in enumerate(vector_hits, start=1):
        doc_id = hit["_id"]
        source = hit.get("_source", {})
        if doc_id not in docs:
            docs[doc_id] = RankedDoc(
                doc_id=doc_id,
                filename=source.get("filename", ""),
                heading_path=source.get("heading_path", ""),
                content=source.get("content", ""),
                vector_score=hit.get("_score", 0.0) if hit.get("_score") else 0.0,
                vector_rank=rank,
            )
        else:
            docs[doc_id].vector_rank = rank
            # script_score 的分数 = cosineSimilarity + 1.0, 范围 0~2
            docs[doc_id].vector_score = hit.get("_score", 0.0) if hit.get("_score") else 0.0

    # 计算 RRF 分数
    for doc in docs.values():
        bm25_part = 1.0 / (k + doc.bm25_rank) if doc.bm25_rank else 0.0
        vector_part = 1.0 / (k + doc.vector_rank) if doc.vector_rank else 0.0
        doc.rrf_score = bm25_part + vector_part

    # 按 RRF 分数降序排序，取 top_k
    ranked = sorted(docs.values(), key=lambda d: d.rrf_score, reverse=True)[:top_k]

    return [
        SearchResultItem(
            doc_id=d.doc_id,
            filename=d.filename,
            heading_path=d.heading_path,
            content=d.content,
            bm25_score=d.bm25_score,
            vector_score=d.vector_score,
            rrf_score=d.rrf_score,
        )
        for d in ranked
    ]
```

- [ ] **Step 2: 创建 tests/test_retrievers/__init__.py**

```python
# test_retrievers package
```

- [ ] **Step 3: 创建 tests/test_retrievers/test_hybrid_retriever.py**

```python
"""Tests for RRF hybrid retriever (pure logic, no ES required)."""

from app.retrievers.hybrid_retriever import _rrf_merge


def _make_hit(doc_id: str, content: str, score: float = 1.0) -> dict:
    """辅助函数：构造 ES hit 对象。"""
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "content": content,
            "filename": f"{doc_id}.md",
            "heading_path": f"Heading for {doc_id}",
            "chunk_index": 0,
        },
    }


def test_rrf_basic_merge():
    """基本 RRF 融合：两路有重叠文档。"""
    bm25 = [_make_hit("doc_a", "A", 5.0), _make_hit("doc_b", "B", 4.0), _make_hit("doc_c", "C", 3.0)]
    vector = [_make_hit("doc_b", "B", 0.9), _make_hit("doc_a", "A", 0.8), _make_hit("doc_d", "D", 0.7)]

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    assert len(results) == 4
    # doc_b 在两路都排第 1 或 2，应该 RRF 最高
    assert results[0].doc_id == "doc_b"
    # doc_a 也在两路，应该第二
    assert results[1].doc_id == "doc_a"


def test_rrf_non_overlapping():
    """两路完全不重叠的文档。"""
    bm25 = [_make_hit("doc_a", "A", 5.0), _make_hit("doc_b", "B", 4.0)]
    vector = [_make_hit("doc_c", "C", 0.9), _make_hit("doc_d", "D", 0.8)]

    results = _rrf_merge(bm25, vector, k=60, top_k=4)

    assert len(results) == 4
    # doc_a (bm25 rank 1) vs doc_c (vector rank 1): 分数相同
    assert results[0].rrf_score == results[1].rrf_score


def test_rrf_only_one_source():
    """只有一路有结果。"""
    bm25 = [_make_hit("doc_a", "A", 5.0), _make_hit("doc_b", "B", 4.0)]
    vector: list[dict] = []

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    assert len(results) == 2
    assert results[0].doc_id == "doc_a"
    assert results[0].bm25_score == 5.0
    assert results[0].vector_score == 0.0


def test_rrf_top_k_limit():
    """top_k 限制生效。"""
    bm25 = [_make_hit(f"doc_{i}", f"C{i}", float(10 - i)) for i in range(10)]
    vector: list[dict] = []

    results = _rrf_merge(bm25, vector, k=60, top_k=3)

    assert len(results) == 3
    assert results[0].doc_id == "doc_0"


def test_rrf_score_formula():
    """验证 RRF 分数计算：doc 在两路都 rank=1, k=60。"""
    bm25 = [_make_hit("doc_x", "X", 5.0)]
    vector = [_make_hit("doc_x", "X", 0.9)]

    results = _rrf_merge(bm25, vector, k=60, top_k=5)

    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert abs(results[0].rrf_score - expected) < 1e-10
```

- [ ] **Step 4: 运行测试验证**

```bash
cd /root/workspace/code/mu/github/rag-es-demo
poetry run pytest tests/test_retrievers/ -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/retrievers/ tests/test_retrievers/
git commit -m "feat: add BM25, vector, and hybrid RRF retrievers with tests"
```

---

## Task 9: IngestService 入库服务 + 测试

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/ingest_service.py`
- Create: `tests/test_services/__init__.py`
- Create: `tests/test_services/test_ingest_service.py`

- [ ] **Step 1: 创建 app/services/__init__.py**

```python
from .chat_service import ChatService
from .ingest_service import IngestService

__all__ = ["ChatService", "IngestService"]
```

- [ ] **Step 2: 创建 app/services/ingest_service.py**

```python
"""Service for ingesting Markdown files into Elasticsearch."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.chunkers import chunk_markdown
from app.core import encode_texts, get_es_client

logger = logging.getLogger(__name__)


def _index_name(kb_id: str) -> str:
    return f"rag_kb_{kb_id}"


async def ingest_markdown(kb_id: str, filename: str, content: str) -> dict:
    """将 Markdown 内容分块、向量化、写入 ES。

    Args:
        kb_id: 知识库 ID
        filename: 上传文件名
        content: Markdown 文本内容

    Returns:
        {doc_id, filename, chunks_count}
    """
    # 1. 分块
    chunks = chunk_markdown(content)
    if not chunks:
        return {"doc_id": "", "filename": filename, "chunks_count": 0}

    # 2. 向量化
    texts = [c.content for c in chunks]
    embeddings = encode_texts(texts)

    # 3. 构建 bulk 操作
    doc_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    index = _index_name(kb_id)

    actions = []
    for chunk, embedding in zip(chunks, embeddings):
        actions.append({"index": {"_index": index, "_id": f"{doc_id}_{chunk.chunk_index}"}})
        actions.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "heading_path": chunk.heading_path,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": embedding,
                "created_at": now,
            }
        )

    # 4. Bulk 写入
    client = await get_es_client()
    resp = await client.bulk(operations=actions, refresh="wait_for")

    if resp.get("errors"):
        failed = [item for item in resp.get("items", []) if "error" in item.get("index", {})]
        logger.error("Bulk indexing errors: %s", failed)

    logger.info("Ingested %d chunks for doc=%s file=%s kb=%s", len(chunks), doc_id, filename, kb_id)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
    }
```

- [ ] **Step 3: 创建 tests/test_services/__init__.py**

```python
# test_services package
```

- [ ] **Step 4: 创建 tests/test_services/test_ingest_service.py**

```python
"""Tests for IngestService with mocked ES and embedder."""

import unittest.mock as mock

import pytest

from app.services.ingest_service import ingest_markdown


@pytest.mark.asyncio
async def test_ingest_markdown_chunks_and_embeds():
    """验证分块→向量化→bulk 写入流程。"""
    content = """# 标题一

内容一。

## 标题二

内容二。
"""
    mock_bulk_response = {
        "errors": False,
        "items": [
            {"index": {"_id": "test_0", "status": 201}},
            {"index": {"_id": "test_1", "status": 201}},
        ],
    }

    with mock.patch("app.services.ingest_service.get_es_client") as mock_get_es, \
         mock.patch("app.services.ingest_service.encode_texts") as mock_encode:

        mock_client = mock.AsyncMock()
        mock_client.bulk = mock.AsyncMock(return_value=mock_bulk_response)
        # get_es_client is async, return the mock client
        mock_get_es.return_value = mock_client
        mock_encode.return_value = [[0.1] * 512, [0.2] * 512]

        result = await ingest_markdown("test-kb-id", "test.md", content)

        assert result["doc_id"] != ""
        assert result["filename"] == "test.md"
        assert result["chunks_count"] == 2

        # 验证 encode_texts 被调用
        mock_encode.assert_called_once()
        # 验证 bulk 被调用
        mock_client.bulk.assert_called_once()
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /root/workspace/code/mu/github/rag-es-demo
poetry run pytest tests/test_services/ -v
```

Expected: 1 test pass.

---

## Task 10: ChatService 问答服务

**Files:**
- Create: `app/services/chat_service.py`

- [ ] **Step 1: 创建 app/services/chat_service.py**

```python
"""Chat service for RAG-based Q&A with SSE streaming."""

import json
import logging
from typing import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import settings
from app.retrievers import hybrid_search
from app.schemas.response import ChatDoneData, ChatErrorData, SearchResultItem, SourceDoc

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是智能知识库助手，请严格根据以下参考资料回答问题。
如果参考资料中没有相关信息，请明确说明"根据现有资料无法回答"。
回答要准确、简洁，使用中文。"""


def _build_context(sources: list[SearchResultItem]) -> str:
    """将检索结果拼接为 Prompt 上下文。"""
    parts = []
    for i, s in enumerate(sources, 1):
        heading = s.heading_path or "无标题"
        parts.append(f"[{i}] {heading}\n{s.content}")
    return "\n\n---\n\n".join(parts)


def _to_source_docs(sources: list[SearchResultItem]) -> list[SourceDoc]:
    return [
        SourceDoc(
            doc_id=s.doc_id,
            filename=s.filename,
            heading_path=s.heading_path,
            content=s.content,
            score=s.rrf_score,
        )
        for s in sources
    ]


def _sse_event(event_type: str, data: dict | str) -> str:
    """格式化 SSE 事件。"""
    return f"event: message\ndata: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


async def chat_stream(
    kb_id: str,
    query: str,
    top_k: int = 5,
) -> AsyncIterator[str]:
    """SSE 流式问答。

    Yields:
        SSE 格式化的字符串，包含 sources、token、done 或 error 事件。
    """
    try:
        # 1. 检索
        sources = await hybrid_search(kb_id, query, top_k=top_k)

        # 2. 发送引用文档
        source_docs = _to_source_docs(sources)
        yield _sse_event("sources", [doc.model_dump() for doc in source_docs])

        # 3. 构建 Prompt
        context = _build_context(sources)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{query}"),
        ]

        # 4. 初始化 LLM
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,
            timeout=settings.llm_timeout,
        )

        # 5. 流式生成回答
        answer_parts = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                answer_parts.append(chunk.content)
                yield _sse_event("token", chunk.content)

        # 6. 发送完成事件
        full_answer = "".join(answer_parts)
        done_data = ChatDoneData(answer=full_answer, sources=source_docs)
        yield _sse_event("done", done_data.model_dump())

    except Exception as e:
        logger.exception("Chat stream error for kb=%s query=%s", kb_id, query)
        error_data = ChatErrorData(message=str(e))
        yield _sse_event("error", error_data.model_dump())
```

---

## Task 11: 路由层 — 知识库 CRUD + 上传 + 检索

**Files:**
- Create: `app/routers/__init__.py`
- Create: `app/routers/knowledge_base.py`
- Create: `app/routers/upload.py`
- Create: `app/routers/search.py`
- Create: `tests/test_routers/__init__.py`
- Create: `tests/test_routers/test_knowledge_base.py`
- Create: `tests/test_routers/test_upload.py`
- Create: `tests/test_routers/test_search.py`

- [ ] **Step 1: 创建 app/routers/__init__.py**

```python
from .chat import router as chat_router
from .knowledge_base import router as kb_router
from .search import router as search_router
from .upload import router as upload_router

__all__ = ["chat_router", "kb_router", "search_router", "upload_router"]
```

- [ ] **Step 2: 创建 app/routers/knowledge_base.py**

```python
"""Knowledge base CRUD endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from app.core import create_kb_index, delete_kb_index, get_kb_info, list_kb_indices
from app.schemas import CreateKbRequest, KbCreateResponse, KbInfo, KbListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


@router.post("", response_model=KbCreateResponse, status_code=201)
async def create_kb(body: CreateKbRequest):
    """创建知识库。"""
    try:
        info = await create_kb_index(name=body.name, description=body.description)
        return KbCreateResponse(kb_id=info.kb_id, index_name=info.index_name, created_at="")
    except Exception as e:
        logger.exception("Failed to create knowledge base")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=KbListResponse)
async def list_kbs():
    """列出所有知识库。"""
    try:
        kbs = await list_kb_indices()
        return KbListResponse(knowledge_bases=kbs)
    except Exception as e:
        logger.exception("Failed to list knowledge bases")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id}", response_model=KbInfo)
async def get_kb(kb_id: str):
    """获取知识库详情。"""
    info = await get_kb_info(kb_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
    return info


@router.delete("/{kb_id}", status_code=204)
async def delete_kb(kb_id: str):
    """删除知识库。"""
    success = await delete_kb_index(kb_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
```

- [ ] **Step 3: 创建 app/routers/upload.py**

```python
"""File upload endpoint for knowledge base ingestion."""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core import get_kb_info
from app.schemas import UploadResponse
from app.services.ingest_service import ingest_markdown

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/upload", tags=["upload"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(kb_id: str, file: UploadFile):
    """上传 Markdown 文件到知识库。"""
    # 校验扩展名
    if not file.filename or not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported")

    # 校验知识库存在
    info = await get_kb_info(kb_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")

    # 读取内容
    content_bytes = await file.read()
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8")

    # 入库
    try:
        result = await ingest_markdown(kb_id, file.filename, content)
        return UploadResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunks_count=result["chunks_count"],
        )
    except Exception as e:
        logger.exception("Ingest failed for file=%s kb=%s", file.filename, kb_id)
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: 创建 app/routers/search.py**

```python
"""Search endpoint for hybrid retrieval."""

import logging

from fastapi import APIRouter, HTTPException

from app.core import get_kb_info
from app.retrievers import hybrid_search
from app.schemas import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(kb_id: str, body: SearchRequest):
    """混合检索（BM25 + Vector + RRF）。"""
    # 校验知识库存在
    info = await get_kb_info(kb_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")

    try:
        results = await hybrid_search(kb_id, body.query, top_k=body.top_k)
        return SearchResponse(results=results, total=len(results))
    except Exception as e:
        logger.exception("Search failed for kb=%s query=%s", kb_id, body.query)
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 6: 创建 tests/conftest.py**

```python
"""Pytest fixtures and test app configuration."""

import pytest
from fastapi.testclient import TestClient


def create_test_app():
    """创建跳过 lifespan 的测试应用，避免测试时连接真实 ES。"""
    from app.main import create_app as _create_app
    app = _create_app()
    # 替换 lifespan 为无操作，避免测试时连接 ES
    app.router.lifespan_context = lambda app: _null_lifespan()
    return app


async def _null_lifespan():
    """空 lifespan，不做任何初始化。"""
    yield


@pytest.fixture
def client():
    """提供 TestClient 实例。"""
    app = create_test_app()
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 7: 创建 tests/test_routers/test_knowledge_base.py**

```python
"""Tests for knowledge base CRUD endpoints (integration with mock ES)."""

import unittest.mock as mock

from fastapi.testclient import TestClient

from tests.conftest import create_test_app


def test_create_kb_returns_201():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.create_kb_index") as mock_create:
        mock_create.return_value = mock.MagicMock(
            kb_id="test123",
            index_name="rag_kb_test123",
            name="Test KB",
            description="",
            doc_count=0,
        )

        resp = client.post("/kb", json={"name": "Test KB"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["kb_id"] == "test123"
        assert data["index_name"] == "rag_kb_test123"


def test_list_kbs_returns_list():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.knowledge_base.list_kb_indices") as mock_list:
        mock_list.return_value = []
        resp = client.get("/kb")
        assert resp.status_code == 200
        assert "knowledge_bases" in resp.json()
```

- [ ] **Step 8: 创建 tests/test_routers/test_upload.py**

```python
"""Tests for file upload endpoint."""

from fastapi.testclient import TestClient

from tests.conftest import create_test_app


def test_upload_non_md_file_returns_400():
    app = create_test_app()
    client = TestClient(app)

    # 上传 .txt 文件
    resp = client.post(
        "/kb/testkb/upload",
        files={"file": ("test.txt", b"content", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Only .md files" in resp.json()["detail"]
```

- [ ] **Step 9: 创建 tests/test_routers/test_search.py**

```python
"""Tests for search endpoint."""

import unittest.mock as mock

from fastapi.testclient import TestClient

from tests.conftest import create_test_app


def test_search_returns_404_for_missing_kb():
    app = create_test_app()
    client = TestClient(app)

    with mock.patch("app.routers.search.get_kb_info") as mock_get:
        mock_get.return_value = None
        resp = client.post("/kb/nonexistent/search", json={"query": "test"})
        assert resp.status_code == 404
```

---

## Task 12: Chat SSE 路由 + Main 入口

**Files:**
- Create: `app/routers/chat.py`
- Create: `app/main.py`
- Create: `tests/test_routers/test_chat.py`

- [ ] **Step 1: 创建 app/routers/chat.py**

```python
"""SSE chat endpoint for RAG-based Q&A."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core import get_kb_info
from app.schemas import ChatRequest
from app.services.chat_service import chat_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/{kb_id}/chat", tags=["chat"])


@router.post("", response_class=StreamingResponse)
async def chat(kb_id: str, body: ChatRequest):
    """SSE 流式智能问答。"""
    # 校验知识库存在
    info = await get_kb_info(kb_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")

    return StreamingResponse(
        chat_stream(kb_id, body.query, top_k=body.top_k),
        media_type="text/event-stream",
    )
```

- [ ] **Step 2: 创建 app/main.py**

```python
"""FastAPI application entry point with lifecycle management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core import close_es_client, init_es_client, get_embedder
from app.routers import chat_router, kb_router, search_router, upload_router

logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init on startup, cleanup on shutdown."""
    # Startup
    logger.info("Starting RAG-ES-Demo...")

    # Initialize ES
    await init_es_client()
    logger.info("Elasticsearch connected")

    # Preload embedding model
    try:
        get_embedder()
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG-ES-Demo...")
    await close_es_client()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """Factory function for creating the FastAPI app."""
    app = FastAPI(
        title="RAG-ES-Demo",
        description="智能化 RAG 问答系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS (开放给前端调试)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(kb_router)
    app.include_router(upload_router)
    app.include_router(search_router)
    app.include_router(chat_router)

    # 健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# uvicorn 入口
app = create_app()
```

- [ ] **Step 3: 创建 tests/test_routers/test_chat.py**

```python
"""Tests for SSE chat endpoint."""

import unittest.mock as mock
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


def test_chat_returns_404_for_missing_kb():
    from app.main import create_app
    app = create_app()
    app.router.lifespan_context = lambda app: _null_lifespan()
    client = TestClient(app)

    with mock.patch("app.routers.chat.get_kb_info") as mock_get:
        mock_get.return_value = None
        resp = client.post("/kb/nonexistent/chat", json={"query": "test"})
        assert resp.status_code == 404


async def _null_lifespan():
    yield
```

- [ ] **Step 4: 运行所有测试验证**

```bash
cd /root/workspace/code/mu/github/rag-es-demo
poetry run pytest tests/ -v --tb=short
```

Expected: All tests pass (chunker 5, retriever 5, ingest 1, router ~6).

---

## Task 13: README 文档

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README.md**

```markdown
# RAG-ES-Demo

基于 Elasticsearch 7.10 的中文 RAG 问答系统。

## 特性

- **多知识库管理** — 每个知识库对应独立 ES 索引
- **Markdown 智能分块** — 按标题层级（# / ## / ###）自动分割
- **两阶段召回 + RRF 融合** — BM25 + 向量(script_score) 检索，RRF 倒数排名融合
- **SSE 流式问答** — JSON 事件流 (sources → token → done)，实时显示引用文档
- **OpenAI 标准协议** — 兼容任何支持 OpenAI API 格式的云端 LLM

## 前置要求

### Elasticsearch + IK 分词器

确保 ES 7.10 已运行并安装 `analysis-ik` 插件：

```bash
# 如果还没安装 IK 插件：
docker exec -it <es-container> bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v7.10.0/elasticsearch-analysis-ik-7.10.0.zip
# 重启 ES
```

### Embedding 模型

本地模型需放置在 `data/models/bge-small-zh-v1.5/` 目录。

## 快速开始

### 1. 安装依赖

```bash
conda activate rag-es-demo
poetry install
```

### 2. 配置环境变量

复制并编辑 `.env` 文件：

```bash
# 修改 .env 中的 LLM_BASE_URL 和 LLM_API_KEY
```

### 3. 启动服务

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/kb` | 创建知识库 |
| GET | `/kb` | 列出所有知识库 |
| GET | `/kb/{kb_id}` | 知识库详情 |
| DELETE | `/kb/{kb_id}` | 删除知识库 |
| POST | `/kb/{kb_id}/upload` | 上传 Markdown 文件入库 |
| POST | `/kb/{kb_id}/search` | 检索（返回 JSON） |
| POST | `/kb/{kb_id}/chat` | SSE 智能问答（返回 text/event-stream） |

## 测试

```bash
poetry run pytest tests/ -v
```

## 架构

```
FastAPI → [IngestService | RetrieverService | ChatService]
                                    ↓
                    ES 7.10 (script_score) + 本地 Embedding + 云端 LLM
```
```

- [ ] **Step 5: 最终 Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and API documentation"
```

---

## 任务执行顺序建议

1. **Task 1-2**: 项目骨架 + 配置 + 数据模型 — 可立即验证（导入不报错）
2. **Task 3-4**: ES 客户端 + Embedder 单例 — 启动时初始化验证
3. **Task 5**: 分块器 + 测试 — 纯逻辑，无外部依赖
4. **Task 6-8**: 检索器 + RRF 测试 — 纯逻辑测试先跑通
5. **Task 9-10**: 服务层 — 连接各组件
6. **Task 11-12**: 路由 + Main — API 层组装
7. **Task 13**: README — 文档

每个 Task 完成后执行 `git commit`，保证每次提交都是可工作状态。

---

## 自审：Spec 覆盖检查

| Spec 需求 | 对应 Task | 状态 |
|-----------|----------|------|
| 按 Markdown 标题分块 | Task 5 | ✅ |
| BM25 + Vector + RRF 检索 | Task 6-8 | ✅ |
| ES 7.10 script_score | Task 7 | ✅ |
| 多知识库独立索引 | Task 3, 11 | ✅ |
| 文件入库 | Task 9, 11 | ✅ |
| SSE JSON 事件流 | Task 10, 12 | ✅ |
| OpenAI 标准协议 LLM | Task 10 | ✅ |
| .env 配置管理 | Task 1, 2 | ✅ |
| ES 健康检查 | Task 3, 12 | ✅ |
| 无鉴权 | 全部 | ✅ |
| top_k=5, pre_k=10, k=60 | Task 2, 8 | ✅ |
| Poetry 依赖管理 | Task 1 | ✅ |
| 测试策略 | Task 5, 8, 9, 11, 12 | ✅ |
| API 端点全覆盖 | Task 11, 12 | ✅ |

**无占位符、无 TBD、无遗漏。** 所有代码步骤均包含完整代码。
