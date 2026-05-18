# RAG-ES-Demo 智能化 RAG 问答系统设计文档

**日期**: 2026-05-18
**状态**: 待审批

---

## 1. 概述

构建基于 Elasticsearch 7.10 的中文 RAG 问答系统，支持多知识库管理、文件入库、两阶段检索（BM25 + 向量）与 RRF 排名融合，以及 SSE 流式智能问答。

### 技术栈

| 组件 | 选择 |
|------|------|
| 语言 | Python 3.10+ |
| Web 框架 | FastAPI + Uvicorn |
| 编排框架 | LangChain 0.3 |
| Embedding | bge-small-zh-v1.5（本地，512 维） |
| 大模型 | OpenAI 标准协议（云端 API） |
| 搜索引擎 | Elasticsearch 7.10.0（Docker，localhost:9200） |
| 包管理 | Poetry + conda 虚拟环境（rag-es-demo） |

### 核心需求

- 按 Markdown 标题层级分块
- 两阶段召回 + RRF 排名融合（BM25 和向量各取 top_k×2，RRF k=60，融合后取 top_k=5）
- ES 7.10 使用 `script_score` 暴力扫描进行向量检索
- 支持检索和智能问答两个独立功能
- 智能问答返回引用文档
- SSE 流式输出使用 JSON 事件格式（sources / token / done）
- 多知识库：每个知识库对应独立 ES 索引
- 无鉴权，开放 API
- 配置通过 `.env` 文件管理

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                    FastAPI                       │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  /upload  │  │ /search  │  │ /chat (SSE)   │  │
│  │  (POST)   │  │  (POST)  │  │   (POST)      │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │              │               │           │
│  ┌────▼──────────────▼───────────────▼────────┐  │
│  │            Service Layer                    │  │
│  │  IngestService  │  RetrieverService         │  │
│  │  (分块+入库)    │  (BM25+Vector+RRF)        │  │
│  │               ChatService                   │  │
│  │          (检索 → 构建Prompt → LLM)          │  │
│  └────────────┬────────────────┬───────────────┘  │
│               │                │                   │
└───────────────┼────────────────┼───────────────────┘
                │                │
        ┌───────▼───────┐  ┌────▼────────────┐
        │  ES 7.10.0    │  │  LLM (云端API)  │
        │  localhost    │  │  OpenAI标准协议  │
        │  :9200        │  │                 │
        └───────────────┘  └─────────────────┘
                ▲
                │
        ┌───────┴───────┐
        │  Embedding    │
        │  bge-small    │
        │  (本地模型)    │
        └───────────────┘
```

### 2.2 目录结构

```
rag-es-demo/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口，生命周期管理
│   ├── config.py            # pydantic-settings 读取 .env
│   ├── core/
│   │   ├── es_client.py     # ES 连接管理（AsyncElasticsearch 单例）
│   │   └── embedder.py      # 本地 Embedding 模型加载（单例）
│   ├── chunkers/
│   │   └── markdown_chunker.py  # 按标题层级分块
│   ├── retrievers/
│   │   ├── bm25_retriever.py    # BM25 文本检索
│   │   ├── vector_retriever.py  # 向量检索 (script_score)
│   │   └── hybrid_retriever.py  # RRF 融合 + 去重排序
│   ├── services/
│   │   ├── ingest_service.py    # 文件上传 → 分块 → 向量化 → 入库
│   │   └── chat_service.py      # 检索 → 构建Prompt → LLM 流式
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py    # 知识库 CRUD
│   │   ├── upload.py            # 文件上传入库
│   │   ├── search.py            # 纯检索 API
│   │   └── chat.py              # SSE 智能问答
│   └── schemas/
│       ├── request.py           # 请求体定义
│       └── response.py          # 响应体 + SSE 事件定义
├── data/
│   ├── models/                  # 本地 Embedding 模型（已存在）
│   └── uploads/                 # 临时上传文件存储
├── tests/
│   ├── conftest.py
│   ├── test_chunkers/
│   ├── test_retrievers/
│   ├── test_services/
│   └── test_routers/
├── .env                         # 环境变量配置
├── pyproject.toml               # Poetry 依赖管理
└── README.md
```

---

## 3. 核心组件设计

### 3.1 配置管理 (`app/config.py`)

使用 `pydantic-settings` 读取 `.env`，定义 `Settings` 类：

```python
class Settings(BaseSettings):
    # ES
    es_host: str = "localhost"
    es_port: int = 9200
    es_user: str = "elastic"
    es_password: str = "Cass@123456"

    # Embedding
    embedding_model_path: str = "data/models/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 32

    # LLM (OpenAI 标准协议)
    llm_base_url: str
    llm_api_key: str
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_timeout: int = 30

    # RRF
    rrf_k: int = 60
    retriever_top_k: int = 5
    retriever_pre_multiplier: int = 2  # 每路取 top_k * 2

    # Service
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # Upload
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50
```

### 3.2 Embedding 单例 (`app/core/embedder.py`)

- 启动时通过 `SentenceTransformer` 加载本地模型
- 提供 `encode(texts: list[str]) -> list[list[float]]` 方法
- 单例模式，避免重复加载
- 批量 encode，batch_size 由配置控制

### 3.3 ES 客户端 (`app/core/es_client.py`)

- 使用 `AsyncElasticsearch`（elasticsearch-py v8 兼容 ES 7.x）
- 提供 `get_es_client() -> AsyncElasticsearch` 单例获取
- 启动时执行健康检查（`ping()`），失败则阻止服务启动
- 关闭时优雅断开连接

### 3.4 Markdown 分块器 (`app/chunkers/markdown_chunker.py`)

按 `#` / `##` / `###` 标题层级分割 Markdown 内容：

**输入**：
```markdown
# 一级标题
内容...
## 二级标题
内容...
### 三级标题
内容...
```

**输出**（每个 Chunk 对象）：
```python
@dataclass
class Chunk:
    content: str          # 该段文本
    heading_path: str     # 层级路径 "一级标题 > 二级标题 > 三级标题"
    chunk_index: int      # 序号
```

不主动做重叠；标题段天然是一个语义单元。

### 3.5 检索器

#### BM25 检索器 (`app/retrievers/bm25_retriever.py`)

- 对 ES 索引执行 `match` 查询 `content` 字段
- 返回 `[(doc_id, content, metadata, bm25_score), ...]`

#### 向量检索器 (`app/retrievers/vector_retriever.py`)

- 使用 ES 7.10 `script_score` 查询：

```json
{
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": {
        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
        "params": { "query_vector": [/* 512维向量 */] }
      }
    }
  },
  "size": ${pre_k}
}
```

- `pre_k = top_k * 2 = 10`
- 返回 `[(doc_id, content, metadata, vector_score), ...]`

#### RRF 混合检索器 (`app/retrievers/hybrid_retriever.py`)

1. 并行调用 BM25 和向量检索，各取 `pre_k = 10` 条
2. 按 `doc_id` 去重
3. 计算 RRF 分数：`rrf_score = 1/(k + bm25_rank) + 1/(k + vector_rank)`
   - 若文档仅出现在一路，另一路 rank 视为无穷大（贡献 0）
4. 按 `rrf_score` 降序，取 `top_k = 5`

---

## 4. 数据流

### 4.1 知识库管理

```
POST /kb
  → 生成 UUID 作为 kb_id
  → 在 ES 创建索引 rag_kb_{kb_id}
  → 初始化 mapping（见 5.1）
  → 返回 {kb_id, index_name, created_at}

GET /kb
  → 列出所有 rag_kb_* 索引
  → 返回 [{kb_id, doc_count, created_at}, ...]

GET /kb/{kb_id}
  → 查询索引统计信息

DELETE /kb/{kb_id}
  → 删除 ES 索引 rag_kb_{kb_id}
```

### 4.2 文件入库流程

```
POST /kb/{kb_id}/upload
  Body: multipart/form-data, file=<*.md>
  │
  ├── 1. 读取 Markdown 内容
  ├── 2. MarkdownChunker.chunk(content) → List[Chunk]
  ├── 3. Embedder.encode([c.content for c in chunks]) → List[vector]
  └── 4. ES bulk index 到 rag_kb_{kb_id}
        每条文档:
        {
          "doc_id": "<uuid>",
          "filename": "上传文件名",
          "heading_path": "一级标题 > 二级标题",
          "chunk_index": 0,
          "content": "该段文本",
          "embedding": [0.1, 0.2, ...],
          "created_at": "2026-05-18T..."
        }
  → 返回 {doc_count, chunks_count}
```

### 4.3 检索流程

```
POST /kb/{kb_id}/search
  Body: { "query": "用户查询", "top_k": 5 }
  │
  ├── BM25Retriever.retrieve(query, kb_id, k=10)
  ├── VectorRetriever.retrieve(query, kb_id, k=10)
  └── HybridRetriever.rrf_merge(bm25, vector, k=60, top_k=5)
  → 返回 {
      "results": [
        {
          "doc_id": "...",
          "filename": "...",
          "heading_path": "...",
          "content": "...",
          "bm25_score": 0.5,
          "vector_score": 0.85,
          "rrf_score": 0.032
        }
      ]
    }
```

### 4.4 SSE 智能问答流程

```
POST /kb/{kb_id}/chat
  Body: { "query": "用户查询", "top_k": 5 }  // top_k 可选，默认 5
  Content-Type: text/event-stream
  │
  ├── 1. HybridRetriever.search(query, kb_id, top_k) → top 5 chunks
  ├── 2. SSE: {"type": "sources", "data": [...]}
  ├── 3. 构建 Prompt:
  │     System: 你是智能知识库助手，请严格根据以下参考资料回答问题。
  │             如果参考资料中没有相关信息，请明确说明。
  │     Context: [按序号列出每个 source 的内容]
  │     User: {query}
  ├── 4. ChatOpenAI.astream(prompt) → 逐 token 流式
  ├── 5. SSE: {"type": "token", "data": "文本片段"} (多次)
  └── 6. SSE: {"type": "done", "data": {"answer": "完整回答", "sources": [...]}}
```

**错误事件**：
```
{"type": "error", "data": {"message": "错误描述"}}
```

---

## 5. ES 索引 Mapping

### 5.1 知识库索引 (`rag_kb_{kb_id}`)

```json
{
  "mappings": {
    "properties": {
      "doc_id": { "type": "keyword" },
      "filename": { "type": "keyword" },
      "heading_path": { "type": "text", "analyzer": "ik_max_word" },
      "chunk_index": { "type": "integer" },
      "content": { "type": "text", "analyzer": "ik_max_word" },
      "embedding": { "type": "dense_vector", "dims": 512 },
      "created_at": { "type": "date" }
    }
  }
}
```

注意：ES 7.10 的 `dense_vector` 不支持 `index: true` + `similarity: cosine` 参数（该特性在 7.11+ 才引入），因此向量检索必须使用 `script_score`。

### 5.2 ik 分词器

依赖 `analysis-ik` 插件支持中文分词。如果未安装，BM25 的中文效果会显著下降。需在 README 中说明插件安装步骤。

---

## 6. 错误处理

| 场景 | HTTP 状态码 | 处理方式 |
|------|-----------|---------|
| ES 连接失败 | 503 | 启动时健康检查；运行时返回明确错误 |
| 知识库不存在 | 404 | 检索/问答/上传时校验索引存在性 |
| 上传非 .md 文件 | 400 | 校验扩展名 |
| Embedding 模型加载失败 | 500 | 启动时预加载，失败则服务不启动 |
| LLM API 超时/失败 | 502 | timeout=30s；SSE 发送 error 事件 |
| 查询过短（<2字符） | 400 | 前端校验 + 后端校验 |
| 文件过大（>50MB） | 413 | 中间件限制 |
| 并发冲突 | 429 | 可选：后续加限流 |

---

## 7. 测试策略

```
tests/
├── conftest.py                  # fixtures: mock ES, mock LLM, test client
├── test_chunkers/
│   └── test_markdown_chunker.py # 分块逻辑：标题分割、heading_path 生成
├── test_retrievers/
│   └── test_hybrid_retriever.py # RRF 算法：去重、分数计算、排序
├── test_services/
│   └── test_ingest_service.py   # 入库流程：mock embedder + mock ES
└── test_routers/
    ├── test_knowledge_base.py   # CRUD API
    ├── test_upload.py           # 上传 API
    ├── test_search.py           # 检索 API
    └── test_chat.py             # SSE 流式问答 API
```

- 纯逻辑单元测试（Chunker、RRF）无需外部依赖
- API 集成测试使用 `httpx.AsyncClient` + `TestClient`
- LLM 和 Embedding 调用全部 mock

---

## 8. 依赖配置 (pyproject.toml)

```toml
[tool.poetry]
name = "rag-es-demo"
version = "0.1.0"
description = "RAG demo with Elasticsearch and LangChain"

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.115"
uvicorn = "^0.34"
python-dotenv = "^1.0"
pydantic-settings = "^2.0"
elasticsearch = {extras = ["async"], version = "^8.0"}
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
```

---

## 9. 环境变量 (.env)

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

---

## 10. API 端点汇总

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/kb` | 创建知识库 |
| GET | `/kb` | 列出所有知识库 |
| GET | `/kb/{kb_id}` | 知识库详情 |
| DELETE | `/kb/{kb_id}` | 删除知识库 |
| POST | `/kb/{kb_id}/upload` | 上传 Markdown 文件入库 |
| POST | `/kb/{kb_id}/search` | 检索（返回 JSON） |
| POST | `/kb/{kb_id}/chat` | SSE 智能问答（返回 text/event-stream） |

---

## 11. 后续可扩展方向（不在当前范围内）

- 支持 PDF / Word 等非 Markdown 文件上传（需文档解析服务）
- ES 升级到支持原生 knn 的版本，或使用 nmslib/faiss 插件
- API 鉴权（JWT / API Key）
- 问答历史记录
- 知识库文档管理（查看、删除单个文档的 chunk）
- 多路召回可配置化（调节 BM25 权重、向量权重）
