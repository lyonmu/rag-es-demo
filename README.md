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

复制并编辑 `.env` 文件，修改 `LLM_BASE_URL` 和 `LLM_API_KEY` 为你的云端 LLM 配置。

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
