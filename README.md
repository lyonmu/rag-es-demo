# RAG-ES-Demo

基于 Elasticsearch 7.10 的中文 RAG 问答系统。

## 特性

- **多知识库管理** — 每个知识库对应独立 ES 索引 + SQLite 元数据
- **Markdown 智能分块** — 按标题层级（# / ## / ###）自动分割
- **两阶段召回 + RRF 融合** — BM25 + 向量(script_score) 检索，RRF 倒数排名融合
- **SSE 流式问答** — JSON 事件流 (sources → token → done)，实时显示引用文档
- **OpenAI 标准协议** — 兼容任何支持 OpenAI API 格式的云端 LLM
- **统一响应结构** — 所有接口返回 HTTP 200 + `{code, message, data}`，5 位错误码

## 前置要求

### 1. 启动 Elasticsearch (Docker Compose)

项目提供 `docker-compose.yml`，一键启动 ES 7.10：

```bash
docker compose up -d
```

配置说明：
- 端口: `9200`
- 用户: `elastic` / 密码: `Cass@123456`
- 数据目录: `./data` 挂载到容器
- 插件目录: `./plugins` 挂载到容器（用于 IK 分词器）
- 内存: 2-3GB (`ES_JAVA_OPTS=-Xms2g -Xmx3g`)

停止服务：
```bash
docker compose down
```

### 2. 安装 IK 分词器

ES 7.10 需要 `analysis-ik` 插件才能正确进行中文分词。

#### 自动安装（推荐）

```bash
# 创建插件目录
mkdir -p plugins

# 下载 IK 分词器压缩包
curl -L -o plugins/analysis-ik.zip https://get.infini.cloud/elasticsearch/analysis-ik/7.10.0

# 解压到 plugins/analysis-ik/ 目录
unzip plugins/analysis-ik.zip -d plugins/analysis-ik/

# 确保权限正确
chmod -R 755 plugins/analysis-ik/
```

> 由于 `docker-compose.yml` 已将 `./plugins` 挂载到容器，解压后重启 ES 即可自动加载。

#### 重启 ES 加载插件

```bash
docker compose restart es
```

#### 验证 IK 分词器是否生效

```bash
# 检查已安装的插件列表
curl -s -u elastic:Cass@123456 http://localhost:9200/_cat/plugins?v

# 预期输出中包含 analysis-ik：
# name   component        version
# es     analysis-ik      7.10.0

# 测试分词效果
curl -s -u elastic:Cass@123456 -X POST "http://localhost:9200/_analyze?pretty" \
  -H "Content-Type: application/json" \
  -d '{"analyzer": "ik_max_word", "text": "北京清华大学"}'

# 预期分词结果：["北京", "清华", "清华大学", "华大", "大学"]
```

> 如果 `/_cat/plugins` 中没有 `analysis-ik`，说明插件未加载成功，请检查 `plugins/analysis-ik/` 目录结构是否正确。

### 3. Embedding 模型

本地模型已包含在仓库 `data/models/bge-small-zh-v1.5/` 目录，无需额外下载。

## 快速开始

### 1. 安装依赖

```bash
conda activate rag-es-demo
poetry install
```

### 2. 配置环境变量

复制并编辑 `.env` 文件：

```bash
cp .env.example .env
```

修改 `LLM_BASE_URL` 和 `LLM_API_KEY` 为你的云端 LLM 配置。

### 3. 启动服务

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API 端点

所有接口统一前缀 `/rag/api/v1`，返回格式 `{code, message, data}`，HTTP 状态码始终为 200。

| 方法   | 路径                                  | 描述                                   |
| ------ | ------------------------------------- | -------------------------------------- |
| POST   | `/rag/api/v1/kb`                      | 创建知识库                             |
| GET    | `/rag/api/v1/kb`                      | 列出所有知识库                         |
| GET    | `/rag/api/v1/kb/{kb_id}`              | 知识库详情                             |
| DELETE | `/rag/api/v1/kb/{kb_id}`              | 删除知识库                             |
| POST   | `/rag/api/v1/kb/{kb_id}/upload`       | 上传 Markdown/TXT 文件入库             |
| POST   | `/rag/api/v1/kb/{kb_id}/search`       | 检索（返回 JSON）                      |
| POST   | `/rag/api/v1/kb/{kb_id}/chat`         | SSE 智能问答（返回 text/event-stream） |

### 响应格式

```json
{
  "code": 10000,
  "message": "成功",
  "data": { ... }
}
```

### 错误码

| 代码    | 说明                   |
| ------- | ---------------------- |
| 10000   | 成功                   |
| 10001   | 通用失败               |
| 10002   | ES 连接失败            |
| 10003   | ES 创建索引失败        |
| 10004   | ES 操作失败            |
| 10100   | 仅支持 .md/.txt 文件   |
| 10101   | 知识库不存在           |
| 10102   | 知识库已存在           |
| 10103   | 知识库删除失败         |
| 10104   | 知识库创建失败         |
| 10200   | 检索失败               |
| 10201   | 查询参数无效           |
| 10300   | LLM 调用失败           |
| 10301   | 问答超时               |
| 10302   | 无参考资料             |

## 测试

```bash
poetry run pytest tests/ -v
```

## 架构

```
FastAPI → [KB Store (SQLite) | IngestService | RetrieverService | ChatService]
                                            ↓
                    ES 7.10 (script_score + IK) + 本地 Embedding + 云端 LLM
```
