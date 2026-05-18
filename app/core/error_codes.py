"""5-digit error code definitions organized by module."""

# ── 100xx: System-level ──
SUCCESS = 10000
GENERAL_ERROR = 10001
ES_CONNECTION_ERROR = 10002
ES_CREATE_INDEX_ERROR = 10003
ES_OPERATION_ERROR = 10004

# ── 101xx: Knowledge Base Management ──
UNSUPPORTED_FILE_TYPE = 10100
KB_NOT_FOUND = 10101
KB_ALREADY_EXISTS = 10102
KB_DELETE_ERROR = 10103
KB_CREATE_ERROR = 10104

# ── 102xx: Search / Retrieval ──
SEARCH_ERROR = 10200
INVALID_SEARCH_QUERY = 10201

# ── 103xx: Chat / Q&A ──
LLM_CALL_ERROR = 10300
CHAT_TIMEOUT = 10301
NO_SOURCES_FOUND = 10302

# ── Code-to-message mapping ──
ERROR_MESSAGES: dict[int, str] = {
    SUCCESS: "成功",
    GENERAL_ERROR: "操作失败",
    ES_CONNECTION_ERROR: "Elasticsearch 连接失败",
    ES_CREATE_INDEX_ERROR: "Elasticsearch 创建索引失败",
    ES_OPERATION_ERROR: "Elasticsearch 操作失败",
    UNSUPPORTED_FILE_TYPE: "仅支持上传 .md / .txt 文件",
    KB_NOT_FOUND: "知识库不存在",
    KB_ALREADY_EXISTS: "知识库已存在",
    KB_DELETE_ERROR: "知识库删除失败",
    KB_CREATE_ERROR: "知识库创建失败",
    SEARCH_ERROR: "检索失败",
    INVALID_SEARCH_QUERY: "查询参数无效",
    LLM_CALL_ERROR: "LLM 调用失败",
    CHAT_TIMEOUT: "问答超时",
    NO_SOURCES_FOUND: "未找到参考资料",
}


def get_error_message(code: int) -> str:
    """Return human-readable message for an error code."""
    return ERROR_MESSAGES.get(code, "未知错误")
