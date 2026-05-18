"""Chat service for RAG-based Q&A with SSE streaming."""

import json
import logging
from typing import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.error_codes import SUCCESS, LLM_CALL_ERROR, get_error_message
from app.retrievers import hybrid_search
from app.schemas.response import ChatDoneData, ChatErrorData, SearchResultItem, SourceDoc

logger = logging.getLogger(__name__)


class ChatService:
    """Placeholder — full implementation in Task 10."""

    pass


SYSTEM_PROMPT = """你是智能知识库助手，请结合以下参考资料和你的专业知识回答问题。

回答要求：
1. 优先以参考资料为核心依据，确保准确性
2. 在参考资料基础上，可适当补充你的专业知识进行扩展说明
3. 如果参考资料与你的知识有冲突，以参考资料为准
4. 回答要结构化、有条理，使用中文

如果参考资料完全无法支撑问题，请先说明资料中的相关内容，再结合你的知识补充回答。"""


def _build_context(sources: list[SearchResultItem]) -> str:
    """Join search results into Prompt context."""
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


def _sse_event(event_type: str, content: dict | str) -> str:
    """Format an SSE event with unified {code, message, data} structure."""
    if event_type == "error":
        payload = {
            "code": LLM_CALL_ERROR,
            "message": get_error_message(LLM_CALL_ERROR),
            "data": {"type": "error", "content": content},
        }
    else:
        payload = {
            "code": SUCCESS,
            "message": get_error_message(SUCCESS),
            "data": {"type": event_type, "content": content},
        }
    return f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def chat_stream(
    kb_id: str,
    query: str,
    top_k: int = 5,
) -> AsyncIterator[str]:
    """SSE streaming Q&A.

    Yields:
        SSE formatted strings: sources, token, done, or error events.
    """
    try:
        # 1. Retrieve
        sources = await hybrid_search(kb_id, query, top_k=top_k)
        source_docs = _to_source_docs(sources)

        # 2. Build Prompt
        context = _build_context(sources)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{query}"),
        ]

        # 3. Initialize LLM
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,
            timeout=settings.llm_timeout,
        )

        # 4. Stream answer first
        answer_parts = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                answer_parts.append(chunk.content)
                yield _sse_event("token", chunk.content)

        # 5. Send done event with answer and sources
        full_answer = "".join(answer_parts)
        done_data = ChatDoneData(answer=full_answer, sources=source_docs)
        yield _sse_event("done", done_data.model_dump())

        # 6. Send sources at the end
        yield _sse_event("sources", [doc.model_dump() for doc in source_docs])

    except Exception as e:
        logger.exception("Chat stream error for kb=%s query=%s", kb_id, query)
        error_data = ChatErrorData(message=str(e))
        yield _sse_event("error", error_data.model_dump())
