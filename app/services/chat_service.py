"""Chat service for RAG-based Q&A with SSE streaming."""

import json
import logging
from typing import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.retrievers import hybrid_search
from app.schemas.response import ChatDoneData, ChatErrorData, SearchResultItem, SourceDoc

logger = logging.getLogger(__name__)


class ChatService:
    """Placeholder — full implementation in Task 10."""

    pass


SYSTEM_PROMPT = """你是智能知识库助手，请严格根据以下参考资料回答问题。
如果参考资料中没有相关信息，请明确说明"根据现有资料无法回答"。
回答要准确、简洁，使用中文。"""


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


def _sse_event(event_type: str, data: dict | str) -> str:
    """Format an SSE event string."""
    return f"event: message\ndata: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


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

        # 2. Send source docs
        source_docs = _to_source_docs(sources)
        yield _sse_event("sources", [doc.model_dump() for doc in source_docs])

        # 3. Build Prompt
        context = _build_context(sources)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{query}"),
        ]

        # 4. Initialize LLM
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,
            timeout=settings.llm_timeout,
        )

        # 5. Stream answer
        answer_parts = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                answer_parts.append(chunk.content)
                yield _sse_event("token", chunk.content)

        # 6. Send done event
        full_answer = "".join(answer_parts)
        done_data = ChatDoneData(answer=full_answer, sources=source_docs)
        yield _sse_event("done", done_data.model_dump())

    except Exception as e:
        logger.exception("Chat stream error for kb=%s query=%s", kb_id, query)
        error_data = ChatErrorData(message=str(e))
        yield _sse_event("error", error_data.model_dump())
