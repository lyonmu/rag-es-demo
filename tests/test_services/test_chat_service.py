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
