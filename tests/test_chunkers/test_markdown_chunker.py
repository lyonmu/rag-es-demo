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


def test_long_heading_section_splits_with_overlap(monkeypatch):
    from app.chunkers import markdown_chunker

    monkeypatch.setattr(markdown_chunker.settings, "chunk_max_chars", 80)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_overlap_chars", 10)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_min_chars", 20)

    body = "".join(str(i % 10) for i in range(150))
    text = "# 长章节\n\n" + body
    chunks = chunk_markdown(text)

    assert len(chunks) >= 2
    assert all(chunk.heading_path == "长章节" for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].content[-10:] == chunks[1].content[:10]


def test_small_tail_merges_into_previous_chunk(monkeypatch):
    from app.chunkers import markdown_chunker

    monkeypatch.setattr(markdown_chunker.settings, "chunk_max_chars", 80)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_overlap_chars", 10)
    monkeypatch.setattr(markdown_chunker.settings, "chunk_min_chars", 30)

    body = "".join(str(i % 10) for i in range(95))
    text = "# 长章节\n\n" + body
    chunks = chunk_markdown(text)

    assert len(chunks) == 1
    assert chunks[0].heading_path == "长章节"
    expected = "# 长章节\n\n" + body
    assert chunks[0].content == expected
