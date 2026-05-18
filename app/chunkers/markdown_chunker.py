"""Markdown chunker that splits by heading hierarchy (# / ## / ###)."""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    heading_path: str
    chunk_index: int


# Match 1-6 level headings
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_markdown(text: str) -> list[Chunk]:
    """Split by Markdown heading levels.

    Each heading and its content (until the next same-level or higher heading) becomes a chunk.
    heading_path records the hierarchy path, e.g. "一级标题 > 二级标题".
    """
    lines = text.split("\n")
    chunks: list[Chunk] = []

    # Collect all heading lines and their positions
    headings: list[tuple[int, int, str]] = []  # (line_index, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((i, level, title))

    if not headings:
        # Document without headings: entire content as one chunk
        content = text.strip()
        if content:
            chunks.append(Chunk(content=content, heading_path="", chunk_index=0))
        return chunks

    # Split content by headings
    for idx, (line_idx, level, title) in enumerate(headings):
        # Determine current heading's range
        start = line_idx
        if idx + 1 < len(headings):
            end = headings[idx + 1][0]
        else:
            end = len(lines)

        section_lines = lines[start:end]
        content = "\n".join(section_lines).strip()

        # Build heading_path: collect all parent headings
        path_parts = _build_heading_path(headings, idx)
        heading_path = " > ".join(path_parts)

        chunks.append(Chunk(content=content, heading_path=heading_path, chunk_index=idx))

    return chunks


def _build_heading_path(
    headings: list[tuple[int, int, str]], current_idx: int
) -> list[str]:
    """Build hierarchy path from root to current heading."""
    _, current_level, current_title = headings[current_idx]
    path = [current_title]

    # Look upward for parent headings (find headings with smaller level)
    for i in range(current_idx - 1, -1, -1):
        _, level, title = headings[i]
        if level < current_level:
            path.insert(0, title)
            current_level = level

    return path
