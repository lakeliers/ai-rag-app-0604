import re
from dataclasses import dataclass


@dataclass
class ChunkCandidate:
    text: str
    chunk_type: str = "child"
    parent_id: str = ""
    parent_text: str = ""


def split_text_fixed(text, chunk_size=500, chunk_overlap=80):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = start + chunk_size - chunk_overlap

    return chunks


def build_section_prefix(section):
    prefix_parts = []
    if section.section_title:
        prefix_parts.append(f"标题：{section.section_title}")
    if section.page:
        prefix_parts.append(f"页码：{section.page}")
    if section.sheet:
        prefix_parts.append(f"工作表：{section.sheet}")
    if section.row_start is not None:
        if section.row_end is not None and section.row_end != section.row_start:
            prefix_parts.append(f"行号：{section.row_start}-{section.row_end}")
        else:
            prefix_parts.append(f"行号：{section.row_start}")
    return "\n".join(prefix_parts)


def format_section_text(section):
    prefix = build_section_prefix(section)
    text = section.text.strip()
    if prefix:
        return f"{prefix}\n{text}"
    return text


def add_overlap(chunks, chunk_overlap):
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for index in range(1, len(chunks)):
        previous_tail = chunks[index - 1][-chunk_overlap:].strip()
        current = chunks[index]
        if previous_tail and not current.startswith(previous_tail):
            current = f"{previous_tail}\n{current}"
        overlapped.append(current)
    return overlapped


def merge_parts(parts, separator, chunk_size):
    chunks = []
    current = ""

    for part in parts:
        candidate = part if not current else f"{current}{separator}{part}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = part

    if current:
        chunks.append(current)

    return chunks


def split_text_recursive(text, chunk_size=500, chunk_overlap=80, separators=None):
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，", ""]
    if not separators:
        return split_text_fixed(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    separator = separators[0]
    if separator == "":
        return split_text_fixed(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    parts = [part.strip() for part in text.split(separator) if part.strip()]
    if len(parts) <= 1:
        return split_text_recursive(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators[1:],
        )

    split_parts = []
    for part in parts:
        if len(part) > chunk_size:
            split_parts.extend(split_text_recursive(
                part,
                chunk_size=chunk_size,
                chunk_overlap=0,
                separators=separators[1:],
            ))
        else:
            split_parts.append(part)

    chunks = merge_parts(split_parts, separator, chunk_size)
    return add_overlap(chunks, chunk_overlap)


def split_table_section(section, chunk_size=500):
    prefix = build_section_prefix(section)
    lines = [line.strip() for line in section.text.splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0]
    rows = lines[1:] or []
    base_prefix = "\n".join(part for part in [prefix, f"表头：{header}"] if part)
    chunks = []
    current_rows = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate = f"{base_prefix}\n" + "\n".join(candidate_rows)
        if len(candidate) <= chunk_size or not current_rows:
            current_rows = candidate_rows
            continue

        chunks.append(f"{base_prefix}\n" + "\n".join(current_rows))
        current_rows = [row]

    if current_rows:
        chunks.append(f"{base_prefix}\n" + "\n".join(current_rows))

    if not rows:
        chunks.append(base_prefix)

    normalized_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.4:
            normalized_chunks.append(chunk)
        else:
            normalized_chunks.extend(split_text_recursive(chunk, chunk_size=chunk_size, chunk_overlap=0))
    return normalized_chunks


def parent_id_for_section(source, section_index, section):
    title = section.section_title or section.content_type or "section"
    safe_title = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", title)[:80]
    return f"{source}:{section_index}:{safe_title}"


def chunk_section(section, source="", section_index=0, chunk_size=500, chunk_overlap=80):
    parent_text = format_section_text(section)
    parent_id = parent_id_for_section(source, section_index, section)

    if section.content_type == "table":
        texts = split_table_section(section, chunk_size=chunk_size)
    else:
        texts = split_text_recursive(parent_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return [
        ChunkCandidate(
            text=text,
            chunk_type="child",
            parent_id=parent_id,
            parent_text=parent_text,
        )
        for text in texts
    ]
