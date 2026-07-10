"""PDF/TXT parsing and heading-aware chunking.

Pipeline: parse page-by-page (never whole-doc in memory) → detect headings via
per-page font-size statistics → maintain a running section path → sentence-
boundary chunking (~CHUNK_TARGET_TOKENS with overlap, real tokenizer counts) →
retrieval-friendly acronym expansion. Embedding/upserting lives in core.index.
"""

import logging
import re
from dataclasses import dataclass, field
from statistics import median
from typing import Iterator

import fitz  # PyMuPDF

from core.config import settings
from core.models import count_tokens

logger = logging.getLogger("ingestion")

# Expand standalone procurement acronyms so BM25 matches both forms.
# Chunk text only — source documents are never modified.
_ACRONYMS: dict[str, str] = {
    "PO": "Purchase Order",
    "POs": "Purchase Orders",
    "PR": "Purchase Requisition",
    "PRs": "Purchase Requisitions",
    "BPA": "Blanket Purchase Agreement",
    "BPAs": "Blanket Purchase Agreements",
    "RFQ": "Request for Quotation",
    "RFQs": "Requests for Quotation",
    "RFP": "Request for Proposal",
    "RFPs": "Requests for Proposal",
}
# Skip tokens already wrapped in parens: "Purchase Order (PO)" stays as-is.
_ACRONYM_RE = re.compile(r"(?<!\()\b(" + "|".join(_ACRONYMS) + r")\b(?!\))")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•\-])")


def expand_acronyms(text: str) -> str:
    return _ACRONYM_RE.sub(lambda m: f"{_ACRONYMS[m.group(1)]} ({m.group(1)})", text)


@dataclass
class Block:
    """A run of body text under one section path on one page."""

    section_path: str
    page: int  # 1-based
    text: str


@dataclass
class Chunk:
    text: str
    section_path: str
    page_start: int
    page_end: int
    chunk_index: int = 0


@dataclass
class _Heading:
    size: float
    title: str


class _SectionTracker:
    """Running section path built from heading font sizes (bigger = higher level)."""

    def __init__(self) -> None:
        self._stack: list[_Heading] = []

    def push(self, size: float, title: str) -> None:
        while self._stack and self._stack[-1].size <= size + 0.1:
            self._stack.pop()
        self._stack.append(_Heading(size, title))
        del self._stack[:-4]  # cap depth

    @property
    def path(self) -> str:
        return " > ".join(h.title for h in self._stack)


def parse_pdf_blocks(data: bytes) -> Iterator[Block]:
    """Stream (section_path, page, text) blocks from a PDF, page by page."""
    tracker = _SectionTracker()
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            spans: list[tuple[float, str]] = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if text:
                            spans.append((float(span["size"]), text))
            if not spans:
                continue
            page_median = median(size for size, _ in spans)
            threshold = settings.heading_font_ratio * page_median
            body: list[str] = []
            for size, text in spans:
                is_heading = size > threshold and len(text) < 120
                if is_heading:
                    if body:
                        yield Block(tracker.path, page_number, " ".join(body))
                        body = []
                    tracker.push(size, text)
                else:
                    body.append(text)
            if body:
                yield Block(tracker.path, page_number, " ".join(body))


def parse_txt_blocks(data: bytes) -> Iterator[Block]:
    text = data.decode("utf-8", errors="replace")
    for i, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        paragraph = paragraph.strip()
        if paragraph:
            yield Block("", 1, paragraph)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_blocks(blocks: Iterator[Block]) -> list[Chunk]:
    """Sentence-boundary-aware packing to ~CHUNK_TARGET_TOKENS with overlap,
    never splitting mid-sentence and never crossing section boundaries."""
    target = settings.chunk_target_tokens
    overlap_target = int(target * settings.chunk_overlap_ratio)
    chunks: list[Chunk] = []

    current: list[tuple[str, int, int]] = []  # (sentence, page, tokens)
    current_tokens = 0
    current_section = ""

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = " ".join(s for s, _, _ in current)
        pages = [p for _, p, _ in current]
        chunks.append(
            Chunk(
                text=text,
                section_path=current_section,
                page_start=min(pages),
                page_end=max(pages),
            )
        )
        # seed the next chunk with trailing sentences as overlap
        overlap: list[tuple[str, int, int]] = []
        overlap_tokens = 0
        for item in reversed(current):
            if overlap_tokens + item[2] > overlap_target:
                break
            overlap.insert(0, item)
            overlap_tokens += item[2]
        current = overlap
        current_tokens = overlap_tokens

    for block in blocks:
        if block.section_path != current_section:
            flush()
            # overlap must not leak across sections
            current = []
            current_tokens = 0
            current_section = block.section_path
        for sentence in _split_sentences(block.text):
            tokens = count_tokens(sentence)
            if current and current_tokens + tokens > target:
                flush()
            current.append((sentence, block.page, tokens))
            current_tokens += tokens
    flush()

    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        header = f"[{chunk.section_path}]\n" if chunk.section_path else ""
        chunk.text = header + expand_acronyms(chunk.text)
    return chunks


def parse_and_chunk(filename: str, data: bytes) -> tuple[list[Chunk], int]:
    """Return (chunks, page_count) for a PDF or TXT upload."""
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count
        chunks = chunk_blocks(parse_pdf_blocks(data))
    else:
        pages = 1
        chunks = chunk_blocks(parse_txt_blocks(data))
    logger.info("parsed %s: %d pages -> %d chunks", filename, pages, len(chunks))
    return chunks, pages
