"""
pdf_ingestion.schema

Data contracts shared across the pipeline. `Token` is the atomic unit
produced by the ingestion layer and consumed by regex_engine, models,
and pipeline (conflict resolution + redaction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True, slots=True)
class BBox:
    """PDF-space bounding box, origin top-left, units = PDF points."""

    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(slots=True)
class Token:
    """
    A single word-level token with its spatial coordinate and its
    offset into the page's reconstructed plain-text string.

    char_start/char_end index into PageTokens.page_text and are what
    let downstream NLP (spaCy tokenizer -> Bi-LSTM-CRF) map its own
    span predictions back onto this token list without a re-scan.
    """

    text: str
    bbox: BBox
    page_num: int          # 0-indexed
    block_no: int
    line_no: int
    word_no: int
    token_index: int       # 0-indexed position within the page's token list
    char_start: int        # inclusive offset into PageTokens.page_text
    char_end: int          # exclusive offset into PageTokens.page_text

    # Populated later in the pipeline; kept here so every stage shares
    # one mutable record instead of parallel dict lookups.
    label: str = "O"        # IOB2 tag, e.g. "B-NRIC", "I-PER", "O"
    source: str = ""        # "regex" | "model" | "" (unlabelled)
    confidence: float = 1.0  # model confidence; 1.0 for deterministic regex hits


@dataclass(slots=True)
class PageTokens:
    """All tokens for one page, plus the reconstructed text used for NER."""

    page_num: int
    page_width: float
    page_height: float
    tokens: List[Token] = field(default_factory=list)
    page_text: str = ""     # tokens joined by single spaces; see extractor.py

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class DocumentTokens:
    """Whole-document container returned by PDFIngestor.extract()."""

    source_path: str
    pages: List[PageTokens] = field(default_factory=list)

    def all_tokens(self):
        for page in self.pages:
            yield from page.tokens

    @property
    def num_pages(self) -> int:
        return len(self.pages)
