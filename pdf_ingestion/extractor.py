"""
pdf_ingestion.extractor

Tier-0 of the pipeline: turns a PDF into a word-level Token Stream +
parallel Spatial Coordinate Table, entirely offline via PyMuPDF (fitz).

Design notes:
- Uses page.get_text("words") — already block/line/word ordered and
  gives (x0, y0, x1, y1, word, block_no, line_no, word_no) per token.
  This is the cheapest reliable source of a spatial coordinate table;
  "rawdict" is available via extract_raw_spans() below for cases where
  sub-word (glyph/span) byte-level precision is needed later.
- page_text is built with a single forward pass (O(n) in token count),
  each token's char_start/char_end recorded as it's appended. This is
  the alignment table spaCy/the Bi-LSTM-CRF tokenizer must key off of
  instead of re-searching page_text per token (an O(n*m) trap when
  tokens repeat, e.g. "Jalan", "Taman", common Malay words).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from pdf_ingestion.schema import BBox, DocumentTokens, PageTokens, Token

logger = logging.getLogger(__name__)


class PDFIngestionError(RuntimeError):
    """Raised for unreadable, encrypted, or empty-text PDFs."""


class PDFIngestor:
    """Loads a PDF and produces a DocumentTokens token stream."""

    def __init__(self, min_word_len: int = 1):
        self.min_word_len = min_word_len

    def extract(self, pdf_path: str | Path) -> DocumentTokens:
        pdf_path = str(pdf_path)
        doc = self._open(pdf_path)

        try:
            document = DocumentTokens(source_path=pdf_path)
            for page in doc:
                document.pages.append(self._extract_page(page))
            return document
        finally:
            doc.close()

    # -- internals -----------------------------------------------------

    def _open(self, pdf_path: str) -> fitz.Document:
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # fitz raises generic RuntimeError/ValueError
            raise PDFIngestionError(f"Could not open PDF: {pdf_path}") from exc

        if doc.is_encrypted:
            # Try an empty-password unlock (common for "restricted" not
            # "protected" exports); otherwise refuse to proceed silently.
            if not doc.authenticate(""):
                doc.close()
                raise PDFIngestionError(
                    f"PDF is password-protected, cannot ingest: {pdf_path}"
                )
        return doc

    def _extract_page(self, page: fitz.Page) -> PageTokens:
        words = page.get_text("words")  # list of (x0,y0,x1,y1,text,block,line,word)

        page_tokens = PageTokens(
            page_num=page.number,
            page_width=page.rect.width,
            page_height=page.rect.height,
        )

        text_parts: list[str] = []
        cursor = 0  # running char offset into the reconstructed page_text

        for idx, (x0, y0, x1, y1, text, block_no, line_no, word_no) in enumerate(
            words
        ):
            if len(text.strip()) < self.min_word_len:
                continue

            char_start = cursor
            char_end = cursor + len(text)

            token = Token(
                text=text,
                bbox=BBox(x0, y0, x1, y1),
                page_num=page.number,
                block_no=block_no,
                line_no=line_no,
                word_no=word_no,
                token_index=len(page_tokens.tokens),
                char_start=char_start,
                char_end=char_end,
            )
            page_tokens.tokens.append(token)
            text_parts.append(text)

            cursor = char_end + 1  # +1 accounts for the joining space below

        page_tokens.page_text = " ".join(text_parts)

        if not page_tokens.tokens:
            logger.warning(
                "Page %d produced zero tokens (image-only / scanned page? "
                "OCR is out of scope for this ingestion tier).",
                page.number,
            )

        return page_tokens

    def extract_raw_spans(self, pdf_path: str | Path, page_num: int) -> list[dict]:
        """
        Byte/span-level fallback for a single page, using 'rawdict'.
        Needed only when a downstream stage requires sub-word glyph
        boxes (e.g. redacting a partial token match); not used by the
        default word-level pipeline.
        """
        doc = self._open(str(pdf_path))
        try:
            page = doc[page_num]
            raw = page.get_text("rawdict")
            spans = []
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        spans.append(span)
            return spans
        finally:
            doc.close()


def iter_documents(pdf_paths: Iterator[str | Path]) -> Iterator[DocumentTokens]:
    """Convenience batch helper for offline directory-scale runs."""
    ingestor = PDFIngestor()
    for path in pdf_paths:
        try:
            yield ingestor.extract(path)
        except PDFIngestionError:
            logger.exception("Skipping unreadable PDF: %s", path)
            continue
