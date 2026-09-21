"""
eval/synthetic_page.py

Evaluation needs to run the REAL pipeline (regex_engine.matcher +
pipeline.inference + pipeline.conflict_resolution), which all operate
on pdf_ingestion.schema.PageTokens/Token objects with real spatial
coordinates. Rendering every test sentence to an actual PDF just to
re-extract tokens would work but is slow at eval scale (thousands of
sentences) and adds PyMuPDF's own text-extraction as a confound.

This builds a PageTokens directly from a gold Sentence, with a
simple fixed-width layout (bboxes are fabricated, monotonically
increasing x0). char_start/char_end are computed exactly like
pdf_ingestion.extractor does, so downstream code (regex_engine's
char-span-to-token mapping) works identically to the real path.
"""

from __future__ import annotations

from pdf_ingestion.schema import BBox, PageTokens, Token
from models.dataset import Sentence

_CHAR_WIDTH = 6.0  # arbitrary but consistent fake glyph width, in points
_ROW_HEIGHT = 12.0


def sentence_to_page(sent: Sentence, page_num: int = 0) -> PageTokens:
    page = PageTokens(page_num=page_num, page_width=1000.0, page_height=200.0)

    x_cursor = 0.0
    char_cursor = 0
    text_parts = []

    for i, tok in enumerate(sent.tokens):
        width = max(len(tok), 1) * _CHAR_WIDTH
        bbox = BBox(x_cursor, 0.0, x_cursor + width, _ROW_HEIGHT)
        x_cursor += width + _CHAR_WIDTH  # gap between words, same spirit as a space

        char_start = char_cursor
        char_end = char_cursor + len(tok)
        char_cursor = char_end + 1  # +1 for the joining space, matches extractor.py

        page.tokens.append(
            Token(
                text=tok,
                bbox=bbox,
                page_num=page_num,
                block_no=0,
                line_no=0,
                word_no=i,
                token_index=i,
                char_start=char_start,
                char_end=char_end,
            )
        )
        text_parts.append(tok)

    page.page_text = " ".join(text_parts)
    return page
